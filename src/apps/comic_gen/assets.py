import os
import uuid
import time
from typing import Dict, Any, List
from urllib.parse import quote
from .models import (
    Character,
    Scene,
    Prop,
    GenerationStatus,
    ImageAsset,
    ImageVariant,
    MAX_VARIANTS_PER_ASSET,
)
from ...models.image import WanxImageModel, ImageGenModel
from ...utils import get_logger
from ...utils.oss_utils import is_object_key

logger = get_logger(__name__)

# 角色形象审美化：网文原文常有“相貌普通/皮肤黑/不起眼”等描述，
# 直接照搬会生成不符合影视审美的形象。这里统一加“帅美/精致/干净皮肤”前缀。
_CHARACTER_QUALITY_TEMPLATE = (
    "Chinese 3D donghua character, cinematic semi-realistic render, {looks}, "
    "ancient Chinese costume with fabric folds, realistic skin and hair, "
    "expressive eyes, detailed face, Unreal Engine 5 style, best quality,"
)
_CHARACTER_EXTRA_NEGATIVE = (
    "ugly, plain unattractive face, deformed, asymmetric eyes, extra limbs, extra fingers, "
    "bad anatomy, malformed hands, old, wrinkled, dull skin, low quality, blurry, jpeg artifacts"
)


def cleanup_old_variants(image_asset: ImageAsset) -> None:
    """
    Enforce variant limit: keep at most MAX_VARIANTS_PER_ASSET non-favorited variants.
    Favorited variants are never removed.
    When over limit, remove oldest non-favorited variants first.
    """
    if not image_asset or not image_asset.variants:
        return

    favorited = [v for v in image_asset.variants if v.is_favorited]
    non_favorited = [v for v in image_asset.variants if not v.is_favorited]

    # Sort non-favorited by created_at (oldest first)
    non_favorited.sort(key=lambda v: v.created_at)

    # Keep only the most recent MAX_VARIANTS_PER_ASSET non-favorited
    if len(non_favorited) > MAX_VARIANTS_PER_ASSET:
        to_remove = len(non_favorited) - MAX_VARIANTS_PER_ASSET
        removed = non_favorited[:to_remove]
        non_favorited = non_favorited[to_remove:]
        for v in removed:
            logger.info(
                f"Auto-removed old variant: {v.id} (created_at: {v.created_at})"
            )

    # Rebuild variants list: favorited first, then non-favorited (newest first)
    non_favorited.reverse()  # Newest first
    image_asset.variants = favorited + non_favorited


# Aspect ratio to image size mapping
ASPECT_RATIO_TO_SIZE = {
    "9:16": "576*1024",  # Portrait
    "3:4": "768*1024",  # Portrait (mild)
    "1:1": "1024*1024",  # Square
    "4:3": "1024*768",  # Landscape (mild)
    "16:9": "1024*576",  # Landscape
}


class AssetGenerator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = WanxImageModel(self.config.get("model", {}))
        self._mulerouter_image_model = None
        self.output_dir = self.config.get("output_dir", "output/assets")
        # Last effective prompt per character id, exposed to the workbench so
        # the UI can show what was actually sent to the model.
        self.last_prompts: Dict[str, Dict[str, Any]] = {}

    def _generate_with_qa(self, prompt, output_path, expectation, **gen_kwargs):
        """Generate an image, then run VLM QA; optionally regenerate with fix terms."""
        from .image_qa import auto_qa_enabled, auto_qa_regen_enabled, verify_image

        style_ref_paths = gen_kwargs.pop("style_ref_paths", None)
        negative = gen_kwargs.get("negative_prompt") or ""
        attempts = 2 if auto_qa_regen_enabled() else 1
        image_path = None
        for attempt in range(attempts):
            gen_kwargs["negative_prompt"] = negative
            if style_ref_paths:
                gen_kwargs["ref_image_paths"] = style_ref_paths
            image_path, _ = self._get_model_for(gen_kwargs.get("model_name")).generate(
                prompt, output_path, **gen_kwargs
            )
            if not auto_qa_enabled():
                break
            verdict = verify_image(output_path, expectation, kind="character")
            if not verdict or verdict.get("ok"):
                break
            logger.warning("Character QA issues: %s", verdict.get("issues"))
            if attempt + 1 >= attempts:
                break
            fix = (verdict.get("fix_terms") or "").strip()
            if fix:
                negative = (negative + ", " + fix).strip(", ")
        return image_path

    def _crop_head_reference(self, image_path: str) -> str:
        """Crop a full-body render to a head/shoulders reference for the
        headshot img2img pass. Using the uncropped full body with high
        ref_strength makes the model reproduce a full-body composition even
        when the prompt asks for a close-up."""
        if not image_path or not os.path.exists(image_path):
            return image_path
        try:
            from PIL import Image

            im = Image.open(image_path).convert("RGB")
            w, h = im.size
            box = (int(w * 0.18), 0, int(w * 0.82), int(h * 0.46))
            crop = im.crop(box)
            if crop.width < 512:
                scale = 512 / max(1, crop.width)
                crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
            out = f"{image_path.rsplit('.', 1)[0]}_headref.png"
            crop.save(out)
            return out
        except Exception as e:
            logger.warning("Failed to crop head reference: %s", e)
            return image_path

    def _infer_disposition(self, character) -> str:
        """长期气质/面相倾向（不是临时表情）。

        Extraction intentionally omits facial expressions, so we infer a
        default from name/description/role. Family elders default to 和蔼;
        villains to 阴险; otherwise neutral/upright.
        """
        name = getattr(character, "name", "") or ""
        blob = f"{name} {getattr(character, 'description', '') or ''} {getattr(character, 'clothing', '') or ''}"
        role = (getattr(character, "role", "") or "").strip()
        if any(
            k in name for k in ("母", "父", "娘", "婶", "婆", "奶", "爷", "亲", "姨")
        ):
            return "kind"
        if any(
            k in blob
            for k in ("和蔼", "慈祥", "温和", "善良", "慈爱", "憨厚", "老实", "亲切")
        ):
            return "kind"
        if any(
            k in blob
            for k in (
                "阴险",
                "狡诈",
                "狠毒",
                "冷酷",
                "阴鸷",
                "邪恶",
                "魔道",
                "反派",
                "杀意",
                "凶残",
            )
        ):
            return "sinister"
        if any(k in blob for k in ("威严", "刚正", "正气", "庄重", "凛然")):
            return "dignified"
        if role == "主角":
            return "upright"
        return "neutral"

    def _disposition_anchors(self, disposition: str) -> tuple:
        if disposition == "kind":
            return (
                "kind gentle benevolent expression, warm soft eyes, gentle friendly smile, "
                "motherly/kind-hearted face, soft natural facial features",
                "sinister, evil grin, menacing, villainous, cruel, harsh angry face, sharp eyebrows",
            )
        if disposition == "sinister":
            return (
                "sinister sly expression, narrow cold eyes, subtle evil smirk, cunning face",
                "kind, friendly smile, innocent",
            )
        if disposition == "dignified":
            return (
                "dignified upright expression, calm firm gaze, composed serious face",
                "sinister, evil grin, villainous, menacing",
            )
        if disposition == "upright":
            return (
                "honest upright expression, calm confident gaze, natural friendly face",
                "sinister, evil grin, villainous, menacing, cruel",
            )
        return (
            "calm neutral friendly expression, natural relaxed face",
            "sinister, evil grin, menacing",
        )

    def _clothing_anchors(self, character) -> tuple:
        blob = f"{getattr(character, 'clothing', '') or ''} {getattr(character, 'description', '') or ''}"
        if any(
            k in blob
            for k in (
                "农家",
                "农妇",
                "村民",
                "朴素",
                "粗布",
                "布衣",
                "平民",
                "妇人",
                "村妇",
            )
        ):
            return (
                "wearing simple coarse linen peasant clothes, plain rural Chinese hanfu, "
                "cloth belt, humble village woman outfit, no jewelry",
                "ball gown, evening dress, wedding dress, noble robe, luxurious costume, "
                "ornate jewelry, earrings, tiara, crown, western dress, Victorian dress",
            )
        return ("", "")

    def _get_model_for(self, model_name: str) -> "ImageGenModel":
        """Route to the correct image adapter based on model name."""
        from ...models.local import local_enabled, get_local_image_model

        if local_enabled("image"):
            return get_local_image_model(self.config.get("model", {}))
        if model_name and model_name.startswith("gpt-image"):
            if self._mulerouter_image_model is None:
                from ...models.mulerouter import MuleRouterImageModel

                self._mulerouter_image_model = MuleRouterImageModel({})
            return self._mulerouter_image_model
        return self.model

    def generate_character(
        self,
        character: Character,
        generation_type: str = "all",
        prompt: str = "",
        positive_prompt: str = None,
        negative_prompt: str = "",
        batch_size: int = 1,
        model_name: str = None,
        i2i_model_name: str = None,
        size: str = None,
        reference_image_paths: list = None,
    ) -> Character:
        """
        Generates character assets based on generation_type.
        Types: 'full_body', 'three_view', 'headshot', 'all'
        """
        character.status = GenerationStatus.PROCESSING

        gender = (getattr(character, "gender", "") or "").lower()
        # Gender is often missing from extraction. Infer it from the name /
        # description so female characters don't drift to a male face under
        # the 3D style (e.g. 韩母 with no gender field → old man with beard).
        if not gender:
            _name = getattr(character, "name", "") or ""
            _blob = f"{_name} {getattr(character, 'description', '') or ''} {getattr(character, 'clothing', '') or ''}"
            if any(
                k in _name
                for k in (
                    "母",
                    "妇",
                    "娘",
                    "姐",
                    "妹",
                    "婆",
                    "姑",
                    "女",
                    "妻",
                    "妃",
                    "后",
                    "公主",
                    "侍女",
                )
            ):
                gender = "female"
            elif any(
                k in _name
                for k in (
                    "父",
                    "公",
                    "爷",
                    "叔",
                    "伯",
                    "兄",
                    "弟",
                    "男",
                    "汉",
                    "夫",
                    "和尚",
                    "道士",
                )
            ):
                gender = "male"
            elif any(
                k in _blob
                for k in (
                    "妇人",
                    "妇女",
                    "女子",
                    "少女",
                    "女人",
                    "妻子",
                    "母亲",
                    "姑娘",
                    "她",
                )
            ):
                gender = "female"
            elif any(
                k in _blob for k in ("男子", "少年", "男人", "丈夫", "父亲", "他")
            ):
                gender = "male"
        import re as _re

        _age_match = _re.search(r"\d+", str(getattr(character, "age", "") or ""))
        _age = int(_age_match.group()) if _age_match else None
        _young = _age is not None and _age <= 16
        if "男" in gender or "male" in gender:
            looks = "handsome youthful boy" if _young else "handsome masculine man"
        elif "女" in gender or "female" in gender:
            looks = "beautiful youthful girl" if _young else "beautiful feminine woman"
        else:
            looks = "handsome or beautiful"
        role = (getattr(character, "role", "") or "配角").strip()
        if "男" in gender or "male" in gender:
            plain_look = "young man" if _young else "man"
        elif "女" in gender or "female" in gender:
            plain_look = "young woman" if _young else "woman"
        else:
            plain_look = "person"

        # Infer age when the extraction left it blank, then add explicit age
        # anchors. Without these, SDXL/IP-Adapter drifts to a young attractive
        # face even when the description says 中年/老妇.
        _desc_blob = f"{getattr(character, 'description', '') or ''} {getattr(character, 'clothing', '') or ''}"
        age_value = _age
        if age_value is None:
            if any(k in _desc_blob for k in ("孩童", "小孩", "幼童")):
                age_value = 10
            elif any(k in _desc_blob for k in ("少女", "少年", "青年", "小伙", "姑娘")):
                age_value = 20
            elif any(
                k in _desc_blob
                for k in ("中年", "妇人", "妇女", "农夫", "农妇", "大叔", "大婶")
            ):
                age_value = 45
            elif any(
                k in _desc_blob
                for k in (
                    "老年",
                    "老妇",
                    "老头",
                    "老者",
                    "老翁",
                    "老婆婆",
                    "苍老",
                    "年长",
                    "年迈",
                    "花白",
                    "白发",
                    "银发",
                    "老丈",
                    "老汉",
                    "老先生",
                )
            ):
                age_value = 65
        age_positive = ""
        age_negative = ""
        if "女" in gender or "female" in gender:
            age_gender_word = "woman"
        elif "男" in gender or "male" in gender:
            age_gender_word = "man"
        else:
            age_gender_word = "person"
        if age_value is not None and age_value >= 60:
            age_positive = (
                f"elderly {age_gender_word}, 60s-70s, grey or white hair, deep age lines, "
                "weathered wrinkled skin, aged plain features"
            )
            age_negative = (
                "young, youthful, teenager, baby face, smooth flawless skin, "
                "idol, glamorous, heavy makeup, beauty filter"
            )
        elif age_value is not None and age_value >= 40:
            age_positive = (
                f"middle-aged {age_gender_word}, 40s-50s, mature face, subtle wrinkles, "
                "weathered sun-tanned skin, plain rustic features"
            )
            age_negative = (
                "young, youthful, teenager, baby face, smooth flawless skin, "
                "idol, glamorous, heavy makeup, beauty filter, delicate model face"
            )
        elif age_value is not None and age_value <= 16:
            age_positive = "teenager, youthful, soft features"
            age_negative = "adult, mature face, wrinkles, beard, stubble"

        raw_disposition = (getattr(character, "disposition", "") or "").strip()
        if any(
            k in raw_disposition
            for k in ("和蔼", "温和", "慈祥", "善良", "慈爱", "亲切")
        ):
            disposition = "kind"
        elif any(
            k in raw_disposition for k in ("阴险", "冷酷", "狡诈", "狠毒", "邪恶")
        ):
            disposition = "sinister"
        elif any(k in raw_disposition for k in ("刚正", "威严", "庄重", "凛然")):
            disposition = "dignified"
        else:
            disposition = self._infer_disposition(character)
        disposition_positive, disposition_negative = self._disposition_anchors(
            disposition
        )
        clothing_positive, clothing_negative = self._clothing_anchors(character)
        _blob_all = f"{getattr(character, 'description', '') or ''} {getattr(character, 'clothing', '') or ''}"
        if any(
            k in _blob_all
            for k in ("金发", "碧眼", "西域", "异族", "洋人", "西式", "欧洲")
        ):
            ethnicity_positive, ethnicity_negative = "", ""
        else:
            ethnicity_positive = (
                "Chinese East Asian person, East Asian facial features, black hair"
            )
            ethnicity_negative = (
                "caucasian, western, european, blonde hair, blue eyes, western costume"
            )

        role_negative = ""
        if role in ("龙套", "群演", "路人"):
            role_negative = (
                "glamorous idol, supermodel, heavy makeup, ornate jewelry, "
                "luxurious costume, delicate doll face, fashion model"
            )
        if "男" in gender or "male" in gender:
            gender_anchor = "1 adult male man"
        elif "女" in gender or "female" in gender:
            gender_anchor = (
                "1 adult female woman, feminine face, soft jawline, no facial hair"
            )
        else:
            gender_anchor = "1 person"
        if role in ("主角", "重要配角"):
            quality_prefix = _CHARACTER_QUALITY_TEMPLATE.format(looks=looks)
            extra_negative = _CHARACTER_EXTRA_NEGATIVE
        elif role in ("龙套", "群演", "路人"):
            quality_prefix = (
                f"Chinese 3D donghua background villager, ordinary plain-faced {plain_look}, "
                "simple ancient clothing, cinematic semi-realistic render,"
            )
            extra_negative = (
                "deformed, asymmetric eyes, extra limbs, extra fingers, bad anatomy, "
                "malformed hands, low quality, blurry, jpeg artifacts"
            )
        else:
            quality_prefix = (
                f"Chinese 3D donghua character, ordinary {plain_look}, cinematic semi-realistic render, "
                "natural proportions, ancient Chinese costume, realistic skin and hair, detailed face,"
            )
            extra_negative = _CHARACTER_EXTRA_NEGATIVE
        # IMPORTANT: gender guards must come FIRST — SDXL truncates prompts at
        # ~77 tokens, so anything appended after the long style negatives is lost.
        gender_negative = ""
        if "男" in gender or "male" in gender:
            gender_negative = (
                "female, woman, girl, feminine face, makeup, lipstick, earrings, "
                "flower hair ornament, hairpin, long eyelashes, delicate features"
            )
        elif "女" in gender or "female" in gender:
            gender_negative = (
                "male, man, boy, beard, mustache, moustache, five o'clock shadow, "
                "stubble, masculine jaw, adam's apple, male pattern baldness"
            )
        negative_prompt = ", ".join(
            part
            for part in (
                gender_negative,
                age_negative,
                ethnicity_negative,
                clothing_negative,
                disposition_negative,
                role_negative,
                "multiple people, two people, group, crowd, duplicate, clone, split screen",
                extra_negative,
                negative_prompt,
            )
            if part
        )

        # Default style suffix if not provided (None means use default, "" means no style)
        style_suffix = (
            positive_prompt
            if positive_prompt is not None
            else "cinematic lighting, movie still, 8k, highly detailed, realistic"
        )

        # Default size if not provided
        effective_size = size or "576*1024"  # Default to portrait for characters

        try:
            # === R2V: Single unified reference sheet (T2I only) ===
            if generation_type == "reference_sheet":
                effective_prompt = (
                    prompt
                    if prompt
                    else f"single character, solo, full body, three-quarter view of {character.name}. {(character.prompt_en or character.description)}. Standing, clean background, studio lighting."
                )
                if quality_prefix not in effective_prompt:
                    effective_prompt = f"{quality_prefix} {effective_prompt}"
                if positive_prompt and positive_prompt not in effective_prompt:
                    effective_prompt = f"{positive_prompt}, {effective_prompt}"
                if not effective_prompt.startswith(gender_anchor):
                    effective_prompt = f"{gender_anchor}, {effective_prompt}"
                if age_positive and age_positive not in effective_prompt:
                    effective_prompt = f"{age_positive}, {effective_prompt}"
                if (
                    disposition_positive
                    and disposition_positive not in effective_prompt
                ):
                    effective_prompt = f"{disposition_positive}, {effective_prompt}"
                if ethnicity_positive and ethnicity_positive not in effective_prompt:
                    effective_prompt = f"{ethnicity_positive}, {effective_prompt}"
                if clothing_positive and clothing_positive not in effective_prompt:
                    effective_prompt = f"{clothing_positive}, {effective_prompt}"

                # Framing must come first — local SDXL truncates long prompts,
                # and composition instructions at the end get dropped, which is
                # why results were cropped half-faces.
                lower_prompt = effective_prompt.lower()
                is_closeup = any(
                    k in lower_prompt
                    for k in ("close-up", "close up", "headshot", "头像")
                )
                if is_closeup:
                    framing = (
                        "head and shoulders close-up, entire face fully inside the frame, "
                        "no cropping"
                    )
                    crop_negative = (
                        "cropped face, face out of frame, half face, cut off head"
                    )
                else:
                    framing = (
                        "single character, full body from head to toe, entire subject fully "
                        "inside the frame, centered with generous margins, no cropping"
                    )
                    crop_negative = (
                        "cropped, cut off, out of frame, partial body, half body, "
                        "missing head, missing feet, head cut off, feet cut off, zoomed face only"
                    )
                effective_prompt = f"{framing}, {effective_prompt}"
                if crop_negative not in negative_prompt:
                    negative_prompt = f"{crop_negative}, {negative_prompt}"

                # Local SDXL truncates prompts (~77 tokens). A long assembled
                # prompt drops the 3D style + age at the tail, producing a young
                # 2D illustration. For local backends, send a compact prompt
                # with the critical anchors first and only the short entity
                # prompt appended.
                try:
                    from ...models.local import local_enabled

                    local_image = local_enabled("image")
                except Exception:
                    local_image = False
                if local_image:
                    if age_value is not None and age_value >= 60:
                        age_short = "elderly, 60s-70s, white or grey hair, deep wrinkles, aged face"
                    elif age_value is not None and age_value >= 40:
                        age_short = "middle-aged, 40s-50s, mature wrinkled face"
                    elif age_value is not None and age_value <= 16:
                        age_short = "teenager"
                    else:
                        age_short = "young adult"
                    disp_short = {
                        "kind": "kind gentle face, warm eyes, gentle smile",
                        "sinister": "sinister face, narrow cold eyes",
                        "dignified": "dignified composed face",
                        "upright": "honest upright face, calm gaze",
                        "neutral": "natural friendly face",
                    }.get(disposition, "natural friendly face")
                    clothing_short = (
                        clothing_positive.split(",")[0] if clothing_positive else ""
                    )
                    compact_parts = [
                        "Chinese 3D donghua cinematic render",
                        "single character, full body from head to toe, centered, no crop",
                        f"East Asian Chinese {age_gender_word}",
                        age_short,
                        disp_short,
                        clothing_short,
                    ]
                    effective_prompt = ", ".join(p for p in compact_parts if p)
                    if getattr(character, "prompt_en", None):
                        effective_prompt += f", {character.prompt_en[:100]}"
                    compact_negative = ", ".join(
                        p
                        for p in (
                            "multiple panels, collage, split screen, character sheet, "
                            "grid of faces, text, watermark, signature, border",
                            crop_negative,
                            "male, man, beard, mustache, caucasian, western, blonde, "
                            "blue eyes, young, baby face, ball gown, jewelry, 2D, "
                            "illustration, anime, cel shading",
                            "low quality, blurry, deformed, extra limbs, bad anatomy",
                        )
                        if p
                    )
                    negative_prompt = compact_negative

                logger.info(
                    "Reference sheet prompt [%s]: %s",
                    character.name,
                    effective_prompt[:600],
                )
                logger.info(
                    "Reference sheet negative [%s]: %s",
                    character.name,
                    negative_prompt[:400],
                )
                self.last_prompts[character.id] = {
                    "positive": effective_prompt,
                    "negative": negative_prompt,
                    "size": size or "576*1024",
                    "local": bool(local_image),
                }

                effective_size = size or "576*1024"

                successful_generations = 0
                last_error = ""
                for i in range(batch_size):
                    try:
                        variant_id = str(uuid.uuid4())
                        sheet_path = os.path.join(
                            self.output_dir,
                            "characters",
                            f"{character.id}_refsheet_{variant_id}.png",
                        )
                        os.makedirs(os.path.dirname(sheet_path), exist_ok=True)

                        self._generate_with_qa(
                            effective_prompt,
                            sheet_path,
                            f"{character.name}: {character.prompt_en or character.description}",
                            negative_prompt=negative_prompt,
                            model_name=model_name,
                            size=effective_size,
                            style_ref_paths=reference_image_paths,
                        )

                        rel_path = os.path.relpath(sheet_path, "output")

                        if not character.reference_sheet:
                            from .models import AssetUnit

                            character.reference_sheet = AssetUnit()

                        from .models import ImageVariant

                        variant = ImageVariant(
                            id=variant_id,
                            url=rel_path,
                            created_at=time.time(),
                            prompt_used=effective_prompt,
                        )
                        character.reference_sheet.image_variants.append(variant)

                        # Clicking "生成" should surface the fresh result even if
                        # an older variant was previously selected.
                        character.reference_sheet.selected_image_id = variant_id
                        character.image_url = rel_path

                        successful_generations += 1

                        # Upload to OSS if configured
                        try:
                            from ...utils.oss_utils import OSSImageUploader

                            uploader = OSSImageUploader()
                            if uploader.is_configured:
                                object_key = uploader.upload_file(
                                    sheet_path, sub_path="assets/characters"
                                )
                                if object_key:
                                    variant.url = object_key
                                    if (
                                        character.reference_sheet.selected_image_id
                                        == variant_id
                                    ):
                                        character.image_url = object_key
                        except Exception as e:
                            logger.error(
                                f"Failed to upload reference sheet to OSS: {e}"
                            )

                        if i < batch_size - 1:
                            time.sleep(1)
                    except Exception as e:
                        last_error = str(e)
                        logger.error(
                            f"Failed to generate reference sheet variant {i + 1}/{batch_size}: {e}"
                        )
                        continue

                if successful_generations == 0:
                    raise RuntimeError(f"生成失败：{last_error}")

                character.status = GenerationStatus.COMPLETED
                return character

            # 1. Full Body (Master)
            if generation_type in ["all", "full_body"]:
                # Use provided prompt or construct default
                if not prompt:
                    # Default prompt - no style included, emphasize clean background
                    # If there's a reference image (reverse generation), emphasize consistency
                    base_prompt = f"Full body character design of {character.name}, concept art. {(character.prompt_en or character.description)}. Standing pose, neutral expression, no emotion, looking at viewer. Clean white background, isolated, no other objects, no scenery, simple background, high quality, masterpiece."
                else:
                    base_prompt = prompt

                # Save the user's prompt WITHOUT style suffix
                if quality_prefix not in base_prompt:
                    base_prompt = f"{quality_prefix} {base_prompt}"
                character.full_body_prompt = base_prompt

                # Generate the image with style suffix appended
                generation_prompt = (
                    f"{gender_anchor}, {style_suffix}, {base_prompt}"
                    if style_suffix and style_suffix not in base_prompt
                    else base_prompt
                )

                # Check for base character reference (for variants)
                ref_image_path = None
                ref_same_type = False
                if character.base_character_id:
                    base_fullbody_path = os.path.join(
                        self.output_dir,
                        "characters",
                        f"{character.base_character_id}_fullbody.png",
                    )
                    if os.path.exists(base_fullbody_path):
                        ref_image_path = base_fullbody_path
                        ref_same_type = True

                # === REVERSE GENERATION: Check for uploaded images to use as reference ===
                # Priority: Three Views > Headshot (uploaded images)
                if not ref_image_path:
                    # Check for uploaded three_views
                    if character.three_view_asset:
                        uploaded_variant = next(
                            (
                                v
                                for v in character.three_view_asset.variants
                                if getattr(v, "is_uploaded_source", False)
                            ),
                            None,
                        )
                        if uploaded_variant:
                            ref_url = uploaded_variant.url
                            if is_object_key(ref_url):
                                ref_image_path = ref_url
                                logger.debug(
                                    f"Reverse generation: Using uploaded three_views as reference: {ref_url}"
                                )
                            else:
                                local_path = os.path.join("output", ref_url)
                                if os.path.exists(local_path):
                                    ref_image_path = local_path
                                    logger.debug(
                                        f"Reverse generation: Using local three_views as reference: {local_path}"
                                    )

                    # Check for uploaded headshot
                    if not ref_image_path and character.headshot_asset:
                        uploaded_variant = next(
                            (
                                v
                                for v in character.headshot_asset.variants
                                if getattr(v, "is_uploaded_source", False)
                            ),
                            None,
                        )
                        if uploaded_variant:
                            ref_url = uploaded_variant.url
                            if is_object_key(ref_url):
                                ref_image_path = ref_url
                                logger.debug(
                                    f"Reverse generation: Using uploaded headshot as reference: {ref_url}"
                                )
                            else:
                                local_path = os.path.join("output", ref_url)
                                if os.path.exists(local_path):
                                    ref_image_path = local_path
                                    logger.debug(
                                        f"Reverse generation: Using local headshot as reference: {local_path}"
                                    )

                # Batch Generation Loop
                successful_generations = 0
                last_error = ""
                for i in range(batch_size):
                    try:
                        variant_id = str(uuid.uuid4())
                        fullbody_path = os.path.join(
                            self.output_dir,
                            "characters",
                            f"{character.id}_fullbody_{variant_id}.png",
                        )
                        os.makedirs(os.path.dirname(fullbody_path), exist_ok=True)

                        # Select model: use I2I model (wan2.6-image) when reference image is provided
                        effective_model_name = model_name
                        effective_generation_prompt = generation_prompt
                        if ref_image_path:
                            # Override to I2I model when using reference image
                            effective_model_name = i2i_model_name or "wan2.6-image"
                            logger.debug(
                                f"Reverse generation: Using I2I model {effective_model_name} with reference image"
                            )

                            # Enhance prompt for reverse generation to emphasize reference consistency (only if not already present)
                            reverse_enhancement = "STRICTLY MAINTAIN the SAME character appearance, face, hairstyle, skin tone, and clothing as the reference image. "
                            if (
                                reverse_enhancement.strip()
                                not in effective_generation_prompt
                            ):
                                effective_generation_prompt = (
                                    f"{reverse_enhancement}{generation_prompt}"
                                )
                                logger.debug(
                                    f"Reverse generation enhanced prompt: {effective_generation_prompt[:100]}..."
                                )

                        self._generate_with_qa(
                            effective_generation_prompt,
                            fullbody_path,
                            f"{character.name}: {character.prompt_en or character.description}",
                            ref_image_path=ref_image_path,
                            negative_prompt=negative_prompt,
                            model_name=effective_model_name,
                            size=effective_size,
                            **(
                                {
                                    "mode": "img2img",
                                    "ref_strength": float(
                                        os.getenv("LOCAL_CHAR_IMG2IMG_STRENGTH", "0.6")
                                    ),
                                }
                                if ref_same_type
                                else {}
                            ),
                        )

                        rel_fullbody_path = os.path.relpath(fullbody_path, "output")

                        # Store in ImageAsset
                        if not character.full_body_asset:
                            from .models import ImageAsset

                            character.full_body_asset = ImageAsset()

                        from .models import ImageVariant

                        variant = ImageVariant(
                            id=variant_id,
                            url=rel_fullbody_path,
                            created_at=time.time(),
                            prompt_used=generation_prompt,
                        )
                        character.full_body_asset.variants.insert(
                            0, variant
                        )  # Prepend new variants

                        # Cleanup old variants (keep max 10 non-favorited)
                        cleanup_old_variants(character.full_body_asset)

                        # Auto-select if it's the first one or we want to update the view
                        if not character.full_body_asset.selected_id or batch_size == 1:
                            character.full_body_asset.selected_id = variant_id
                            character.full_body_image_url = (
                                rel_fullbody_path  # Legacy sync
                            )

                        successful_generations += 1
                        logger.debug(
                            f"Full body variant {i + 1}/{batch_size} generated successfully"
                        )

                        # Add small delay between API calls to avoid rate limiting (except for last one)
                        if i < batch_size - 1:
                            time.sleep(1)
                    except Exception as e:
                        last_error = str(e)
                        logger.error(
                            f"Failed to generate full body variant {i + 1}/{batch_size}: {e}"
                        )
                        continue

                    try:
                        from ...utils.oss_utils import OSSImageUploader

                        uploader = OSSImageUploader()
                        if uploader.is_configured:
                            object_key = uploader.upload_file(
                                fullbody_path, sub_path="assets/characters"
                            )
                            if object_key:
                                logger.debug(
                                    f"Uploaded full body variant {i + 1} to OSS: {object_key}"
                                )
                                variant.url = object_key
                                if character.full_body_asset.selected_id == variant.id:
                                    character.full_body_image_url = object_key
                    except Exception as e:
                        logger.error(
                            f"Failed to upload full body variant {i + 1} to OSS: {e}"
                        )

                logger.info(
                    f"Full body generation complete: {successful_generations}/{batch_size} variants generated"
                )
                character.full_body_updated_at = time.time()

                if successful_generations == 0:
                    raise RuntimeError(f"生成失败：{last_error}")

                # Mark downstream as inconsistent if generating only full body
                if generation_type == "full_body":
                    character.is_consistent = False

            # Ensure full body exists for derived assets
            # Use selected variant or legacy url
            current_full_body_url = character.full_body_image_url
            if character.full_body_asset and character.full_body_asset.selected_id:
                selected_variant = next(
                    (
                        v
                        for v in character.full_body_asset.variants
                        if v.id == character.full_body_asset.selected_id
                    ),
                    None,
                )
                if selected_variant:
                    current_full_body_url = selected_variant.url

            # === REVERSE GENERATION: Allow using uploaded images as reference if no full body ===
            # Check for uploaded images to use as reference when no full body exists
            uploaded_reference_url = None
            if not current_full_body_url:
                # Check for any uploaded image that can be used as reference
                # Priority: The asset type being generated > other types
                if generation_type == "three_view" and character.headshot_asset:
                    # If generating three_view, check for uploaded headshot
                    uploaded_variant = next(
                        (
                            v
                            for v in character.headshot_asset.variants
                            if getattr(v, "is_uploaded_source", False)
                        ),
                        None,
                    )
                    if uploaded_variant:
                        uploaded_reference_url = uploaded_variant.url
                        logger.debug(
                            f"Reverse generation: Will use uploaded headshot as reference for three_view"
                        )

                elif generation_type == "headshot" and character.three_view_asset:
                    # If generating headshot, check for uploaded three_views
                    uploaded_variant = next(
                        (
                            v
                            for v in character.three_view_asset.variants
                            if getattr(v, "is_uploaded_source", False)
                        ),
                        None,
                    )
                    if uploaded_variant:
                        uploaded_reference_url = uploaded_variant.url
                        logger.debug(
                            f"Reverse generation: Will use uploaded three_views as reference for headshot"
                        )

                # Also check own asset type for uploaded source
                if not uploaded_reference_url:
                    own_asset = (
                        character.three_view_asset
                        if generation_type == "three_view"
                        else character.headshot_asset
                    )
                    if own_asset:
                        uploaded_variant = next(
                            (
                                v
                                for v in own_asset.variants
                                if getattr(v, "is_uploaded_source", False)
                            ),
                            None,
                        )
                        if uploaded_variant:
                            uploaded_reference_url = uploaded_variant.url
                            logger.debug(
                                f"Reverse generation: Will use own uploaded image as reference"
                            )

            if (
                generation_type in ["three_view", "headshot"]
                and not current_full_body_url
                and not uploaded_reference_url
            ):
                raise ValueError(
                    "Full body image is required to generate derived assets. Upload an image or generate a full body first."
                )

            # Handle reference image path: could be OSS Object Key or local path
            # Prioritize full body, fall back to uploaded reference
            reference_url = current_full_body_url or uploaded_reference_url
            if reference_url:
                if is_object_key(reference_url):
                    # OSS Object Key - pass directly, image.py will handle signing
                    fullbody_path = reference_url
                    logger.debug(f"Using OSS Object Key for reference: {reference_url}")
                else:
                    # Local relative path - prepend output directory
                    fullbody_path = os.path.join("output", reference_url)
                    logger.debug(f"Using local path for reference: {fullbody_path}")
            else:
                fullbody_path = None

            # 2. Three View Sheet (Derived)
            if generation_type in ["all", "three_view"]:
                if not prompt or generation_type == "all":
                    # Add reference consistency emphasis
                    base_prompt = f"Character Reference Sheet for {character.name}. {(character.prompt_en or character.description)}. Three-view character design: Front view, Side view, and Back view. STRICTLY MAINTAIN the SAME character appearance, face, hairstyle, and clothing as the reference image. Full body, standing pose, neutral expression. Consistent clothing and details across all views. Simple white background, clean lines, studio lighting, high quality."
                else:
                    base_prompt = prompt

                # Save the user's prompt WITHOUT style suffix
                if quality_prefix not in base_prompt:
                    base_prompt = f"{quality_prefix} {base_prompt}"
                character.three_view_prompt = base_prompt

                # Generate with style suffix appended
                generation_prompt = (
                    f"{gender_anchor}, {style_suffix}, {base_prompt}"
                    if style_suffix and style_suffix not in base_prompt
                    else base_prompt
                )

                sheet_negative = (
                    negative_prompt
                    + ", background, scenery, landscape, shadows, complex background, text, watermark, messy, distorted, extra limbs"
                )

                successful_generations = 0
                last_error = ""
                for i in range(batch_size):
                    try:
                        variant_id = str(uuid.uuid4())
                        sheet_path = os.path.join(
                            self.output_dir,
                            "characters",
                            f"{character.id}_sheet_{variant_id}.png",
                        )

                        self._generate_with_qa(
                            generation_prompt,
                            sheet_path,
                            f"{character.name}: {character.prompt_en or character.description}",
                            ref_image_path=fullbody_path,
                            negative_prompt=sheet_negative,
                            ref_strength=0.8,
                            model_name=i2i_model_name,
                        )

                        rel_sheet_path = os.path.relpath(sheet_path, "output")

                        if not character.three_view_asset:
                            from .models import ImageAsset

                            character.three_view_asset = ImageAsset()

                        from .models import ImageVariant

                        variant = ImageVariant(
                            id=variant_id,
                            url=rel_sheet_path,
                            created_at=time.time(),
                            prompt_used=generation_prompt,
                        )
                        character.three_view_asset.variants.insert(0, variant)

                        # Cleanup old variants (keep max 10 non-favorited)
                        cleanup_old_variants(character.three_view_asset)

                        if (
                            not character.three_view_asset.selected_id
                            or batch_size == 1
                        ):
                            character.three_view_asset.selected_id = variant_id
                            character.three_view_image_url = (
                                rel_sheet_path  # Legacy sync
                            )
                            character.image_url = rel_sheet_path  # Legacy mapping

                        successful_generations += 1
                        logger.debug(
                            f"Three view variant {i + 1}/{batch_size} generated successfully"
                        )

                        if i < batch_size - 1:
                            time.sleep(1)
                    except Exception as e:
                        last_error = str(e)
                        logger.error(
                            f"Failed to generate three view variant {i + 1}/{batch_size}: {e}"
                        )
                        continue

                    # Try uploading to OSS if configured - store Object Key (not full URL)
                    try:
                        from ...utils.oss_utils import OSSImageUploader

                        uploader = OSSImageUploader()
                        if uploader.is_configured:
                            object_key = uploader.upload_file(
                                sheet_path, sub_path="assets/characters"
                            )
                            if object_key:
                                logger.debug(
                                    f"Uploaded three view variant {i + 1} to OSS: {object_key}"
                                )
                                variant.url = object_key
                                if character.three_view_asset.selected_id == variant.id:
                                    character.three_view_image_url = object_key
                                    character.image_url = object_key
                    except Exception as e:
                        logger.error(
                            f"Failed to upload three view variant {i + 1} to OSS: {e}"
                        )

                logger.info(
                    f"Three view generation complete: {successful_generations}/{batch_size} variants generated"
                )
                character.three_view_updated_at = time.time()

                # Raise exception if all variants failed
                if successful_generations == 0:
                    raise RuntimeError(f"生成失败：{last_error}")

            # 3. Headshot (Derived)
            if generation_type in ["all", "headshot"]:
                if not prompt or generation_type == "all":
                    # Add reference consistency emphasis
                    base_prompt = f"Close-up portrait of the SAME character {character.name}. {(character.prompt_en or character.description)}. STRICTLY MAINTAIN the SAME face, hairstyle, skin tone, and facial features as the reference image. Zoom in on face and shoulders, detailed facial features, neutral expression, looking at viewer, high quality, masterpiece."
                else:
                    base_prompt = prompt

                # Save the user's prompt WITHOUT style suffix
                if quality_prefix not in base_prompt:
                    base_prompt = f"{quality_prefix} {base_prompt}"
                character.headshot_prompt = base_prompt

                # Generate with style suffix appended
                generation_prompt = (
                    f"{gender_anchor}, {style_suffix}, {base_prompt}"
                    if style_suffix and style_suffix not in base_prompt
                    else base_prompt
                )

                successful_generations = 0
                last_error = ""
                for i in range(batch_size):
                    try:
                        variant_id = str(uuid.uuid4())
                        avatar_path = os.path.join(
                            self.output_dir,
                            "characters",
                            f"{character.id}_avatar_{variant_id}.png",
                        )

                        self._generate_with_qa(
                            generation_prompt,
                            avatar_path,
                            f"{character.name}: {character.prompt_en or character.description}",
                            ref_image_path=self._crop_head_reference(fullbody_path),
                            negative_prompt=negative_prompt,
                            ref_strength=0.55,
                            model_name=i2i_model_name,
                        )

                        rel_avatar_path = os.path.relpath(avatar_path, "output")

                        if not character.headshot_asset:
                            from .models import ImageAsset

                            character.headshot_asset = ImageAsset()

                        from .models import ImageVariant

                        variant = ImageVariant(
                            id=variant_id,
                            url=rel_avatar_path,
                            created_at=time.time(),
                            prompt_used=generation_prompt,
                        )
                        character.headshot_asset.variants.insert(0, variant)

                        # Cleanup old variants (keep max 10 non-favorited)
                        cleanup_old_variants(character.headshot_asset)

                        if not character.headshot_asset.selected_id or batch_size == 1:
                            character.headshot_asset.selected_id = variant_id
                            character.headshot_image_url = (
                                rel_avatar_path  # Legacy sync
                            )
                            character.avatar_url = rel_avatar_path  # Legacy mapping

                        successful_generations += 1
                        logger.debug(
                            f"Headshot variant {i + 1}/{batch_size} generated successfully"
                        )

                        if i < batch_size - 1:
                            time.sleep(1)
                    except Exception as e:
                        last_error = str(e)
                        logger.error(
                            f"Failed to generate headshot variant {i + 1}/{batch_size}: {e}"
                        )
                        continue

                    # Try uploading to OSS if configured - store Object Key (not full URL)
                    try:
                        from ...utils.oss_utils import OSSImageUploader

                        uploader = OSSImageUploader()
                        if uploader.is_configured:
                            object_key = uploader.upload_file(
                                avatar_path, sub_path="assets/characters"
                            )
                            if object_key:
                                logger.debug(
                                    f"Uploaded headshot variant {i + 1} to OSS: {object_key}"
                                )
                                variant.url = object_key
                                if character.headshot_asset.selected_id == variant.id:
                                    character.headshot_image_url = object_key
                                    character.avatar_url = object_key
                    except Exception as e:
                        logger.error(
                            f"Failed to upload headshot variant {i + 1} to OSS: {e}"
                        )

                logger.info(
                    f"Headshot generation complete: {successful_generations}/{batch_size} variants generated"
                )
                character.headshot_updated_at = time.time()

                # Raise exception if all variants failed
                if successful_generations == 0:
                    raise RuntimeError(f"生成失败：{last_error}")

            # Update consistency status (Legacy support, but also useful for quick checks)
            if generation_type == "all":
                character.is_consistent = True
            elif (
                character.three_view_updated_at >= character.full_body_updated_at
                and character.headshot_updated_at >= character.full_body_updated_at
            ):
                character.is_consistent = True

            character.status = GenerationStatus.COMPLETED

        except Exception as e:
            logger.error(f"Failed to generate character {character.name}: {e}")
            character.status = GenerationStatus.FAILED
            raise  # Re-raise to propagate error to caller

        return character

    def generate_scene(
        self,
        scene: Scene,
        positive_prompt: str = None,
        negative_prompt: str = "",
        batch_size: int = 1,
        model_name: str = None,
        size: str = None,
        reference_image_paths: list = None,
    ) -> Scene:
        """Generates a scene reference image."""
        scene.status = GenerationStatus.PROCESSING

        from .image_qa import (
            auto_qa_enabled,
            auto_qa_regen_enabled,
            lint_prompt,
            verify_image,
        )

        # Use provided prompts or fall back to default cinematic style
        if positive_prompt is None:
            positive_prompt = (
                "cinematic lighting, movie still, 8k, highly detailed, realistic"
            )

        # Default size for scenes (landscape)
        effective_size = size or "1024*576"

        scene_style = positive_prompt or ""
        for _bad in (
            "semi-realistic characters",
            "natural human proportions",
            "detailed realistic skin and hair strands",
            "characters",
        ):
            scene_style = scene_style.replace(_bad, "")
        scene_style = scene_style[:180]
        prompt = (
            f"Cinematic 3D donghua environment, semi-realistic render, grand xianxia immortal realm, floating mountains, "
            f"celestial palaces, glowing spell arrays, talismans, spirit qi, mist, "
            f"ancient Chinese architecture, timber-frame, tiled roof, paper windows, "
            f"no tatami, no shoji, no kimono. "
            f"empty ancient Chinese landscape, no people, no humans, no characters, "
            f"no portraits, no figures. {scene.name}. {(scene.prompt_en or scene.description)}. "
            f"{scene_style}, no modern elements, cinematic lighting, detailed environment, best quality."
        )
        negative_prompt = ", ".join(
            p
            for p in (
                negative_prompt,
                "modern road, asphalt, cars, vehicles, power lines, electric poles, "
                "modern buildings, concrete, plastic, 21st century, urban, "
                "people, person, human, crowd, portrait, character, figure, "
                "Japanese, tatami, shoji, engawa, kimono, Japanese castle, Japanese temple",
            )
            if p
        )
        prompt, _qa_neg = lint_prompt(prompt, "scene")
        if _qa_neg:
            negative_prompt = (negative_prompt + ", " + _qa_neg).strip(", ")

        try:
            for _ in range(batch_size):
                variant_id = str(uuid.uuid4())
                output_path = os.path.join(
                    self.output_dir, "scenes", f"{scene.id}_{variant_id}.png"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                gen_negative = negative_prompt
                attempts = 2 if auto_qa_regen_enabled() else 1
                image_path = None
                for _attempt in range(attempts):
                    image_path, _ = self._get_model_for(model_name).generate(
                        prompt,
                        output_path,
                        negative_prompt=gen_negative,
                        model_name=model_name,
                        size=effective_size,
                        **(
                            {"ref_image_paths": reference_image_paths}
                            if reference_image_paths
                            else {}
                        ),
                    )
                    if not auto_qa_enabled():
                        break
                    _verdict = verify_image(
                        output_path,
                        f"{scene.name}: {scene.prompt_en or scene.description}",
                        kind="scene",
                    )
                    if not _verdict or _verdict.get("ok"):
                        break
                    logger.warning("Scene QA issues: %s", _verdict.get("issues"))
                    if _attempt + 1 >= attempts:
                        break
                    _fix = (_verdict.get("fix_terms") or "").strip()
                    if _fix:
                        gen_negative = (gen_negative + ", " + _fix).strip(", ")

                rel_path = os.path.relpath(output_path, "output")

                if not scene.image_asset:
                    from .models import ImageAsset

                    scene.image_asset = ImageAsset()

                from .models import ImageVariant

                variant = ImageVariant(
                    id=variant_id,
                    url=rel_path,
                    created_at=time.time(),
                    prompt_used=prompt,
                )
                scene.image_asset.variants.insert(0, variant)

                if not scene.image_asset.selected_id or batch_size == 1:
                    scene.image_asset.selected_id = variant_id
                    scene.image_url = rel_path  # Legacy sync

                # Try uploading to OSS if configured - store Object Key (not full URL)
                try:
                    from ...utils.oss_utils import OSSImageUploader

                    uploader = OSSImageUploader()
                    if uploader.is_configured:
                        object_key = uploader.upload_file(
                            output_path, sub_path="assets/scenes"
                        )
                        if object_key:
                            logger.debug(f"Uploaded scene variant to OSS: {object_key}")
                            variant.url = object_key
                            if scene.image_asset.selected_id == variant.id:
                                scene.image_url = object_key
                except Exception as e:
                    logger.error(f"Failed to upload scene variant to OSS: {e}")

            scene.status = GenerationStatus.COMPLETED
        except Exception as e:
            logger.error(f"Failed to generate scene {scene.name}: {e}")
            scene.status = GenerationStatus.FAILED
            raise  # Re-raise to propagate error to caller

        return scene

    def generate_prop(
        self,
        prop: Prop,
        positive_prompt: str = None,
        negative_prompt: str = "",
        batch_size: int = 1,
        model_name: str = None,
        size: str = None,
        reference_image_paths: list = None,
    ) -> Prop:
        """Generates a prop reference image."""
        prop.status = GenerationStatus.PROCESSING

        # Use provided prompts or fall back to default cinematic style
        if positive_prompt is None:
            positive_prompt = (
                "cinematic lighting, movie still, 8k, highly detailed, realistic"
            )

        # Default size for props (square)
        effective_size = size or "1024*1024"

        prompt = (
            f"{positive_prompt}, ancient Chinese historical setting, no modern elements, "
            f"Prop Design: {prop.name}. {(prop.prompt_en or prop.description)}. Isolated on white background, "
            "high quality, detailed."
        )
        negative_prompt = ", ".join(
            p
            for p in (
                negative_prompt,
                "modern object, plastic, electric, 21st century, urban",
            )
            if p
        )

        try:
            for _ in range(batch_size):
                variant_id = str(uuid.uuid4())
                output_path = os.path.join(
                    self.output_dir, "props", f"{prop.id}_{variant_id}.png"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                image_path, _ = self._get_model_for(model_name).generate(
                    prompt,
                    output_path,
                    negative_prompt=negative_prompt,
                    model_name=model_name,
                    size=effective_size,
                    **(
                        {"ref_image_paths": reference_image_paths}
                        if reference_image_paths
                        else {}
                    ),
                )

                rel_path = os.path.relpath(output_path, "output")

                if not prop.image_asset:
                    from .models import ImageAsset

                    prop.image_asset = ImageAsset()

                from .models import ImageVariant

                variant = ImageVariant(
                    id=variant_id,
                    url=rel_path,
                    created_at=time.time(),
                    prompt_used=prompt,
                )
                prop.image_asset.variants.insert(0, variant)

                if not prop.image_asset.selected_id or batch_size == 1:
                    prop.image_asset.selected_id = variant_id
                    prop.image_url = rel_path  # Legacy sync

                # Try uploading to OSS if configured - store Object Key (not full URL)
                try:
                    from ...utils.oss_utils import OSSImageUploader

                    uploader = OSSImageUploader()
                    if uploader.is_configured:
                        object_key = uploader.upload_file(
                            output_path, sub_path="assets/props"
                        )
                        if object_key:
                            logger.debug(f"Uploaded prop variant to OSS: {object_key}")
                            variant.url = object_key
                            if prop.image_asset.selected_id == variant.id:
                                prop.image_url = object_key
                except Exception as e:
                    logger.error(f"Failed to upload prop variant to OSS: {e}")

            prop.status = GenerationStatus.COMPLETED
        except Exception as e:
            logger.error(f"Failed to generate prop {prop.name}: {e}")
            prop.status = GenerationStatus.FAILED
            raise  # Re-raise to propagate error to caller

        return prop
