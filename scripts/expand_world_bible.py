import json
import pathlib
import re
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROXY = "http://10.99.99.2:9090/litellm/v1"
MODEL = "qwen3.8-flash"
HEADERS = {
    "Authorization": "Bearer placeholder",
    "Content-Type": "application/json",
    "User-Agent": "curl/8.0",
}
SID = "5acd134f-773c-4cd5-81e3-1803b62e9271"
EID = "9d79e29f-b56c-4051-81da-82124fa74615"
BASE = "http://127.0.0.1:17177"
PROGRESS = pathlib.Path("/tmp/bible_progress.txt")
SEGMENTS = 10


def log(msg):
    line = time.strftime("[%H:%M:%S] ") + str(msg)
    print(line, flush=True)
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def llm(prompt, max_tokens=2500, retries=3):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    for i in range(retries):
        try:
            r = requests.post(
                PROXY + "/chat/completions", headers=HEADERS, json=payload, timeout=150
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-zA-Z]*\n", "", text)
                text = re.sub(r"\n```$", "", text)
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start:end + 1]
            return json.loads(text)
        except Exception as exc:
            log("  llm retry %d: %s" % (i + 1, type(exc).__name__))
            time.sleep(3)
    return None


def main():
    text = (ROOT / "data/utf8/11_utf8.txt").read_text(encoding="utf-8")
    ms = list(
        re.finditer(r"^第[0-9零一二三四五六七八九十百千万]{1,8}章[^\n]*", text, re.M)
    )
    total = len(ms)
    log("chapters=%d segments=%d" % (total, SEGMENTS))

    keys = [
        "locations",
        "architecture",
        "costume",
        "props",
        "factions",
        "cultivation_terms",
        "creatures",
        "customs",
        "visual_motifs",
        "avoid_modern",
    ]
    merged = {k: {} for k in keys}

    step = max(1, total // SEGMENTS)
    for seg in range(SEGMENTS):
        start = seg * step
        if start >= total:
            break
        picks = [
            start,
            min(start + step // 2, total - 1),
            min(start + step - 1, total - 1),
        ]
        parts = []
        for i in picks:
            end = ms[i + 1].start() if i + 1 < total else len(text)
            parts.append(text[ms[i].start() : min(end, ms[i].start() + 1400)])
        sample = "\n\n".join(parts)
        prompt = (
            "只提取《凡人修仙传》这段文本里的世界观与美术要素，不要剧情摘要。只输出 JSON：\n"
            '{"locations":[],"architecture":[],"costume":[],"props":[],"factions":[],'
            '"cultivation_terms":[],"creatures":[],"customs":[],"visual_motifs":[],"avoid_modern":[]}\n'
            "每项为简短中文短语；没有就留空数组。\n\n" + sample
        )
        log("segment %d/%d start" % (seg + 1, SEGMENTS))
        data = llm(prompt)
        if not data:
            log("segment %d/%d failed" % (seg + 1, SEGMENTS))
            continue
        if not isinstance(data, dict):
            log("segment %d bad shape, skip" % (seg + 1))
            continue
        for k in keys:
            for item in data.get(k) or []:
                item = str(item).strip()
                if not item or len(item) > 40:
                    continue
                merged[k][item] = merged[k].get(item, 0) + 1
        log("segment %d/%d ok" % (seg + 1, SEGMENTS))
        time.sleep(1)

    # keep top items
    notes = {
        k: [x for x, _ in sorted(v.items(), key=lambda kv: -kv[1])[:40]]
        for k, v in merged.items()
    }
    (ROOT / "output/world_bible_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("notes collected, running final merge")

    final_prompt = (
        "根据以下《凡人修仙传》全书分段提炼的世界观要素，输出一份增强版世界观 bible JSON（中文），"
        "用于统一 AI 图像/视频生成。字段：era, worldview, cultivation_realm, geography[], factions[], "
        "architecture, costume, props_artifacts, creatures, customs, visual_style, forbidden[], "
        "positive_anchors[] (英文, 20-30 条), negative_anchors[] (英文, 12-20 条), scene_examples[], "
        "details {locations[], architecture_styles[], costume_items[], props_items[], faction_notes[], "
        "cultivation_notes[], creature_notes[], custom_notes[]}, summary。只输出 JSON。\n\n"
        + json.dumps(notes, ensure_ascii=False)
    )
    bible = llm(final_prompt, max_tokens=5000)
    if not bible:
        log("final merge failed; keeping notes only")
        return
    pathlib.Path(ROOT / "output/world_bible_v2.json").write_text(
        json.dumps(bible, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    existing = ROOT / "data/world_bible.json"
    if existing.exists():
        existing.replace(ROOT / "data/world_bible_v1.json")
    existing.write_text(
        json.dumps(bible, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        requests.put(BASE + "/series/" + SID, json={"world_bible": bible}, timeout=60)
        series = requests.get(BASE + "/series/" + SID, timeout=30).json()
        ad = series.get("art_direction") or {}
        style = ad.get("style_config") or {}
        pos = style.get("positive_prompt", "")
        neg = style.get("negative_prompt", "")
        if "no modern elements" not in pos:
            style["positive_prompt"] = (
                "ancient Chinese xianxia world, ancient China, no modern elements, "
                + pos
            )
        if "asphalt" not in neg:
            style["negative_prompt"] = (
                "modern road, asphalt, car, vehicle, power lines, modern building, concrete, "
                "plastic, modern clothing, glasses, " + neg
            )
        body = {
            "selected_style_id": style.get("id", "donghua_3d_cinematic"),
            "style_config": style,
            "custom_styles": [style],
            "ai_recommendations": [],
        }
        requests.post(
            BASE + "/projects/" + EID + "/art_direction/save", json=body, timeout=60
        )
        requests.put(BASE + "/series/" + SID, json={"art_direction": body}, timeout=60)
        pc = series.get("prompt_config") or {}
        line = "【世界观】古代东方修仙世界：木石建筑、粗布与长袍、灵石丹药飞剑符箓，严禁现代元素。\n"
        for k in (
            "storyboard_polish",
            "video_polish",
            "r2v_polish",
            "storyboard_extraction",
        ):
            if pc.get(k) and line not in pc[k]:
                pc[k] = line + pc[k]
        requests.put(BASE + "/series/" + SID + "/prompt_config", json=pc, timeout=60)
        log("world bible v2 applied")
    except Exception as exc:
        log("apply failed: %s" % exc)
    log("DONE")


if __name__ == "__main__":
    main()
