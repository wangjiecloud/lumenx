/** Chinese display names for the English "chrome" step names used by
 *  StepHeader / StepPageHeader. Unknown names fall through unchanged. */
const STEP_NAME_ZH: Record<string, string> = {
    SCRIPT: "脚本",
    STYLE: "风格定调",
    "ART DIRECTION": "风格定调",
    ASSETS: "资产库",
    "ASSET LIBRARY": "资产库",
    CAST: "本集素材",
    STORYBOARD: "分镜",
    "STORYBOARD R2V": "分镜",
    MOTION: "动效生成",
    ASSEMBLY: "合成",
    VOICE: "配音",
    MIX: "终混",
    EXPORT: "导出",
};

export function localizeStepName(name: string): string {
    return STEP_NAME_ZH[(name || "").trim().toUpperCase()] || name;
}
