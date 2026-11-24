"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { Video, Sparkles, RefreshCw } from "lucide-react";
import { useProjectStore } from "@/store/projectStore";
import VideoCreator from "./VideoCreator";
import VideoSidebar from "./VideoSidebar";
import { api, VideoTask } from "@/lib/api";
import { resolveModelId } from "@/lib/modelCatalog";
import StepHeader from "@/components/shared/StepHeader";
import StepGuide from "@/components/shared/StepGuide";
import WorkflowActionButton from "@/components/shared/WorkflowActionButton";

export default function VideoGenerator() {
    const tStep = useTranslations("stepHeader");
    const currentProject = useProjectStore((state) => state.currentProject);
    const updateProject = useProjectStore((state) => state.updateProject);
    const [tasks, setTasks] = useState<VideoTask[]>([]);
    const [autoRunning, setAutoRunning] = useState(false);

    // Shared state for Remix functionality
    const [remixData, setRemixData] = useState<Partial<VideoTask> | null>(null);

    // Get default model from project settings
    const defaultI2vModel = resolveModelId(
        'i2v',
        currentProject?.model_settings?.i2v_model,
        'video_sidebar',
    );

    // Generation Params (Lifted State)
    const [params, setParams] = useState({
        resolution: "720p",
        duration: 5,
        seed: undefined as number | undefined,
        generateAudio: true,  // Default to AI Sound enabled
        audioUrl: "",
        promptExtend: true,
        negativePrompt: "",
        batchSize: 1,
        cameraMovement: "none" as string,
        subjectMotion: "still" as string,
        model: defaultI2vModel,
        shotType: "single" as string,  // 'single' or 'multi' (only for wan2.6-i2v)
        generationMode: "i2v" as string,  // 'i2v' or 'r2v'
        referenceVideoUrls: [] as string[],  // Reference videos for R2V (max 3)
        // Kling params
        mode: "std" as string,
        sound: false,
        cfgScale: 0.5,
        // Vidu params
        viduAudio: true,
        movementAmplitude: "auto" as string,
    });

    // Sync model from project settings when project changes
    useEffect(() => {
        setParams((p) => ({
            ...p,
            model: resolveModelId(
                'i2v',
                currentProject?.model_settings?.i2v_model,
                'video_sidebar',
            ),
        }));
    }, [currentProject?.model_settings?.i2v_model]);

    // Sync tasks from project
    useEffect(() => {
        if (currentProject?.video_tasks) {
            setTasks(currentProject.video_tasks);
        }
    }, [currentProject?.video_tasks]);

    // Poll for updates
    useEffect(() => {
        const hasActiveTasks = tasks.some(t => t.status === "pending" || t.status === "processing");
        if (!hasActiveTasks || !currentProject) return;

        const interval = setInterval(async () => {
            try {
                const project = await api.getProject(currentProject.id);
                if (project.video_tasks) {
                    setTasks(project.video_tasks);
                    updateProject(currentProject.id, { video_tasks: project.video_tasks });
                }
            } catch (error) {
                console.error("Failed to poll project status:", error);
            }
        }, 3000);

        return () => clearInterval(interval);
    }, [tasks, currentProject?.id]);

    const handleTaskCreated = (updatedProject: any) => {
        if (updatedProject.video_tasks) {
            setTasks(updatedProject.video_tasks);
            updateProject(currentProject!.id, { video_tasks: updatedProject.video_tasks });
        }
    };

    const handleRemix = (task: VideoTask) => {
        setRemixData({
            image_url: task.image_url,
            prompt: task.prompt,
            negative_prompt: task.negative_prompt,
            seed: task.seed,
            duration: task.duration,
            audio_url: task.audio_url,
            prompt_extend: task.prompt_extend
        });

        // Update params state
        setParams(p => ({
            ...p,
            duration: task.duration || 5,
            seed: task.seed,
            resolution: task.resolution || "720p",
            generateAudio: task.generate_audio,
            audioUrl: task.audio_url || "",
            promptExtend: task.prompt_extend ?? true,
            negativePrompt: task.negative_prompt || "",
            // Reset motion params as they are not stored directly in task (they are in prompt)
            cameraMovement: "none",
            subjectMotion: "still"
        }));
    };

    const handleAutoMotion = async () => {
        if (!currentProject || autoRunning) return;
        if (!confirm("将把已渲染的分镜按镜头做成 4 秒动效漫画片段（静帧运镜，不调用视频模型），确定继续？")) return;
        setAutoRunning(true);
        try {
            const res = await api.motionComic(currentProject.id);
            alert(`已生成 ${res.count} 段动效漫画片段（可在右侧查看/预览）。`);
            const project = await api.getProject(currentProject.id);
            updateProject(currentProject.id, project);
        } catch (error: any) {
            alert(error?.response?.data?.detail || error?.message || "自动生成失败");
        } finally {
            setAutoRunning(false);
        }
    };

    const queueCount = tasks.filter(t => t.status === "pending" || t.status === "processing").length;
    const doneCount = tasks.filter(t => t.status === "completed").length;

    return (
        <div className="flex flex-col h-full w-full overflow-hidden">
            <StepHeader
                stepNumber={5}
                totalSteps={6}
                icon={<Video />}
                englishName="Motion Generator"
                title={tStep("motionTitle")}
                subtitle={tStep("motionSubtitle")}
                trailing={
                    <div className="flex items-center gap-3">
                        <WorkflowActionButton
                            variant="primary"
                            size="sm"
                            leftIcon={autoRunning ? <RefreshCw className="animate-spin" /> : <Sparkles />}
                            onClick={handleAutoMotion}
                            disabled={autoRunning}
                            title="按镜头分组，用静帧运镜生成动效片段"
                        >
                            {autoRunning ? "生成中…" : "自动生成动效漫画"}
                        </WorkflowActionButton>
                        {tasks.length > 0 ? (
                            <span className="font-mono text-[0.625rem] uppercase tracking-[0.18em] text-text-muted">
                                <span className="text-foreground font-medium">{tasks.length}</span>
                                <span className="ml-1.5">shots</span>
                                {doneCount > 0 ? (
                                    <>
                                        <span className="mx-1.5 text-text-muted/40">·</span>
                                        <span className="text-primary">{doneCount}</span>
                                        <span className="ml-1.5">done</span>
                                    </>
                                ) : null}
                                {queueCount > 0 ? (
                                    <>
                                        <span className="mx-1.5 text-text-muted/40">·</span>
                                        <span className="text-foreground">{queueCount}</span>
                                        <span className="ml-1.5">in queue</span>
                                    </>
                                ) : null}
                            </span>
                        ) : null}
                    </div>
                }
            />
            <StepGuide
                steps={[
                    "点「自动生成动效漫画」：按镜头分组，用静帧做推拉/平移片段（不调用视频模型）。",
                    "点生成视频片段（本地 LTX，单段约 2–5 分钟）。",
                    "播放检查，不满意可重新生成。",
                    "完成后进入「6. 合成」。",
                ]}
            />
            <div className="flex flex-1 overflow-hidden">
                {/* Left: Creator (70%) */}
                <div className="w-[70%] h-full border-r border-glass-border">
                    <VideoCreator
                        onTaskCreated={handleTaskCreated}
                        remixData={remixData}
                        onRemixClear={() => setRemixData(null)}
                        params={params}
                        onParamsChange={(newParams) => setParams(p => ({ ...p, ...newParams }))}
                    />
                </div>

                {/* Right: Sidebar (30%) */}
                <div className="w-[30%] h-full">
                    <VideoSidebar
                        tasks={tasks}
                        onRemix={handleRemix}
                        params={params}
                        setParams={setParams}
                    />
                </div>
            </div>
        </div>
    );
}
