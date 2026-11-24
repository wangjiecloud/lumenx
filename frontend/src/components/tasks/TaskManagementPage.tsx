"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Pause,
  Play,
  Square,
  RefreshCw,
  Loader2,
  CheckCircle2,
  XCircle,
  Ban,
} from "lucide-react";
import { useTranslations } from "next-intl";
import { api } from "@/lib/api";
import { toast } from "@/store/toastStore";

interface BackgroundTask {
  id: string;
  kind: string;
  title: string;
  status: "running" | "paused" | "pending" | "completed" | "failed" | "canceled";
  progress: number;
  message?: string | null;
  error?: string | null;
  project_id?: string | null;
  series_id?: string | null;
  asset_type?: string | null;
  created_at: number;
  updated_at: number;
  pause_requested?: boolean;
  stop_requested?: boolean;
}

const POLL_MS = 2000;

export default function TaskManagementPage() {
  const t = useTranslations("tasks");
  const [tasks, setTasks] = useState<BackgroundTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [showFinished, setShowFinished] = useState(false);
  const mounted = useRef(true);

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const data = (await api.listTasks(false)) as BackgroundTask[];
      if (mounted.current) setTasks(data);
    } catch (error) {
      if (!silent) {
        console.error("Failed to load background tasks:", error);
        toast.error(t("loadFailed"));
      }
    } finally {
      if (!silent && mounted.current) setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    mounted.current = true;
    load();
    const timer = setInterval(() => load(true), POLL_MS);
    return () => {
      mounted.current = false;
      clearInterval(timer);
    };
  }, [load]);

  const runAction = async (task: BackgroundTask, action: "pause" | "resume" | "stop") => {
    setBusy((prev) => ({ ...prev, [task.id]: true }));
    try {
      if (action === "pause") await api.pauseTask(task.id);
      else if (action === "resume") await api.resumeTask(task.id);
      else await api.stopTask(task.id);
      await load(true);
    } catch (error) {
      console.error(`Failed to ${action} task:`, error);
      toast.error(t(`${action}Failed`));
    } finally {
      setBusy((prev) => {
        const next = { ...prev };
        delete next[task.id];
        return next;
      });
    }
  };

  const active = tasks.filter((x) => !["completed", "failed", "canceled"].includes(x.status));
  const visible = showFinished ? tasks : active;

  const statusMeta: Record<string, { label: string; className: string; icon: typeof Activity }> = {
    running: { label: t("statusRunning"), className: "text-primary bg-primary/10 border-primary/30", icon: Activity },
    paused: { label: t("statusPaused"), className: "text-amber-400 bg-amber-500/10 border-amber-500/30", icon: Pause },
    pending: { label: t("statusPending"), className: "text-text-secondary bg-surface border-glass-border", icon: Loader2 },
    completed: { label: t("statusCompleted"), className: "text-green-400 bg-green-500/10 border-green-500/30", icon: CheckCircle2 },
    failed: { label: t("statusFailed"), className: "text-red-400 bg-red-500/10 border-red-500/30", icon: XCircle },
    canceled: { label: t("statusCanceled"), className: "text-text-muted bg-surface border-glass-border", icon: Ban },
  };

  const kindLabel = (kind: string) => {
    switch (kind) {
      case "asset_generation":
        return t("kindAsset");
      case "video":
        return t("kindVideo");
      case "motion_ref":
        return t("kindMotionRef");
      default:
        return kind;
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <header className="px-4 md:px-7 pt-5 md:pt-6 pb-3 flex flex-col md:flex-row md:items-end gap-3 md:gap-5">
        <div className="flex-1 min-w-0">
          <div className="font-mono text-[0.625rem] font-medium uppercase tracking-[0.2em] text-text-muted">
            WORKSPACE · <span className="text-primary font-semibold">{t("eyebrow")}</span>
          </div>
          <h1 className="text-[1.625rem] md:text-[2.125rem] font-display atelier-display font-semibold text-foreground leading-tight tracking-tight mt-1">
            {t("title")}
          </h1>
          <p className="text-[0.8125rem] text-text-secondary mt-1">{t("subtitle")}</p>
        </div>
        <div className="flex items-center flex-wrap gap-2.5 md:pb-1">
          <label className="inline-flex items-center gap-2 text-[0.75rem] text-text-secondary cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showFinished}
              onChange={(e) => setShowFinished(e.target.checked)}
              className="accent-[var(--primary)]"
            />
            {t("showFinished")}
          </label>
          <button
            onClick={() => load()}
            disabled={loading}
            className="glass-button flex items-center gap-2 text-[0.8125rem] font-semibold disabled:opacity-50"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            {t("refresh")}
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-4 md:px-7 pb-10 pt-3">
        {loading ? (
          <div className="flex items-center justify-center py-24 text-text-secondary">
            <Loader2 size={20} className="animate-spin mr-2" />
            {t("loading")}
          </div>
        ) : visible.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 text-text-secondary">
            <div className="w-16 h-16 rounded-2xl bg-glass border border-glass-border flex items-center justify-center mb-4">
              <Activity size={26} className="text-text-muted" />
            </div>
            <p className="text-sm font-medium">{showFinished ? t("emptyAll") : t("emptyActive")}</p>
            <p className="text-xs text-text-muted mt-1">{t("emptyHint")}</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2.5">
            {visible.map((task) => {
              const meta = statusMeta[task.status] ?? statusMeta.pending;
              const StatusIcon = meta.icon;
              const isActive = !["completed", "failed", "canceled"].includes(task.status);
              const isBusy = !!busy[task.id];
              const created = new Date(task.created_at * 1000);
              return (
                <motion.div
                  key={task.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-panel rounded-xl border border-glass-border p-4"
                >
                  <div className="flex flex-wrap items-center gap-3">
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[0.625rem] font-mono font-semibold uppercase tracking-wider ${meta.className}`}>
                      <StatusIcon size={12} className={task.status === "running" && task.progress < 100 ? "animate-pulse" : ""} />
                      {meta.label}
                    </span>
                    <span className="text-[0.6875rem] font-mono uppercase tracking-wider text-text-muted">
                      {kindLabel(task.kind)}
                    </span>
                    <span className="flex-1 min-w-[160px] text-sm font-medium text-foreground truncate">
                      {task.title}
                    </span>
                    <span className="text-[0.6875rem] font-mono text-text-muted">
                      {created.toLocaleTimeString()}
                    </span>
                  </div>

                  <div className="mt-3 flex items-center gap-3">
                    <div className="flex-1 h-1.5 rounded-full bg-surface-inset overflow-hidden">
                      <div
                        className="h-full bg-primary transition-[width] duration-500"
                        style={{ width: `${Math.max(0, Math.min(100, task.progress))}%` }}
                      />
                    </div>
                    <span className="text-[0.6875rem] font-mono text-text-secondary w-10 text-right">
                      {Math.round(task.progress)}%
                    </span>
                  </div>

                  {(task.message || task.error) && (
                    <p className={`mt-2 text-[0.75rem] ${task.error ? "text-red-400" : "text-text-secondary"}`}>
                      {task.error || task.message}
                    </p>
                  )}

                  {isActive && (
                    <div className="mt-3 flex items-center gap-2">
                      {task.status === "paused" || task.pause_requested ? (
                        <button
                          onClick={() => runAction(task, "resume")}
                          disabled={isBusy || task.stop_requested}
                          className="glass-button flex items-center gap-1.5 text-[0.75rem] font-semibold disabled:opacity-50"
                        >
                          <Play size={13} />
                          {t("resume")}
                        </button>
                      ) : (
                        <button
                          onClick={() => runAction(task, "pause")}
                          disabled={isBusy || task.stop_requested}
                          className="glass-button flex items-center gap-1.5 text-[0.75rem] font-semibold disabled:opacity-50"
                        >
                          <Pause size={13} />
                          {t("pause")}
                        </button>
                      )}
                      <button
                        onClick={() => runAction(task, "stop")}
                        disabled={isBusy || task.stop_requested}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-red-500/30 text-red-400 hover:bg-red-500/10 transition-colors text-[0.75rem] font-semibold disabled:opacity-50"
                      >
                        <Square size={13} />
                        {task.stop_requested ? t("stopping") : t("stop")}
                      </button>
                      <span className="text-[0.6875rem] text-text-muted ml-1">{t("pauseHint")}</span>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
