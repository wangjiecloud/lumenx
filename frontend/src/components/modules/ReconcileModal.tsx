"use client";
/**
 * ReconcileModal — 跨集实体对齐。
 *
 * 触发时机：Script 步骤「提取实体」完成后（仅当本集属于某个系列）。
 * 展示“本集新实体”与“系列共享资产库”的匹配，默认采纳建议：
 *   · 高置信（≥75）→ 合并到系列
 *   · 低置信 → 作为新资产加入系列
 *   · 支持手动改为「合并到指定系列实体」（别名系统：韩立 / 二愣子 / 厉飞雨）
 *   · 「[全部确认]」一键应用
 *   · 「[确认并去查看 →]」应用后跳到资产页
 */
import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Check, X, Users, MapPin, Box, ArrowRight, Loader2 } from "lucide-react";
import { useTranslations } from "next-intl";
import { api, type ReconcileSuggestion, type ReconcileAction, type ReconcileCandidate } from "@/lib/api";
import { useProjectStore } from "@/store/projectStore";
import WorkflowActionButton from "@/components/shared/WorkflowActionButton";

interface ReconcileModalProps {
    isOpen: boolean;
    scriptId: string | null;
    onClose: () => void;
    onApplied?: () => void;
}

type Kind = "character" | "scene" | "prop";

interface Row {
    kind: Kind;
    suggestion: ReconcileSuggestion;
    action: "merge_into_series" | "create_new_in_series" | "skip";
    targetId: string | null;
    updateDescription: boolean;
}

type CandidateMap = {
    character: ReconcileCandidate[];
    scene: ReconcileCandidate[];
    prop: ReconcileCandidate[];
};

const EMPTY_CANDIDATES: CandidateMap = { character: [], scene: [], prop: [] };

export default function ReconcileModal({ isOpen, scriptId, onClose, onApplied }: ReconcileModalProps) {
    const t = useTranslations("reconcile");
    const [rows, setRows] = useState<Row[] | null>(null);
    const [candidates, setCandidates] = useState<CandidateMap>(EMPTY_CANDIDATES);
    const [loading, setLoading] = useState(false);
    const [applying, setApplying] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!isOpen || !scriptId) return;
        let cancelled = false;
        setLoading(true);
        setError(null);
        api.getReconcileSuggestions(scriptId)
            .then((data) => {
                if (cancelled) return;
                const init: Row[] = [];
                const seed = (kind: Kind, list: ReconcileSuggestion[]) => {
                    for (const s of list) {
                        init.push({
                            kind,
                            suggestion: s,
                            action:
                                s.confidence >= 75 && s.suggested_series_id
                                    ? "merge_into_series"
                                    : "create_new_in_series",
                            targetId: s.suggested_series_id ?? null,
                            updateDescription: false,
                        });
                    }
                };
                seed("character", data.characters);
                seed("scene", data.scenes);
                seed("prop", data.props);
                const cand = data.series_candidates;
                setCandidates({
                    character: cand?.characters ?? [],
                    scene: cand?.scenes ?? [],
                    prop: cand?.props ?? [],
                });
                setRows(init);
            })
            .catch((err) => {
                if (cancelled) return;
                setError(err?.response?.data?.detail || err?.message || "加载失败");
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, [isOpen, scriptId]);

    const counts = useMemo(() => {
        const base = { total: 0, merge: 0, create: 0, skip: 0 };
        if (!rows) return base;
        base.total = rows.length;
        for (const r of rows) {
            if (r.action === "merge_into_series") base.merge++;
            else if (r.action === "create_new_in_series") base.create++;
            else base.skip++;
        }
        return base;
    }, [rows]);

    const handleSetAction = (idx: number, action: Row["action"]) => {
        setRows((prev) =>
            prev?.map((r, i) => {
                if (i !== idx) return r;
                if (action === "merge_into_series" && !r.targetId) {
                    const first = candidates[r.kind][0];
                    return { ...r, action, targetId: first?.id ?? null };
                }
                return { ...r, action };
            }) ?? null,
        );
    };

    const handleSetTarget = (idx: number, targetId: string) => {
        setRows((prev) =>
            prev?.map((r, i) =>
                i === idx
                    ? { ...r, action: "merge_into_series", targetId: targetId || null }
                    : r,
            ) ?? null,
        );
    };

    const handleToggleUpdateDescription = (idx: number) => {
        setRows((prev) =>
            prev?.map((r, i) =>
                i === idx ? { ...r, updateDescription: !r.updateDescription } : r,
            ) ?? null,
        );
    };

    const buildPayload = (): {
        characters: ReconcileAction[];
        scenes: ReconcileAction[];
        props: ReconcileAction[];
    } => {
        const payload = {
            characters: [] as ReconcileAction[],
            scenes: [] as ReconcileAction[],
            props: [] as ReconcileAction[],
        };
        for (const r of rows ?? []) {
            const act: ReconcileAction = {
                local_id: r.suggestion.local_id,
                action: r.action,
                target_series_id:
                    r.action === "merge_into_series" ? r.targetId ?? undefined : undefined,
                update_description:
                    r.action === "merge_into_series" ? r.updateDescription : false,
            };
            if (r.kind === "character") payload.characters.push(act);
            else if (r.kind === "scene") payload.scenes.push(act);
            else payload.props.push(act);
        }
        return payload;
    };

    const handleApply = async (navigateToCast: boolean) => {
        if (!scriptId || !rows) return;
        setApplying(true);
        setError(null);
        try {
            await api.applyReconcile(scriptId, buildPayload());
            try {
                const merged = await api.getProject(scriptId);
                useProjectStore.getState().updateProject(scriptId, merged);
            } catch {
                /* refetch best-effort */
            }
            onApplied?.();
            onClose();
            if (navigateToCast) {
                document.dispatchEvent(new CustomEvent("lumenx:navigateStep", { detail: "cast" }));
            }
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || "应用失败");
        } finally {
            setApplying(false);
        }
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[100] grid place-items-center bg-overlay backdrop-blur-sm"
                    onClick={onClose}
                >
                    <motion.div
                        initial={{ scale: 0.96, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.96, opacity: 0 }}
                        transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                        className="relative w-full max-w-2xl max-h-[80vh] flex flex-col rounded-2xl border border-glass-border bg-elevated shadow-[0_24px_64px_-12px_rgba(0,0,0,0.7)]"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <header className="flex items-center gap-3 px-6 py-5 border-b border-glass-border">
                            <div className="grid h-9 w-9 place-items-center rounded-full border border-primary/40 bg-primary/10 text-primary">
                                <Sparkles size={16} />
                            </div>
                            <div className="flex-1 min-w-0">
                                <h2 className="font-display text-display font-medium text-foreground">{t("title")}</h2>
                                <p className="text-xs text-text-secondary mt-0.5">
                                    {loading
                                        ? t("loading")
                                        : t("subtitle", {
                                              total: counts.total,
                                              merge: counts.merge,
                                              create: counts.create,
                                          })}
                                </p>
                            </div>
                            <button
                                onClick={onClose}
                                aria-label="关闭"
                                className="p-2 hover:bg-hover-bg rounded-lg text-text-muted hover:text-foreground transition-colors"
                            >
                                <X size={16} />
                            </button>
                        </header>

                        <div className="flex-1 overflow-y-auto px-6 py-4 custom-scrollbar">
                            {loading ? (
                                <div className="flex items-center justify-center py-12 text-text-muted">
                                    <Loader2 size={20} className="animate-spin" />
                                </div>
                            ) : error ? (
                                <div className="rounded-lg border border-status-failed-border/40 bg-status-failed-bg/50 px-4 py-3 text-status-failed-fg text-sm">
                                    {error}
                                </div>
                            ) : !rows || rows.length === 0 ? (
                                <p className="text-center text-text-muted py-12 text-sm">{t("noEntities")}</p>
                            ) : (
                                <div className="space-y-1.5">
                                    <p className="px-1 pb-1 text-[0.6875rem] text-text-muted leading-relaxed">
                                        提示：若某角色是别名/化名（例如“厉飞雨”其实是“韩立”），把右侧动作切换为
                                        <span className="text-primary"> 合并到系列 </span>
                                        并选择目标角色即可；后续会记住该别名。
                                    </p>
                                    {rows.map((row, idx) => (
                                        <ReconcileRow
                                            key={`${row.kind}-${row.suggestion.local_id}`}
                                            row={row}
                                            candidates={candidates[row.kind]}
                                            onActionChange={(action) => handleSetAction(idx, action)}
                                            onTargetChange={(id) => handleSetTarget(idx, id)}
                                            onToggleUpdateDescription={() => handleToggleUpdateDescription(idx)}
                                        />
                                    ))}
                                </div>
                            )}
                        </div>

                        <footer className="flex items-center gap-2 px-6 py-4 border-t border-glass-border">
                            <span className="flex-1 font-mono text-[0.65625rem] uppercase tracking-[0.16em] text-text-muted">
                                {counts.merge > 0 && <span className="text-primary mr-2">合并 {counts.merge}</span>}
                                {counts.create > 0 && <span className="text-pink-300 mr-2">新增 {counts.create}</span>}
                                {counts.skip > 0 && <span className="text-text-muted">跳过 {counts.skip}</span>}
                            </span>
                            <WorkflowActionButton variant="ghost" size="sm" onClick={onClose}>
                                {t("cancel")}
                            </WorkflowActionButton>
                            <WorkflowActionButton
                                variant="secondary"
                                size="sm"
                                loading={applying}
                                onClick={() => handleApply(false)}
                                disabled={!rows || rows.length === 0}
                            >
                                {t("confirmAll")}
                            </WorkflowActionButton>
                            <WorkflowActionButton
                                variant="primary"
                                size="sm"
                                loading={applying}
                                rightIcon={<ArrowRight />}
                                onClick={() => handleApply(true)}
                                disabled={!rows || rows.length === 0}
                            >
                                {t("confirmAndGoCast")}
                            </WorkflowActionButton>
                        </footer>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}

function ReconcileRow({
    row,
    candidates,
    onActionChange,
    onTargetChange,
    onToggleUpdateDescription,
}: {
    row: Row;
    candidates: ReconcileCandidate[];
    onActionChange: (a: Row["action"]) => void;
    onTargetChange: (id: string) => void;
    onToggleUpdateDescription: () => void;
}) {
    const t = useTranslations("reconcile");
    const Icon = row.kind === "character" ? Users : row.kind === "scene" ? MapPin : Box;
    const conf = row.suggestion.confidence;
    const isHighConf = conf >= 75;
    const isMediumConf = conf > 0 && conf < 75;
    const hasTargets = candidates.length > 0;
    const localAliases = row.suggestion.local_aliases ?? [];
    const matchedAliases = row.suggestion.suggested_series_aliases ?? [];

    return (
        <div className="flex items-start gap-3 px-3 py-2.5 rounded-lg border border-glass-border bg-glass">
            <Icon size={14} className="shrink-0 mt-1 text-text-muted" />
            <div className="flex-1 min-w-0 space-y-1">
                <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-sans text-[0.8125rem] font-medium text-foreground truncate">
                        {row.suggestion.local_name}
                    </span>
                    {localAliases.length > 0 && (
                        <span className="font-sans text-[0.625rem] text-text-muted truncate">
                            别名：{localAliases.join(" / ")}
                        </span>
                    )}
                    {row.suggestion.suggested_series_id && (
                        <>
                            <span className="font-mono text-[0.625rem] text-text-muted">→</span>
                            <span className={`font-sans text-[0.8125rem] truncate ${isHighConf ? "text-foreground" : "text-text-secondary"}`}>
                                {row.suggestion.suggested_series_name}
                            </span>
                            {matchedAliases.length > 0 && (
                                <span className="font-sans text-[0.625rem] text-text-muted truncate">
                                    (系列别名：{matchedAliases.join(" / ")})
                                </span>
                            )}
                            <span
                                className={`font-mono text-[0.59375rem] px-1.5 py-0.5 rounded-full ${
                                    isHighConf
                                        ? "bg-primary/15 text-primary"
                                        : isMediumConf
                                        ? "bg-amber-400/15 text-amber-300"
                                        : "bg-glass text-text-muted"
                                }`}
                            >
                                {conf}%
                            </span>
                        </>
                    )}
                    {!row.suggestion.suggested_series_id && (
                        <span className="font-mono text-[0.59375rem] px-1.5 py-0.5 rounded-full bg-pink-400/15 text-pink-300">
                            {t("new")}
                        </span>
                    )}
                </div>

                {row.action === "merge_into_series" && hasTargets && (
                    <select
                        value={row.targetId ?? ""}
                        onChange={(e) => onTargetChange(e.target.value)}
                        className="w-full bg-input-bg border border-glass-border rounded px-2 py-1 text-[0.6875rem] text-foreground focus:outline-none focus:border-primary"
                    >
                        <option value="">选择要合并到的系列实体…</option>
                        {candidates.map((c) => (
                            <option key={c.id} value={c.id}>
                                {c.name}
                                {c.aliases && c.aliases.length > 0 ? `（${c.aliases.join("、")}）` : ""}
                            </option>
                        ))}
                    </select>
                )}

                {row.action === "merge_into_series" && (
                    <label className="flex items-center gap-1.5 text-[0.6875rem] text-text-muted cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={row.updateDescription}
                            onChange={onToggleUpdateDescription}
                            className="h-3 w-3 accent-[var(--primary)]"
                        />
                        用本集描述更新系列角色（外观确实变了时勾选）
                    </label>
                )}
            </div>

            <div className="shrink-0">
                <select
                    value={row.action}
                    onChange={(e) => onActionChange(e.target.value as Row["action"])}
                    className="bg-input-bg border border-glass-border rounded px-2 py-1 text-[0.71875rem] text-foreground focus:outline-none focus:border-primary"
                >
                    {hasTargets && <option value="merge_into_series">{t("actionMerge")}</option>}
                    <option value="create_new_in_series">{t("actionCreateNew")}</option>
                    <option value="skip">{t("actionSkip")}</option>
                </select>
            </div>

            <div className="shrink-0 w-5 grid place-items-center mt-1">
                {row.action !== "skip" ? (
                    <Check size={14} className={row.action === "merge_into_series" ? "text-primary" : "text-pink-300"} />
                ) : (
                    <X size={14} className="text-text-muted" />
                )}
            </div>
        </div>
    );
}
