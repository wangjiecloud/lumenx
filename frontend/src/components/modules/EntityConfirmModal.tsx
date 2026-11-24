"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Users, MapPin, Box, Check, X, Loader2 } from "lucide-react";
import { useTranslations } from "next-intl";

export interface ExtractionPreview {
    characters: { name: string; description?: string }[];
    scenes: { name: string; description?: string }[];
    props: { name: string; description?: string }[];
}

interface EntityConfirmModalProps {
    isOpen: boolean;
    preview: ExtractionPreview | null;
    currentCounts: { characters: number; scenes: number; props: number };
    loading?: boolean;
    onConfirm: (preserveFrames: boolean) => void;
    onDiscard: () => void;
}

export default function EntityConfirmModal({
    isOpen,
    preview,
    currentCounts,
    loading = false,
    onConfirm,
    onDiscard,
}: EntityConfirmModalProps) {
    const t = useTranslations("script");
    const [preserveFrames, setPreserveFrames] = useState(false);

    if (!preview) return null;

    const sections = [
        { key: "characters" as const, icon: Users, items: preview.characters, prev: currentCounts.characters },
        { key: "scenes" as const, icon: MapPin, items: preview.scenes, prev: currentCounts.scenes },
        { key: "props" as const, icon: Box, items: preview.props, prev: currentCounts.props },
    ];

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[100] grid place-items-center bg-overlay backdrop-blur-sm"
                    onClick={onDiscard}
                >
                    <motion.div
                        initial={{ scale: 0.96, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.96, opacity: 0 }}
                        transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                        className="relative w-full max-w-lg max-h-[70vh] flex flex-col rounded-2xl border border-glass-border bg-elevated shadow-[0_24px_64px_-12px_rgba(0,0,0,0.7)]"
                        onClick={e => e.stopPropagation()}
                    >
                        {/* Header */}
                        <header className="px-6 py-5 border-b border-glass-border">
                            <h2 className="font-display text-display font-medium text-foreground">
                                {t("extractConfirmTitle")}
                            </h2>
                            <p className="text-xs text-text-secondary mt-1">
                                {t("extractConfirmSubtitle")}
                            </p>
                        </header>

                        {/* Body */}
                        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
                            {sections.map(({ key, icon: Icon, items, prev }) => (
                                <div key={key} className="space-y-2">
                                    <div className="flex items-center gap-2 text-sm text-text-secondary">
                                        <Icon size={14} />
                                        <span className="font-medium">
                                            {t(`entityKind_${key}`)}
                                        </span>
                                        <span className="ml-auto text-xs opacity-70">
                                            {prev} → {items.length}
                                        </span>
                                    </div>
                                    {items.length > 0 ? (
                                        <div className="flex flex-wrap gap-1.5">
                                            {items.map((item, i) => (
                                                <span
                                                    key={i}
                                                    className="inline-flex items-center px-2 py-0.5 rounded-md bg-elevated border border-glass-border text-xs text-foreground"
                                                    title={item.description}
                                                >
                                                    {item.name}
                                                </span>
                                            ))}
                                        </div>
                                    ) : (
                                        <p className="text-xs text-text-tertiary italic">{t("noEntities")}</p>
                                    )}
                                </div>
                            ))}
                        </div>

                        {/* Footer */}
                        <footer className="flex items-center justify-end gap-3 px-6 py-4 border-t border-glass-border">
                            <label className="mr-auto flex items-center gap-2 text-xs text-text-secondary cursor-pointer select-none">
                                <input
                                    type="checkbox"
                                    checked={preserveFrames}
                                    onChange={(e) => setPreserveFrames(e.target.checked)}
                                    className="h-3.5 w-3.5 accent-[var(--primary)]"
                                />
                                保留现有分镜（勾选则沿用当前 storyboard，仅重建实体）
                            </label>
                            <button
                                onClick={onDiscard}
                                disabled={loading}
                                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm text-text-secondary hover:text-foreground hover:bg-hover-bg transition-colors disabled:opacity-50 disabled:pointer-events-none"
                            >
                                <X size={14} />
                                {t("extractDiscard")}
                            </button>
                            <button
                                onClick={() => onConfirm(preserveFrames)}
                                disabled={loading}
                                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium bg-primary text-white hover:bg-primary/90 transition-colors disabled:opacity-60 disabled:pointer-events-none"
                            >
                                {loading ? <Loader2 size={14} className="animate-spin" /> : <Check size={14} />}
                                {t("extractApply")}
                            </button>
                        </footer>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
