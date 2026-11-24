"use client";
/** StepGuide — 每个工作流步骤顶部的可折叠操作指引。 */
import { useState } from "react";
import { Lightbulb, ChevronDown, ChevronUp } from "lucide-react";

export interface StepGuideProps {
    title?: string;
    steps: string[];
    defaultOpen?: boolean;
    className?: string;
}

export default function StepGuide({
    title = "操作指引",
    steps,
    defaultOpen = true,
    className = "",
}: StepGuideProps) {
    const [open, setOpen] = useState(defaultOpen);

    return (
        <div className={`mx-6 mt-4 shrink-0 rounded-lg border border-primary/20 bg-primary/5 px-4 py-2.5 ${className}`}>
            <button
                type="button"
                onClick={() => setOpen((v) => !v)}
                className="flex w-full items-center gap-2 text-left"
            >
                <Lightbulb size={14} className="text-primary shrink-0" />
                <span className="text-[0.8125rem] font-medium text-foreground">{title}</span>
                <span className="ml-auto text-text-muted">
                    {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </span>
            </button>
            {open && (
                <ol className="mt-2 list-decimal space-y-1 pl-5 text-[0.75rem] leading-relaxed text-text-secondary">
                    {steps.map((step, i) => (
                        <li key={i}>{step}</li>
                    ))}
                </ol>
            )}
        </div>
    );
}
