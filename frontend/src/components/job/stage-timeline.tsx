"use client";

// Stage 时间线(§38): 逐阶段显示状态与进度, 不做后端状态推断。
import type { Stage } from "@/lib/types";
import { Badge, toneForStatus } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn, statusText } from "@/lib/utils";

function StageIcon({ status }: { status: string }) {
  if (status === "SUCCESS") return <span className="text-green-600">✓</span>;
  if (status === "FAILED") return <span className="text-red-600">✗</span>;
  if (status === "RUNNING") return <span className="animate-pulse text-blue-600">●</span>;
  if (status === "SKIPPED") return <span className="text-neutral-400">⊘</span>;
  return <span className="text-neutral-300">○</span>;
}

export function StageTimeline({ stages }: { stages: Stage[] }) {
  if (!stages || stages.length === 0) {
    return <p className="text-sm text-neutral-500">暂无阶段信息</p>;
  }
  return (
    <ol className="space-y-1">
      {stages.map((s) => (
        <li
          key={s.name}
          className="flex items-center gap-3 rounded-md px-2 py-1.5 hover:bg-neutral-50"
        >
          <span className="w-4 text-center">
            <StageIcon status={s.status} />
          </span>
          <span className="flex-1 truncate text-sm font-medium text-neutral-800">
            {s.name}
          </span>
          {s.status === "RUNNING" && (
            <span className="w-28">
              <Progress value={s.progress} className="h-1.5" />
            </span>
          )}
          {s.error_message && (
            <span className="max-w-64 truncate text-xs text-red-600" title={s.error_message}>
              {s.error_message}
            </span>
          )}
          <Badge tone={toneForStatus(s.status)}>{statusText(s.status)}</Badge>
          <span className={cn("w-10 text-right text-xs text-neutral-400")}>
            {s.progress}%
          </span>
        </li>
      ))}
    </ol>
  );
}
