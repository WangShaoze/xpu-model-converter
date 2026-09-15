"use client";

// 实时事件日志(§28/§38 Live Logs): SSE 拉取 Job 事件, 断线自动重连(Last-Event-ID)。
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { JobEventStream } from "@/lib/sse";
import type { JobEvent } from "@/lib/types";
import { formatTime } from "@/lib/utils";
import { cn } from "@/lib/utils";

export function LiveEvents({
  jobId,
  token,
  active,
}: {
  jobId: string;
  token: string | null;
  active: boolean;
}) {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!active) return;
    const stream = new JobEventStream(jobId, token, {
      onEvent: (ev) => setEvents((prev) => [...prev, ev]),
      onStatus: setConnected,
    });
    stream.start();
    return () => stream.stop();
  }, [jobId, token, active]);

  useEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [events]);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium text-neutral-700">实时事件</span>
        <span
          className={cn(
            "flex items-center gap-1.5 text-xs",
            connected ? "text-green-600" : "text-neutral-400",
          )}
        >
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              connected ? "bg-green-500" : "bg-neutral-300",
            )}
          />
          {connected ? "已连接" : "连接中…"}
        </span>
      </div>
      <div
        ref={boxRef}
        className="h-72 overflow-y-auto rounded-md border border-neutral-200 bg-neutral-950 p-3 font-mono text-xs text-neutral-300"
      >
        {events.length === 0 && <p className="text-neutral-600">等待事件…</p>}
        {events.map((ev, i) => (
          <div key={i} className="flex gap-2 py-0.5">
            <span className="shrink-0 text-neutral-600">
              {formatTime(ev.timestamp ?? null).split(" ")[1] ?? ""}
            </span>
            <Badge
              tone={ev.event.startsWith("stage_failed") ? "red" : "blue"}
              className="shrink-0 px-1.5 py-0 text-[10px]"
            >
              {ev.event}
            </Badge>
            {ev.stage && (
              <span className="shrink-0 text-neutral-500">[{ev.stage}]</span>
            )}
            <span className="break-all text-neutral-300">
              {ev.message || JSON.stringify(ev.data)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
