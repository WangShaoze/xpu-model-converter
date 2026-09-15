import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type Tone = "neutral" | "green" | "red" | "amber" | "blue" | "gray";

const tones: Record<Tone, string> = {
  neutral: "bg-neutral-100 text-neutral-700",
  green: "bg-green-100 text-green-700",
  red: "bg-red-100 text-red-700",
  amber: "bg-amber-100 text-amber-700",
  blue: "bg-blue-100 text-blue-700",
  gray: "bg-neutral-200 text-neutral-600",
};

export function toneForStatus(status?: string): Tone {
  switch (status) {
    case "SUCCESS":
    case "READY":
      return "green";
    case "FAILED":
      return "red";
    case "RUNNING":
    case "PROCESSING":
      return "blue";
    case "QUEUED":
    case "CREATED":
      return "amber";
    case "CANCELLED":
      return "gray";
    default:
      return "neutral";
  }
}

export function Badge({
  className,
  tone = "neutral",
  ...props
}: HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}
