// 轻量 className 合并工具(替代 clsx+tailwind-merge, 零依赖)。
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

// 字节数格式化
export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

// 时间格式化(ISO → 本地可读)
export function formatTime(iso?: string | null): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("zh-CN", { hour12: false });
}

// 后端状态枚举 → 中文展示
const STATUS_TEXT: Record<string, string> = {
  CREATED: "已创建",
  QUEUED: "排队中",
  RUNNING: "运行中",
  SUCCESS: "成功",
  FAILED: "失败",
  CANCELLED: "已取消",
  SKIPPED: "已跳过",
  PENDING: "等待中",
  READY: "就绪",
  PROCESSING: "处理中",
  OFFLINE: "离线",
};

export function statusText(status?: string | null): string {
  if (!status) return "-";
  return STATUS_TEXT[status] ?? status;
}
