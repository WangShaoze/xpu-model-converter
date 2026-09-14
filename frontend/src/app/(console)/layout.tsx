import type { ReactNode } from "react";
import { AppShell } from "@/components/layout/app-shell";

// 受保护路由组布局: 侧边导航 + 顶栏 + 登录守卫。
// 路由组 (console) 不影响 URL, /dashboard /projects /jobs /settings 路径保持不变;
// login/register 位于组外, 不套用此外壳。
export default function ConsoleLayout({ children }: { children: ReactNode }) {
  return <AppShell>{children}</AppShell>;
}
