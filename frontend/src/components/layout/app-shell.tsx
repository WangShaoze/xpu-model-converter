"use client";

// 应用外壳: 认证守卫 + 侧边导航 + 顶栏。未登录跳转 /login。
import { useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { useAuth } from "@/context/auth-context";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/table";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard", label: "仪表盘" },
  { href: "/projects", label: "项目" },
  { href: "/jobs", label: "任务" },
  { href: "/settings", label: "设置" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const { user, loading, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [loading, user, router]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }
  if (!user) return null;

  return (
    <div className="flex min-h-screen bg-neutral-50">
      <aside className="flex w-56 flex-col border-r border-neutral-200 bg-white">
        <div className="flex h-14 items-center border-b border-neutral-200 px-5 font-semibold text-neutral-900">
          Model Converter Hub
        </div>
        <nav className="flex-1 space-y-1 p-3">
          {NAV.map((item) => {
            const active = pathname === item.href || pathname.startsWith(item.href + "/");
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "block rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-neutral-900 text-white"
                    : "text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="border-t border-neutral-200 p-3 text-xs text-neutral-500">
          v1.0 模型转换 Web SaaS
        </div>
      </aside>
      <div className="flex flex-1 flex-col">
        <header className="flex h-14 items-center justify-between border-b border-neutral-200 bg-white px-6">
          <span className="text-sm text-neutral-500">模型转换 Web SaaS</span>
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-neutral-700">
              {user.username}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                logout();
                router.replace("/login");
              }}
            >
              退出
            </Button>
          </div>
        </header>
        <main className="flex-1 p-6">{children}</main>
      </div>
    </div>
  );
}
