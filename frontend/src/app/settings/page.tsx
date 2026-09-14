"use client";

// 设置页: 当前账号信息 + API 连接信息。
import { useAuth } from "@/context/auth-context";
import { API_BASE } from "@/lib/api";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export default function SettingsPage() {
  const { user } = useAuth();
  if (!user) return null;

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-neutral-900">Settings</h1>

      <Card>
        <CardHeader title="账号信息" />
        <CardContent>
          <dl className="grid grid-cols-1 gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
            <div>
              <dt className="text-xs text-neutral-500">用户名</dt>
              <dd className="mt-0.5 font-medium text-neutral-900">{user.username}</dd>
            </div>
            <div>
              <dt className="text-xs text-neutral-500">邮箱</dt>
              <dd className="mt-0.5 font-medium text-neutral-900">{user.email}</dd>
            </div>
            <div>
              <dt className="text-xs text-neutral-500">用户 ID</dt>
              <dd className="mt-0.5 font-mono text-xs text-neutral-700">{user.id}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="连接信息" description="前端 API 网关地址" />
        <CardContent>
          <p className="font-mono text-sm text-neutral-700">{API_BASE}</p>
        </CardContent>
      </Card>
    </div>
  );
}
