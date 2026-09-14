"use client";

// 注册页: 注册成功后跳转登录页, 由用户主动登录。
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input, Label } from "@/components/ui/input";
import { ApiError, registerUser } from "@/lib/api";

export default function RegisterPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: { preventDefault: () => void }) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await registerUser({ username, email, password });
      router.replace("/login?registered=1");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "注册失败, 请重试");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-50 px-4">
      <Card className="w-full max-w-sm">
        <CardHeader title="注册账号" description="创建 XPU Model Hub 账号" />
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4">
            <div>
              <Label htmlFor="username">用户名</Label>
              <Input
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                minLength={3}
                required
              />
            </div>
            <div>
              <Label htmlFor="email">邮箱</Label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div>
              <Label htmlFor="password">密码</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                minLength={8}
                required
              />
            </div>
            {error && <p className="text-sm text-red-600">{error}</p>}
            <Button type="submit" className="w-full" disabled={busy}>
              {busy ? "注册中…" : "注册"}
            </Button>
            <p className="text-center text-sm text-neutral-500">
              已有账号?{" "}
              <Link href="/login" className="text-neutral-900 underline">
                登录
              </Link>
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
