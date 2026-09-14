"use client";

// 认证上下文: token/user 存 localStorage, 提供 login/register/logout。
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import { authenticate, clearAuth, getUser, getToken } from "@/lib/api";
import type { User } from "@/lib/types";

interface AuthContextValue {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setUser(getUser());
    setToken(getToken());
    setLoading(false);
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const data = await authenticate("/auth/login", { username, password });
    setUser(data.user);
    setToken(data.access_token);
  }, []);

  const register = useCallback(
    async (username: string, email: string, password: string) => {
      const data = await authenticate("/auth/register", {
        username,
        email,
        password,
      });
      setUser(data.user);
      setToken(data.access_token);
    },
    [],
  );

  const logout = useCallback(() => {
    clearAuth();
    setUser(null);
    setToken(null);
  }, []);

  const value = useMemo(
    () => ({ user, token, loading, login, register, logout }),
    [user, token, loading, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth 必须在 AuthProvider 内使用");
  return ctx;
}
