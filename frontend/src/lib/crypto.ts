// 密码应用层加密(§46): 取后端 RSA 公钥, 用浏览器原生 Web Crypto 做
// RSA-OAEP(SHA-256) 加密后 base64 传输, 请求体中不出现明文密码。
// 与 HTTPS/TLS 配合形成纵深防御; 零第三方依赖。
const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

let cachedKey: Promise<CryptoKey> | null = null;

function pemToDer(pem: string): ArrayBuffer {
  const b64 = pem.replace(/-----[A-Z ]+-----/g, "").replace(/\s+/g, "");
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function bufToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

/** 获取(并缓存)后端密码加密公钥; force=true 用于加密失败后重新拉取(密钥轮换)。 */
export function getPublicKey(force = false): Promise<CryptoKey> {
  if (force || !cachedKey) {
    cachedKey = (async () => {
      const res = await fetch(`${API_BASE}/api/v1/auth/public-key`);
      if (!res.ok) throw new Error("无法获取密码加密公钥");
      const data: { public_key: string } = await res.json();
      return crypto.subtle.importKey(
        "spki",
        pemToDer(data.public_key),
        { name: "RSA-OAEP", hash: "SHA-256" },
        false,
        ["encrypt"],
      );
    })();
    cachedKey.catch(() => {
      cachedKey = null;
    });
  }
  return cachedKey;
}

/** 用 RSA-OAEP 公钥加密密码, 返回 base64 密文。 */
export async function encryptPassword(password: string): Promise<string> {
  const data = new TextEncoder().encode(password);
  try {
    const key = await getPublicKey();
    const buf = await crypto.subtle.encrypt({ name: "RSA-OAEP" }, key, data);
    return bufToBase64(buf);
  } catch {
    // 可能公钥已轮换: 强制重新拉取后重试一次
    const key = await getPublicKey(true);
    const buf = await crypto.subtle.encrypt({ name: "RSA-OAEP" }, key, data);
    return bufToBase64(buf);
  }
}

/** 前端密码策略预检(与后端 crypto.validate_password_policy 保持一致)。 */
export function passwordPolicyError(password: string): string {
  if (password.length < 8 || password.length > 128) {
    return "密码长度需为 8-128 位";
  }
  if (!/[a-zA-Z]/.test(password)) return "密码需至少包含一个字母";
  if (!/\d/.test(password)) return "密码需至少包含一个数字";
  return "";
}
