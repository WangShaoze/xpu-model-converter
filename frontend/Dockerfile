# xpu-model-converter 前端镜像(Phase 7 §40): Next.js 16 生产模式
# 构建上下文为 frontend/ 目录:
#   docker build -t xpu-model-converter-frontend .
FROM node:22-bookworm-slim AS builder

WORKDIR /app

# NEXT_PUBLIC_* 会在构建期内联进客户端 bundle, 默认指向宿主机映射的 API 端口
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build && npm prune --omit=dev

# 运行期: 仅保留生产依赖
FROM node:22-bookworm-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.ts ./next.config.ts

EXPOSE 3000
CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0", "--port", "3000"]
