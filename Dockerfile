FROM oven/bun:1 AS base
WORKDIR /app

COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile --production

COPY tsconfig.json ./
COPY src/ ./src/

EXPOSE 5000
ENV NODE_ENV=production

CMD ["bun", "run", "src/index.ts"]
