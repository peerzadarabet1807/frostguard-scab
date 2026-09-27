# syntax=docker/dockerfile:1.7
# FrostGuard-Scab web UI: React build served by nginx. Build context is the repository root
# (the sample leaves are imported from data/samples).

FROM node:22-alpine AS build
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
COPY data/samples/ /app/data/samples/
ARG VITE_API_URL=http://localhost:8000
ARG VITE_BASE=/
ENV VITE_API_URL=${VITE_API_URL} VITE_BASE=${VITE_BASE}
RUN npm run build

FROM nginx:1.27-alpine
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/web/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD wget -qO /dev/null http://localhost:8080/ || exit 1
