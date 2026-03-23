FROM node:20-alpine AS builder

WORKDIR /app

COPY scann_v2/frontend/package*.json ./
RUN npm ci

COPY scann_v2/frontend/ ./
RUN npm run build


FROM nginx:1.27-alpine

COPY docker/frontend/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
