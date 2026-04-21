# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build          # outputs to /frontend/dist


# ── Stage 2: Python runtime ────────────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces requires the app to run as a non-root user
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python deps first (layer-cached)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy project files
COPY --chown=user:user . .

# Copy compiled frontend from Stage 1
COPY --from=builder --chown=user:user /frontend/dist ./frontend/dist

USER user

EXPOSE 7860
CMD ["python", "run_server.py"]
