# Stage 1: Build Bureau-Grade React Forensics Dashboard
FROM node:20-alpine AS ui-builder
WORKDIR /ui
COPY module4_runtime_ui/frontend_ui/package.json ./
RUN npm install
COPY module4_runtime_ui/frontend_ui/ ./
RUN npm run build

# Stage 2: FastAPI Runtime, Live AMLSim Streamer, Audit Logger & Static UI Host
FROM python:3.11-slim
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5

COPY contracts /app/contracts
RUN pip install --no-cache-dir -e /app/contracts

COPY module4_runtime_ui/backend_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY module1_data_pipeline /app/module1_data_pipeline
COPY module4_runtime_ui/backend_api /app/module4_runtime_ui/backend_api
COPY --from=ui-builder /ui/dist /app/static_ui

EXPOSE 8080
CMD ["uvicorn", "module4_runtime_ui.backend_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
