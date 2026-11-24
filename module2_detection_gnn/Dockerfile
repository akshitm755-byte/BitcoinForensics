FROM python:3.11-slim
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5

COPY contracts /app/contracts
RUN pip install --no-cache-dir -e /app/contracts

# Install lightweight CPU-only PyTorch wheel for serverless Cloud Run inference
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu

COPY module2_detection_gnn/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY module1_data_pipeline /app/module1_data_pipeline
COPY module2_detection_gnn /app/module2_detection_gnn

EXPOSE 8080
CMD ["uvicorn", "module2_detection_gnn.src.server:app", "--host", "0.0.0.0", "--port", "8080"]
