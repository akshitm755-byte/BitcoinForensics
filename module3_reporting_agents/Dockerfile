FROM python:3.11-slim
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5 MPLBACKEND=Agg

COPY contracts /app/contracts
RUN pip install --no-cache-dir -e /app/contracts
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu

COPY module3_reporting_agents/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY module1_data_pipeline /app/module1_data_pipeline
COPY module2_detection_gnn /app/module2_detection_gnn
COPY module3_reporting_agents /app/module3_reporting_agents

EXPOSE 8080
CMD ["uvicorn", "module3_reporting_agents.src.server:app", "--host", "0.0.0.0", "--port", "8080"]
