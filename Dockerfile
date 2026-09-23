FROM python:3.10-slim
WORKDIR /app
COPY document_ai/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY document_ai/ ./document_ai/
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "document_ai.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
