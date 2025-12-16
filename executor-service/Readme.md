```bash
docker run -it -d --rm \
  --entrypoint="bash" \
  -e AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=vieer;AccountKey=6x/2L3frh6Ug/j0UUig4Zl9kulaq+J4LKmza8ttqj74a2sYmZDNl7FY9dV+tVqFmhYfWXsk6Pl+Q+AStAZ9bqQ==;EndpointSuffix=core.windows.net" \
  -e AZURE_QUEUE_NAME="test" \
  -e LLM_BASE_URL="http://llm-service:8000" \
  -e POLL_INTERVAL=2 \
  -e MAX_LLM_RETRIES=5 \
  -v /home/cloud_user/ai/executor-service/app:/app/app \
  -v /home/cloud_user/ai/executor-service/workdir:/app/workdir \
  -v /home/cloud_user/ai/executor-service/logs:/app/logs \
  executor-service
```