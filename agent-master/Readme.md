```bash
docker build -t agent-service .
```

```bash
docker run -d \
  -v /home/cloud_user/ai/agent-master/app:/app/app \
  --network ai-net \
  -e LLM_BASE_URL="http://llm-service:8000" \
  -e AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=vieer;AccountKey=6x/2L3frh6Ug/j0UUig4Zl9kulaq+J4LKmza8ttqj74a2sYmZDNl7FY9dV+tVqFmhYfWXsk6Pl+Q+AStAZ9bqQ==;EndpointSuffix=core.windows.net" \
  -e AZURE_QUEUE_NAME="test" \
  -p 8001:8001 \
  agent-service
```

```bash
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Create a python project with flask app , create simple web app"}'
```