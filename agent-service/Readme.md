```bash
docker build -t agent-service .
```

```bash
docker run -d \
  -v /home/cloud_user/ai/agent-service/app:/app/app \
  --network ai-net \
  -e LLM_BASE_URL="http://llm-service:8000" \
  -e AZURE_STORAGE_CONNECTION_STRING="" \
  -e AZURE_QUEUE_NAME="test" \
  -p 8001:8001 \
  agent-service
```

```bash
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Create a python project with flask app , create simple web app"}'
```