#!/bin/bash

echo "=== Docker Container Debug Script ==="
echo "$(date): Starting debug investigation"

echo -e "\n1. Checking container status..."
docker ps -a | grep reddit-classifier

echo -e "\n2. Checking container logs (last 50 lines)..."
docker logs reddit-classifier-api --tail 50

echo -e "\n3. Checking container resource usage..."
docker stats reddit-classifier-api --no-stream

echo -e "\n4. Checking if port 9092 is accessible..."
netstat -tlnp | grep 9092

echo -e "\n5. Testing container network connectivity..."
timeout 5 telnet localhost 9092 || echo "Cannot connect to port 9092"

echo -e "\n6. Checking model files in container..."
docker exec reddit-classifier-api ls -la *.pt 2>/dev/null || echo "Cannot access container or model files missing"

echo -e "\n7. Checking if Python process is running in container..."
docker exec reddit-classifier-api ps aux | grep python 2>/dev/null || echo "Cannot access container"

echo -e "\n8. Checking container memory and CPU limits..."
docker inspect reddit-classifier-api | grep -A 10 -B 5 "Memory\|Cpu"

echo -e "\n9. Testing with minimal curl request (30 second timeout)..."
curl --max-time 30 -X POST -H "Content-Type: application/json" \
  -d '{"contents": [{"content": "test", "url": "http://test.com"}]}' \
  http://localhost:9092 -v 2>&1 || echo "Curl request failed or timed out"

echo -e "\n10. Checking system resources..."
echo "Memory usage:"
free -h
echo "CPU load:"
cat /proc/loadavg

echo -e "\n=== Debug investigation complete ===" 