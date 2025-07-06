#!/bin/bash

echo "=== Health Monitor Script ==="

# Function to test API health
test_api_health() {
    echo "Testing API health..."
    response=$(curl --max-time 30 --connect-timeout 5 -s -o /dev/null -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"contents": [{"content": "health check", "url": "http://test.com"}]}' \
        http://localhost:9092)
    
    if [ "$response" = "200" ]; then
        echo "✓ API is healthy (HTTP $response)"
        return 0
    else
        echo "✗ API is unhealthy (HTTP $response)"
        return 1
    fi
}

# Function to restart container
restart_container() {
    echo "Restarting hung container..."
    docker-compose down
    sleep 5
    docker-compose up -d
    sleep 30  # Wait for startup
}

# Main monitoring loop
while true; do
    echo "$(date): Checking API health..."
    
    if test_api_health; then
        echo "API is working fine"
    else
        echo "API is hung, restarting container..."
        restart_container
        
        # Test again after restart
        if test_api_health; then
            echo "Container restarted successfully"
        else
            echo "Container still not responding after restart"
        fi
    fi
    
    echo "Sleeping for 5 minutes..."
    sleep 300  # Check every 5 minutes
done 