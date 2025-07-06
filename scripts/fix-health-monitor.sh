#!/bin/bash

echo "=== Fixing Health Monitor ==="

# Stop the current broken instance
echo "Stopping current health monitor..."
./health-monitor.sh stop

# Kill any remaining processes if needed
if [ -f "health-monitor.pid" ]; then
    OLD_PID=$(cat health-monitor.pid)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Force killing old process: $OLD_PID"
        kill -9 "$OLD_PID"
    fi
    rm -f health-monitor.pid
fi

# Clean up any old log files
rm -f health-monitor.log

echo "Starting fixed health monitor..."
./health-monitor.sh start

echo "Waiting 10 seconds for logs to appear..."
sleep 10

echo "Checking status..."
./health-monitor.sh status

echo "=== Fix complete ===" 