#!/bin/bash

# Health Monitor Daemon for Reddit Classifier API
# Usage: ./health-monitor.sh [start|stop|status|restart]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/health-monitor.log"
PID_FILE="$SCRIPT_DIR/health-monitor.pid"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to test API health
test_api_health() {
    log "Testing API health..."
    response=$(curl --max-time 30 --connect-timeout 5 -s -o /dev/null -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"contents": [{"content": "health check", "url": "http://test.com"}]}' \
        http://localhost:9092 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        log "✓ API is healthy (HTTP $response)"
        return 0
    else
        log "✗ API is unhealthy (HTTP $response)"
        return 1
    fi
}

# Function to restart container
restart_container() {
    log "Restarting hung container..."
    cd "$PROJECT_DIR"
    docker-compose down >> "$LOG_FILE" 2>&1
    sleep 5
    docker-compose up -d >> "$LOG_FILE" 2>&1
    sleep 30  # Wait for startup
    log "Container restart completed"
}

# Main monitoring function
monitor_health() {
    log "=== Health Monitor Started ==="
    log "PID: $$"
    log "Log file: $LOG_FILE"
    log "Monitoring interval: 5 minutes"
    
    while true; do
        log "Checking API health..."
        
        if test_api_health; then
            log "API is working fine"
        else
            log "API is hung, restarting container..."
            restart_container
            
            # Test again after restart
            log "Testing API after restart..."
            if test_api_health; then
                log "Container restarted successfully"
            else
                log "WARNING: Container still not responding after restart"
            fi
        fi
        
        log "Sleeping for 5 minutes..."
        sleep 300  # Check every 5 minutes
    done
}

# Function to start the daemon
start_daemon() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Health monitor is already running (PID: $(cat "$PID_FILE"))"
        return 1
    fi
    
    echo "Starting health monitor daemon..."
    echo "Log file: $LOG_FILE"
    
    # Create the log file if it doesn't exist
    touch "$LOG_FILE"
    
    # Start the monitor in background by re-executing this script with a special flag
    nohup "$0" --daemon > /dev/null 2>&1 &
    
    # Save PID
    echo $! > "$PID_FILE"
    
    echo "Health monitor started with PID: $(cat "$PID_FILE")"
    echo "To view logs: tail -f $LOG_FILE"
    echo "To stop: $0 stop"
}

# Function to stop the daemon
stop_daemon() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Health monitor is not running (no PID file found)"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping health monitor (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        log "=== Health Monitor Stopped ==="
        echo "Health monitor stopped"
    else
        echo "Health monitor is not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
}

# Function to check daemon status
check_status() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Health monitor is running (PID: $(cat "$PID_FILE"))"
        echo "Log file: $LOG_FILE"
        echo "Last 5 log entries:"
        tail -n 5 "$LOG_FILE" 2>/dev/null || echo "No logs yet"
    else
        echo "Health monitor is not running"
        if [ -f "$PID_FILE" ]; then
            rm -f "$PID_FILE"
        fi
    fi
}

# Main script logic
case "${1:-start}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    restart)
        stop_daemon
        sleep 2
        start_daemon
        ;;
    status)
        check_status
        ;;
    --daemon)
        # Special flag to run as daemon
        monitor_health
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo "  start   - Start the health monitor daemon"
        echo "  stop    - Stop the health monitor daemon"
        echo "  restart - Restart the health monitor daemon"
        echo "  status  - Check daemon status and recent logs"
        exit 1
        ;;
esac 