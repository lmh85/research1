#!/bin/bash
CHECK_INTERVAL=60
ZERO_COUNT=0
MAX_ZERO=5

while true; do
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ' ')

    echo "$(date '+%Y-%m-%d %H:%M:%S') GPU util: $GPU_UTIL%"

    if [ "$GPU_UTIL" = "0" ]; then
        ZERO_COUNT=$((ZERO_COUNT+1))
        echo "GPU 0%，连续次数：$ZERO_COUNT"
        if [ $ZERO_COUNT -ge $MAX_ZERO ]; then
            echo "GPU 连续 5 次为 0%，自动关机..."
            /usr/bin/shutdown -h now
            exit 0
        fi
    else
        ZERO_COUNT=0
    fi

    sleep $CHECK_INTERVAL
done
