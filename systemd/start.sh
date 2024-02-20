#!/bin/bash
source /opt/deploy/miniforge3/bin/activate /opt/deploy/miniforge3/envs/segment_anything
cd /opt/deploy/SAM_service/sam_service
uvicorn sam_queue:app --access-log --workers 1 --forwarded-allow-ips='*' --proxy-headers --uds /tmp/uvicorn.sock

