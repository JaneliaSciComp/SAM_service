#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate segment_anything
cd /app/SAM_service/sam_service || exit
uvicorn sam_queue:app --access-log --workers 1 --forwarded-allow-ips='*' --proxy-headers --host 0.0.0.0 "$@"

