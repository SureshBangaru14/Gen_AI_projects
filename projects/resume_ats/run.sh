#!/bin/bash

echo "Starting FastAPI..."

uvicorn app.main:app \
--host 0.0.0.0 \
--port 8001 &


echo "Starting Streamlit..."

python -m streamlit run streamlit/app.py \
--server.port 8501 \
--server.address 0.0.0.0



# chmod +x run.sh # Give permission:
# run sh file  : ./run.sh
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8010