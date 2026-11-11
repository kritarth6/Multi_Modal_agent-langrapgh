#!/usr/bin/env bash

# 1. Health Check
curl -X GET "http://127.0.0.1:8000/api/health"

# 2. Get Mermaid Graph Diagram
curl -X GET "http://127.0.0.1:8000/api/graph"

# 3. Post Chat Request (Text Query)
curl -X POST "http://127.0.0.1:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Calculate the CAGR for revenue starting at 100M and growing to 250M over 4 years",
       "thread_id": "curl-test-session"
     }'

