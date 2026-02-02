#!/bin/bash
# Universal Stage Simple API Request Script
# Sends video processing requests without requiring jq

SERVER_URL="http://localhost:8091"
API_ENDPOINT="/v1/chat/completions"

echo "========================================"
echo "Universal Stage API Request"
echo "========================================"
echo ""

# Check server
echo "[*] Checking server at ${SERVER_URL}..."
if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "[✓] Server is running"
else
    echo "[✗] Server is not running at ${SERVER_URL}"
    echo "[i] Start the server with: bash examples/start_universal_stage_server.sh"
    exit 1
fi
echo ""

# Prepare request
echo "[*] Preparing request..."
REQUEST_BODY=$(cat <<'EOF'
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Process the following videos"
                },
                {
                    "type": "video_paths",
                    "video_paths": ["/root/test/sample_demo_1.mp4", "/root/test/sample_demo_2.mp4"]
                }
            ]
        }
    ],
    "max_tokens": 1,
    "temperature": 1.0
}
EOF
)

echo "[*] Request payload:"
echo "$REQUEST_BODY"
echo ""

# Send request
echo "[*] Sending request to ${SERVER_URL}${API_ENDPOINT}..."
RESPONSE=$(curl -s -X POST \
    "${SERVER_URL}${API_ENDPOINT}" \
    -H "Content-Type: application/json" \
    -d "$REQUEST_BODY")

echo "[✓] Response received:"
echo ""

# Format and display response using jq
echo "Full Response:"
echo "$RESPONSE" | jq '.'
echo ""

# Extract and format the content field if it exists
echo "[*] Extraction and Formatting of 'content' field:"
CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // empty')
if [ ! -z "$CONTENT" ]; then
    echo "$CONTENT" | jq '.'
else
    echo "No content field found in response."
fi
echo ""

echo ""
echo "========================================"
echo "Request complete"
echo "========================================"
