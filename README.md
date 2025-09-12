# PharmaDissolveMCP Server

This document provides instructions on how to deploy and run the PharmaDissolveMCP server.

## Prerequisites

- Python 3.8+
- `pip` for installing packages

## 1. Setup

First, create and activate a virtual environment to isolate the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Installation

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## 3. Environment Variables

Before running the server, you need to configure the following environment variables. These are essential for connecting to the language model service.

```bash
# Set your API key for the LLM provider
export OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Set the model identifier
export LLM_MODEL="deepseek/deepseek-chat-v3.1:free"

# (Optional) Set these if required by your provider or for tracking
export OPENROUTER_SITE_URL="http://localhost"
export OPENROUTER_APP_TITLE="PharmaDissolve-MCP"
```

## 4. Running the Server

Once the environment is configured, you can start the Flask server:

```bash
python serve.py
```

The server will start on port 8080 by default. You can access it at `http://localhost:8080`.

## 5. Using the API

You can send a POST request to the `/predict` endpoint with a JSON payload to get a prediction.

**Endpoint:** `http://localhost:8080/predict`

**Method:** `POST`

**Body:**
```json
{
  "query": "Predict dissolution profile for Ibuprofen 200mg tablet"
}
```

**Example using `curl`:**
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"query": "Predict dissolution profile for Ibuprofen 200mg tablet"}' \
http://localhost:8080/predict
```
