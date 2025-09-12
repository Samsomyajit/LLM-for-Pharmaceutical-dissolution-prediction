from flask import Flask, request, jsonify
from pharmadissolve_mcp import PharmaDissolveMCP
import os

app = Flask(__name__)

# Initialize the PharmaDissolveMCP system
# Ensure the necessary environment variables are set for the model
api_key = os.getenv("OPENROUTER_API_KEY", "xxxxxxxxx")
model = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3.1:free")

# Ensure the knowledge base file exists before initializing
kb_path = "RAG_database.xlsx"
if not os.path.exists(kb_path):
    raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

system = PharmaDissolveMCP(api_key=api_key, model_name=model, kb_path=kb_path)

@app.route('/')
def health_check():
    """
    A simple endpoint to confirm that the server is running.
    """
    return "PharmaDissolveMCP server is running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    The main prediction endpoint. It expects a JSON payload with a "query" key.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid request. 'query' key is required."}), 400

    query = data['query']
    
    try:
        # Run the prediction
        outcome = system.run(query=query, n_candidates=3)
        
        # You can customize the response as needed.
        # For example, you can return the paths to the artifacts or the content of the files.
        response_data = {
            "run_id": outcome["run_id"],
            "report_path": outcome["report_path"],
            "profile_path": outcome["profile_path"],
            "qc_metrics": outcome["qc_metrics"],
            "judge": outcome["judge"],
            "sources": outcome["sources"]
        }
        
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # The server will run on http://0.0.0.0:8080
    # Use the PORT environment variable if available, otherwise default to 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
