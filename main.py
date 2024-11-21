import os
import logging
from flask import Flask, request, jsonify
import openai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    try:
        # Parse request JSON
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        # Log the incoming request
        logger.info(f"Request data: {request_data}")
        
        # Send to OpenAI API
        response = openai.ChatCompletion.create(
            model=request_data.get("model", "gpt-3.5-turbo"),
            messages=request_data.get("messages", []),
            max_tokens=request_data.get("max_tokens", 100),
            temperature=request_data.get("temperature", 0.7),
            stream=request_data.get("stream", False)
        )
        
        # Return response
        return jsonify(response)
    
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Custom LLM API!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
