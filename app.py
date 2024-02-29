import os
from flask import Flask, request, Response, g, render_template, jsonify
import marko
import google.generativeai as genai
import requests

# Configure generativeai
api_key= "YOUR_API_KEY"
genai.configure(api_key=api_key)

app = Flask(__name__)
app.debug = True

# Model configuration
config = {
  'temperature': 0,
  'top_k': 20,
  'top_p': 0.9,
  'max_output_tokens': 500
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",generation_config=config,safety_settings=safety_settings)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template("chat.html")


@app.route('/chat', methods=['POST'])
def chat():
    # Get image URL from request
    image_url = request.json.get('image_url')

    # Fetch image data
    image_response = requests.get(image_url)
    if image_response.status_code != 200:
        return jsonify({"error": "Failed to fetch image"})

    image_data = image_response.content

    prompt_parts = [
        "You are Zuhair, an image descriptor. You will be given an image which will be of a maths question. You need to describe the image such that a person not seeing the image but your image description can exactly get what the image means. Donot miss any detail and donot give wrong description if not sure as it might not help in problem solving. \n\nUser's image:\n\n",
        {
            "mime_type": image_response.headers.get('content-type'),
            "data": image_data
        },
    ]

    # Generate response
    response = model.generate_content(prompt_parts)

    return jsonify({
        "response": marko.convert(response.text)
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
