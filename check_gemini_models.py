import os
import requests
import json

# Ensure API_KEY is set in the environment before running this script
API_KEY = os.environ.get('API_KEY')

if not API_KEY:
    print("API_KEY not set. Please set it using $env:API_KEY='YOUR_KEY_HERE' (Windows PowerShell)")
    print("or export API_KEY='YOUR_KEY_HERE' (Linux/macOS) before running this script.")
    exit(1) # Exit with an error code

list_models_endpoint = f"https://generativelanguage.googleapis.com/v1/models?key={API_KEY}"

print(f"Attempting to list models from: {list_models_endpoint}")

try:
    response = requests.get(list_models_endpoint)
    response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
    models = response.json()

    print("\n✅ Successfully retrieved models:")
    # print(json.dumps(models, indent=2)) # Uncomment this line to see the full JSON response if needed

    print("\nAvailable 'generateContent' models:")
    found_gemini_pro_family = False
    for model_info in models.get('models', []):
        if 'generateContent' in model_info.get('supported_generation_methods', []):
            print(f"- {model_info['name']}")
            if 'gemini-1.0-pro' in model_info['name'] or 'gemini-pro' in model_info['name']:
                found_gemini_pro_family = True

    if not found_gemini_pro_family:
        print("\n⚠️ WARNING: Neither 'gemini-pro' nor 'gemini-1.0-pro' found in the list of supported models for generateContent.")
        print("This means there might be an issue with your API key, Google Cloud Project setup, or region support.")
        print("Please ensure the Generative Language API is enabled for your project.")

except requests.exceptions.RequestException as e:
    print(f"\n❌ Failed to list models: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        print(f"Response content: {e.response.text}")
    print("\nPlease check your internet connection and API_KEY.")