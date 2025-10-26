from django.shortcuts import render
import requests
import os
from openai import OpenAI  # <-- Keep this import

# ===============================================================
# 1. CONFIGURE THE LLM CLIENT (Ollama)
# ===============================================================
try:
    # Point to your local Ollama API endpoint (default port 11434)
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',  # Ollama doesn’t actually check this key
    )
    LLM_MODEL_NAME = "llama3:latest"  # Change this if using another model
    print("✅ Ollama client initialized for model:", LLM_MODEL_NAME)
except Exception as e:
    print(f"❌ Error initializing Ollama client: {e}")
    print("⚠️ Make sure your Ollama server is running (use 'ollama serve').")
    client = None

# ===============================================================
# 2. CONFIGURE FASTAPI ENDPOINT (DISEASE PREDICTION API)
# ===============================================================
DISEASE_API_URL = "http://127.0.0.1:8001/predict"


# ===============================================================
# 3. HELPER FUNCTION: ASK LLM FOR ADDITIONAL INFO OR ADVICE
# ===============================================================
def get_llm_response(user_question, disease_info):
    """
    Sends a message to the LLM (via Ollama) combining user question and model output.
    Returns a generated response string or an error message.
    """
    if client is None:
        return "LLM client not initialized. Please start the Ollama server."

    # --- Create the Prompt ---
    system_prompt = (
        "You are 'Agri-Intel', a friendly and expert agricultural assistant. "
        "Your job is to help users identify plant diseases and give them clear, "
        "actionable, step-by-step treatment plans. "
        "Always be encouraging and professional."
    )
    
    user_prompt = ""
    
    if disease_info and "error" not in disease_info:
        # Case 1: Image was provided AND successfully analyzed
        disease_name = disease_info.get('disease', 'Unknown Disease')
        confidence = disease_info.get('confidence', 0)
        
        user_prompt = (
            f"A user has uploaded a photo of their plant. My vision model has identified "
            f"the disease as '{disease_name}' with {confidence*100:.0f}% confidence.\n\n"
            f"The user's question is: '{user_question}'\n\n"
            "Please do the following:\n"
            "1. Gently confirm the disease (e.g., 'It looks like your plant has...').\n"
            "2. Provide a clear, step-by-step treatment plan (e.g., pruning, fungicides, etc.).\n"
            "3. Give advice on how to prevent this disease in the future."
        )
    else:
        # Case 2: No image, or image analysis failed
        user_prompt = (
            f"A user did not upload an image (or it failed to process). "
            f"They have a general agriculture question.\n\n"
            f"Their question is: '{user_question}'\n\n"
            "Please answer their question based on your general agricultural knowledge."
        )

    # --- Call the Ollama API ---
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting LLM response: {e}"


# ===============================================================
# 4. MAIN VIEW: HANDLE UPLOAD + FASTAPI + LLM RESPONSE
# ===============================================================
def upload_view(request):
    """
    Handles uploading a photo and sending it to the FastAPI model.
    Renders 'upload.html' for GET requests.
    Renders 'result.html' for POST requests after sending the photo to FastAPI.
    """
    if request.method == "POST" and 'photo' in request.FILES:
        # Get uploaded photo
        image = request.FILES['photo']

        # Prepare the file to send to FastAPI
        # FIX: The key 'image' must match your FastAPI 'create_prediction' parameter
        files = {'image': (image.name, image.read(), image.content_type)} 

        # Get user question from the form
        user_question = request.POST.get("question", "").strip()

        # Send image to FastAPI prediction service
        disease_data = None # Use this variable to store API response or error
        try:
            response = requests.post(DISEASE_API_URL, files=files, timeout=10)
            response.raise_for_status()
            disease_data = response.json()
        except requests.exceptions.RequestException as e:
            disease_data = {"error": f"FastAPI server not reachable: {e}"}

        # Ask LLM for advice
        # Pass the user's question and the disease data to the LLM
        final_llm_answer = get_llm_response(user_question, disease_data)

        # Render the result page
        # FIX: Change variable names to match result.html
        return render(request, "result.html", {
            "disease_info": disease_data,      # Was "result"
            "final_answer": final_llm_answer,  # Was "llm_reply"
            "question": user_question,         # Was "user_question"
        })

    # For GET request — show the upload form
    return render(request, "upload.html")