# backend_django/app_uploads/views.py

from django.shortcuts import render
import requests
import os
import faiss
import pickle
from openai import OpenAI
import numpy as np
from nomic import embed  # For Nomic embeddings

# ===============================================================
# 1. CONFIGURE THE LLM CLIENT (Ollama)
# ===============================================================
try:
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama'
    )
    LLM_MODEL_NAME = "llama3:latest"
    print("✅ Ollama client initialized for model:", LLM_MODEL_NAME)
except Exception as e:
    print(f"❌ Error initializing Ollama client: {e}")
    client = None

# ===============================================================
# 2. FASTAPI PREDICTION SERVICE
# ===============================================================
DISEASE_API_URL = "http://127.0.0.1:8001/predict"

# ===============================================================
# 3. LOAD FAISS RAG DATABASE WITH ROBUST PICKLE HANDLING
# ===============================================================
def extract_text(obj):
    """Recursively extract text from any object, including Pydantic models."""
    texts = []
    if isinstance(obj, str):
        texts.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            texts.extend(extract_text(v))
    elif hasattr(obj, '__fields_set__'):  # Pydantic model
        texts.extend(extract_text(obj.dict()))
    elif hasattr(obj, '__dict__'):
        texts.extend(extract_text(obj.__dict__))
    else:
        texts.append(str(obj))
    return texts

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faiss_path = os.path.join(BASE_DIR, "vectorstore", "index.faiss")
    pkl_path = os.path.join(BASE_DIR, "vectorstore", "index.pkl")

    print(f"🔍 Looking for FAISS index at: {faiss_path}")
    print(f"🔍 Looking for pickle file at: {pkl_path}")

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found at {pkl_path}")

    # Load FAISS index
    faiss_index = faiss.read_index(faiss_path)

    # Load pickle with safe encoding
    with open(pkl_path, "rb") as f:
        rag_data = pickle.load(f, encoding='latin1')

    # Extract all text from the pickle object
    documents = extract_text(rag_data)
    print(f"✅ RAG database loaded with {len(documents)} documents")

except Exception as e:
    print(f"❌ Error loading RAG database: {e}")
    faiss_index = None
    documents = []

# ===============================================================
# 4. RAG SEARCH FUNCTION
# ===============================================================
def search_rag_database(query, disease_name, top_k=5):
    if faiss_index is None or not documents:
        return "RAG database not available. Using general knowledge only."

    try:
        print(f"🔍 Searching RAG for disease: {disease_name}, query: {query}")

        embedding_result = embed.text(
            texts=[f"{query} {disease_name}"],
            model='nomic-embed-text-v1'
        )
        query_embedding = embedding_result['embeddings']

        query_vector = np.array(query_embedding).astype('float32')

        scores, indices = faiss_index.search(query_vector, top_k)

        relevant_docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(documents):
                relevant_docs.append({
                    'content': documents[idx],
                    'score': scores[0][i]
                })
                print(f"📄 Found document {idx} with score: {scores[0][i]:.4f}")

        if relevant_docs:
            return "\n\n---\n\n".join([doc['content'] for doc in relevant_docs])
        else:
            return "No specific disease information found in the database."

    except Exception as e:
        print(f"❌ Error in RAG search: {e}")
        return f"Error retrieving disease information: {e}"

# ===============================================================
# 5. LLM FUNCTION WITH RAG
# ===============================================================
def get_llm_response_with_rag(user_question, disease_info):
    if client is None:
        return "LLM client not initialized. Please start the Ollama server."

    disease_name = "Unknown Disease"
    if disease_info and "error" not in disease_info:
        disease_name = disease_info.get('disease', 'Unknown Disease')

    rag_context = search_rag_database(user_question, disease_name)

    system_prompt = (
        "You are 'Agri-Intel', a friendly agricultural assistant. "
        "You help users diagnose plant diseases and provide step-by-step treatment plans. "
        "Always be professional and encouraging. Use the provided disease information from the database to give accurate advice."
    )

    if disease_info and "error" not in disease_info:
        confidence = disease_info.get('confidence', 0)
        user_prompt = (
            f"## DISEASE DIAGNOSIS FROM IMAGE ANALYSIS:\n"
            f"- **Disease Identified**: {disease_name}\n"
            f"- **Confidence Level**: {confidence*100:.0f}%\n\n"
            f"## RELEVANT DISEASE INFORMATION FROM DATABASE:\n"
            f"{rag_context}\n\n"
            f"## USER'S QUESTION:\n\"{user_question}\"\n\n"
            "## PLEASE PROVIDE:\n"
            "1. Diagnosis confirmation\n"
            "2. Step-by-step treatment plan\n"
            "3. Prevention strategies\n"
            "4. Additional advice\n\n"
            "Mention if database info contradicts the diagnosis."
        )
    else:
        user_prompt = (
            f"## RELEVANT DATABASE INFORMATION:\n{rag_context}\n\n"
            f"## USER'S QUESTION:\n\"{user_question}\"\n\n"
            "Provide a helpful, accurate answer using the above info."
        )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting LLM response: {e}"

# ===============================================================
# 6. UPLOAD VIEW
# ===============================================================
def upload_view(request):
    if request.method == "POST" and 'photo' in request.FILES:
        image = request.FILES['photo']
        files = {'image': (image.name, image.read(), image.content_type)}
        user_question = request.POST.get("question", "").strip()

        # --- FastAPI prediction ---
        disease_data = None
        try:
            response = requests.post(DISEASE_API_URL, files=files, timeout=15)
            response.raise_for_status()
            disease_data = response.json()
        except requests.exceptions.RequestException as e:
            disease_data = {"error": f"FastAPI server not reachable: {e}"}

        # --- LLM + RAG response ---
        final_llm_answer = get_llm_response_with_rag(user_question, disease_data)

        print("✅ Disease data:", disease_data)
        print("✅ LLM answer length:", len(final_llm_answer))

        return render(request, "result.html", {
            "disease_info": disease_data,
            "final_answer": final_llm_answer,
            "question": user_question,
        })

    return render(request, "upload.html")
