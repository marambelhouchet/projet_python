from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import io
# --- Import keras directly ---
import tensorflow.keras as keras # Changed import
import json
import os # Added for path joining

app = FastAPI(title="Agri-Intel Disease Predictor")

# --- Load the REAL Model and Class Names ---
# Use os.path.join for better path handling
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_model_best.h5") # The NEW model trained without preprocessing layer
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")
model = None
class_names = None

@app.on_event("startup")
async def startup_event():
    global model, class_names
    # Check if files exist before attempting to load
    if not os.path.exists(MODEL_PATH):
        print(f"!!! FATAL ERROR: Model file not found at {MODEL_PATH} !!!")
        print("!!! Please ensure the trained model is downloaded from Colab and placed here. !!!")
        return # Stop startup if model file is missing
    if not os.path.exists(CLASSES_PATH):
        print(f"!!! FATAL ERROR: Class names file not found at {CLASSES_PATH} !!!")
        print("!!! Please ensure the class names file is downloaded from Colab and placed here. !!!")
        return # Stop startup if class names file is missing

    try:
        # --- Load model normally ---
        # No scope or compile=False needed now because preprocessing layer was removed before saving
        print(f"Attempting to load model from: {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
        print("Model structure loaded.")

        # --- Load class names ---
        print(f"Attempting to load class names from: {CLASSES_PATH}")
        with open(CLASSES_PATH, 'r') as f:
            class_names = json.load(f)
        print("--- Model and class names loaded successfully ---")
        print(f"Model input shape expected: {model.input_shape}")
        print(f"Number of classes loaded: {len(class_names)}")

    except Exception as e:
        print(f"!!! ERROR DURING STARTUP LOADING: {e} !!!")
        print(f"!!! Could not load model: {MODEL_PATH} or classes: {CLASSES_PATH} !!!")
        print("!!! Check file paths and TensorFlow/Keras compatibility. Ensure files are not corrupted. !!!")
        model = None # Ensure model is None if loading failed
        class_names = None

# --- This function MUST still apply preprocessing ---
def preprocess_image(image: Image.Image):
    # 1. Resize the image
    image = image.resize((224, 224))
    # 2. Convert to NumPy array (ensure 3 channels if needed)
    image_array = np.asarray(image)
    if image_array.ndim == 2: # Handle grayscale images by converting to RGB
        print("Preprocessing: Image was grayscale, converting to RGB.")
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] == 4: # Handle RGBA images by removing alpha channel
        print("Preprocessing: Image was RGBA, removing alpha channel.")
        image_array = image_array[:,:,:3]
    elif image_array.shape[2] != 3:
         raise ValueError(f"Unexpected number of channels: {image_array.shape[2]}")

    # Ensure shape is (224, 224, 3)
    if image_array.shape != (224, 224, 3):
        raise ValueError(f"Incorrect image shape after conversion: {image_array.shape}")

    # 3. Expand dimensions to create a "batch" of 1
    image_batch = np.expand_dims(image_array, axis=0)
    # 4. Apply MobileNetV2 preprocessing (crucial!) - STILL NEEDED HERE
    # print(f"Preprocessing: Input batch shape: {image_batch.shape}, dtype: {image_batch.dtype}")
    processed_batch = keras.applications.mobilenet_v2.preprocess_input(image_batch.astype(np.float32)) # Ensure float32
    # print(f"Preprocessing: Output batch shape: {processed_batch.shape}, dtype: {processed_batch.dtype}, min: {processed_batch.min()}, max: {processed_batch.max()}")
    return processed_batch # Return the preprocessed batch

@app.post("/predict")
async def create_prediction(image: UploadFile = File(...)):
    global model, class_names

    if model is None or class_names is None:
        print("Prediction request failed: Model or class names not loaded.")
        return {"error": "Model is not loaded. Check server startup logs."}, 503

    # 1. Read the image file
    try:
        contents = await image.read()
        if not contents:
            return {"error": "Received empty image file."}, 400
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB') # Ensure image is RGB
        print(f"Prediction request: Received image '{image.filename}', size {len(contents)} bytes.")
    except Exception as e:
        print(f"Prediction error: Failed to read or open image '{image.filename}': {e}")
        return {"error": f"Failed to read or open image: {e}"}, 400

    # 2. Preprocess the image (includes mobilenet preprocessing now)
    try:
        processed_image = preprocess_image(pil_image)
        # print(f"Prediction request: Preprocessed image shape: {processed_image.shape}")
    except Exception as e:
        print(f"Prediction error: Failed to preprocess image '{image.filename}': {e}")
        return {"error": f"Failed to preprocess image: {e}"}, 500

    # 3. Run prediction
    try:
        predictions = model.predict(processed_image)
        # print(f"Prediction request: Raw predictions shape: {predictions.shape}")

        # 4. Get the top prediction
        predicted_index = np.argmax(predictions[0])
        if predicted_index >= len(class_names):
             print(f"Prediction error: Predicted index {predicted_index} out of bounds for {len(class_names)} classes.")
             return {"error": "Prediction resulted in an invalid class index."}, 500

        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions[0]))
        confidence_pct = round(confidence * 100, 2)
        print(f"Prediction result: Class='{predicted_class}', Confidence={confidence_pct}%")


        # 5. Return the result
        return {
            "disease": predicted_class,
            "confidence": confidence_pct # Return percentage
        }
    except Exception as e:
        print(f"!!! PREDICTION FAILED: {e} !!!") # Log error server-side
        # Optionally include more details from e in the log, but not the user response
        return {"error": f"Prediction failed due to an internal server error." }, 500


if __name__ == "__main__":
    # Use reload=True for development, but remove it for production
    print("Starting Uvicorn server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)

