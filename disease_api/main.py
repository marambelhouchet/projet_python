from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow.keras as keras
import json
import tensorflow as tf
import os

app = FastAPI(title="Agri-Intel Disease Predictor")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_model_best.h5")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")

model = None
class_names = None

# More comprehensive custom objects
custom_objects = {
    'TrueDivide': tf.keras.layers.Lambda(lambda x: x / 255.0),
    'Rescaling': tf.keras.layers.Rescaling(1./255),
    'Cast': tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
    'tf': tf,
}

@app.on_event("startup")
async def startup_event():
    global model, class_names
    try:
        print(f"Loading model from: {MODEL_PATH}")
        print(f"Loading classes from: {CLASSES_PATH}")
        
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at: {MODEL_PATH}")
            return
        if not os.path.exists(CLASSES_PATH):
            print(f"Class names file not found at: {CLASSES_PATH}")
            return

        # Try multiple loading strategies
        load_success = False
        
        # Strategy 1: Load with custom objects and compile=False
        try:
            print("Attempting Strategy 1: Custom objects with compile=False")
            with keras.utils.custom_object_scope(custom_objects):
                model = keras.models.load_model(MODEL_PATH, compile=False)
            load_success = True
            print("✓ Model loaded with Strategy 1")
        except Exception as e1:
            print(f"Strategy 1 failed: {e1}")
            
            # Strategy 2: Load with safe_mode=False
            try:
                print("Attempting Strategy 2: safe_mode=False")
                with keras.utils.custom_object_scope(custom_objects):
                    model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
                load_success = True
                print("✓ Model loaded with Strategy 2")
            except Exception as e2:
                print(f"Strategy 2 failed: {e2}")
                
                # Strategy 3: Try loading weights only (requires knowing architecture)
                try:
                    print("Attempting Strategy 3: Manual model reconstruction")
                    load_success = load_model_manually()
                    if load_success:
                        print("✓ Model loaded with Strategy 3")
                except Exception as e3:
                    print(f"Strategy 3 failed: {e3}")

        if load_success:
            # Load class names
            with open(CLASSES_PATH, 'r') as f:
                class_names = json.load(f)
            
            print("--- Model and class names loaded successfully ---")
            print(f"Model input shape: {model.input_shape}")
            print(f"Number of classes: {len(class_names)}")
            print(f"Available classes: {class_names}")
        else:
            print("!!! All loading strategies failed !!!")
            
    except Exception as e:
        print(f"!!! ERROR IN STARTUP: {e} !!!")

def load_model_manually():
    """Manual model loading as last resort"""
    global model
    try:
        # Create a simple MobileNetV2 based model (adjust based on your architecture)
        base_model = keras.applications.MobileNetV2(
            weights=None,
            input_shape=(224, 224, 3),
            include_top=False,
            pooling='avg'
        )
        
        # Add custom classification head
        x = keras.layers.Dense(128, activation='relu')(base_model.output)
        predictions = keras.layers.Dense(len(class_names), activation='softmax')(x)
        
        model = keras.Model(inputs=base_model.input, outputs=predictions)
        
        # Load weights (you'd need a weights file for this)
        # model.load_weights(MODEL_PATH.replace('.h5', '_weights.h5'))
        
        return False  # Change to True if you have weights file
    except Exception as e:
        print(f"Manual loading failed: {e}")
        return False

def preprocess_image(image: Image.Image):
    """Preprocess image for MobileNetV2"""
    # Resize to match model expected input
    image = image.resize((224, 224))
    
    # Convert to numpy array
    image_array = np.asarray(image)
    
    # Handle different image formats
    if image_array.ndim == 2:  # Grayscale
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = image_array[:,:,:3]
    
    # Expand dimensions and apply MobileNetV2 preprocessing
    image_batch = np.expand_dims(image_array, axis=0)
    processed_batch = keras.applications.mobilenet_v2.preprocess_input(image_batch.astype(np.float32))
    
    return processed_batch

@app.get("/")
async def root():
    return {"message": "Agri-Intel Disease Predictor API is running"}

@app.get("/health")
async def health_check():
    if model is None or class_names is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {
        "status": "healthy", 
        "model_loaded": True, 
        "classes_count": len(class_names),
        "input_shape": model.input_shape if model else None
    }

@app.post("/predict")
async def create_prediction(image: UploadFile = File(...)):
    global model, class_names

    if model is None or class_names is None:
        return {"error": "Model is not loaded. Check server logs."}, 503

    try:
        contents = await image.read()
        if len(contents) == 0:
            return {"error": "Empty image file"}, 400
            
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return {"error": f"Failed to read or open image: {e}"}, 400

    try:
        processed_image = preprocess_image(pil_image)
    except Exception as e:
        return {"error": f"Failed to preprocess image: {e}"}, 500

    try:
        predictions = model.predict(processed_image)
        
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions[0]))
        confidence_pct = round(confidence * 100, 2)

        return {
            "disease": predicted_class,
            "confidence": confidence_pct
        }
        
    except Exception as e:
        print(f"!!! PREDICTION FAILED: {e} !!!")
        return {"error": "Prediction failed due to an internal server error."}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)