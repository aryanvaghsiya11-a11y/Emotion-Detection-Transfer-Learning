import pickle
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array

# --- CONFIGURATION ---
MODEL_PATH = "model.pkl"  # Your saved model file
IMAGE_PATH = "test_image.jpg"  # Replace with an actual image path
CLASS_NAMES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

def load_prediction_model(filepath):
    """Loads the Keras model from a Pickle file."""
    print(f"Loading model from {filepath}...")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_emotion(model, img_path):
    """Loads an image and predicts the emotion."""
    # 1. Read Image
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Image not found!")
        return

    # 2. Preprocess (Resize to 224x224 and Normalize)
    # MobileNetV2 expects 224x224
    image_resized = cv2.resize(image, (224, 224))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
    image_array = image_array / 255.0  # Normalize

    # 3. Predict
    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    
    # 4. Result
    class_idx = np.argmax(predictions[0])
    label = CLASS_NAMES[class_idx]
    confidence = np.max(predictions[0]) * 100

    print(f"\n✅ Prediction: {label}")
    print(f"📊 Confidence: {confidence:.2f}%")
    
    # Show Image
    cv2.putText(image, f"{label} ({confidence:.1f}%)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the function
    try:
        my_model = load_prediction_model(MODEL_PATH)
        predict_emotion(my_model, IMAGE_PATH)
    except FileNotFoundError:
        print("❌ Error: 'model.pkl' not found. Make sure it is in the same folder.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        