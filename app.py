import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from deep_translator import GoogleTranslator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)

model = load_model('plant_disease_model.h5')

# Load class index-to-name mapping
with open("class_names.json", "r") as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for name, index in class_indices.items():
    class_names[index] = name

# Base disease data in English
disease_data = {
    "Pepper__bell___Bacterial_spot": {
        "en": {
            "name": "Pepper Bell - Bacterial Spot",
            "remedy": [
                "Use disease-free seeds and resistant varieties.",
                "Avoid overhead watering and rotate crops (3–4 years).",
                "Remove infected plants and debris.",
                "Spray copper-based fungicide weekly (e.g., Copper oxychloride).",
                "Use biocontrol sprays like Bacillus subtilis (e.g., Serenade)."
            ]
        }
    },
    "Potato___Late_blight": {
        "en": {
            "name": "Potato - Late Blight",
            "remedy": [
                "Practice 2–3 year crop rotation.",
                "Avoid overhead irrigation; ensure good field drainage.",
                "Remove infected plants and destroy crop residues.",
                "Spray fungicides like Mancozeb, Chlorothalonil, or Metalaxyl at early signs.",
                "Monitor weather and apply preventively in cool, wet conditions."
            ]
        }
    },
    "Tomato__Tomato_mosaic_virus": {
        "en": {
            "name": "Tomato - Mosaic Virus",
            "remedy": [
                "Use certified virus-free seeds and resistant tomato varieties.",
                "Disinfect tools and hands regularly with bleach or milk solution.",
                "Avoid smoking or handling tobacco near plants (virus spreads via tobacco).",
                "Remove and burn infected plants immediately.",
                "Practice crop rotation and field sanitation to prevent spread."
            ]
        }
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "en": {
            "name": "Tomato - Yellow Leaf Curl Virus",
            "remedy": [
                "Use certified virus-free seeds and resistant tomato varieties.",
                "Disinfect tools and hands regularly with bleach or milk solution.",
                "Avoid smoking or handling tobacco near plants (virus spreads via tobacco).",
                "Remove and burn infected plants immediately.",
                "Practice crop rotation and field sanitation to prevent spread."
            ]
        }
    },
    "Tomato_Late_blight": {
        "en": {
            "name": "Tomato - Late Blight",
            "remedy": [
                "Use resistant tomato varieties and certified disease-free seeds.",
                "Avoid overhead watering; ensure good air circulation.",
                "Remove and destroy infected plants and debris.",
                "Apply fungicides like Mancozeb, Chlorothalonil, or Copper-based sprays at first signs.",
                "Rotate crops and avoid planting tomatoes near potatoes."
            ]
        }
    }
}

# UI text base (English)
ui_text_base = {
    "title": "LeafLens - Crop Disease Detector",
    "select_language": "Select Language:",
    "upload_leaf": "Upload Leaf Image:",
    "predict": "Predict",
    "predicted_disease": "Predicted Disease",
    "remedies": "Suggested Remedies"
}

def translate_text(text, lang):
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=lang).translate(text)
    except:
        return text  # fallback to English if translation fails

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    remedy = []
    image_path = None

    lang = request.args.get("language") or request.form.get("language") or "en"

    # Translate UI text
    ui_text = {key: translate_text(val, lang) for key, val in ui_text_base.items()}

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            image_path = os.path.join("static", file.filename)
            file.save(image_path)

            img = image.load_img(image_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            predicted_index = np.argmax(pred)
            class_name = class_names[predicted_index]

            print("Predicted class:", class_name)

            disease_info = disease_data.get(class_name, {}).get("en", {})
            eng_name = disease_info.get("name", class_name)
            eng_remedy = disease_info.get("remedy", [])

            # Translate disease name and remedies
            prediction = translate_text(eng_name, lang)
            remedy = [translate_text(r, lang) for r in eng_remedy]

    return render_template("index.html",
                           prediction=prediction,
                           remedy=remedy,
                           image_path=image_path,
                           lang=lang,
                           ui_text=ui_text)

if __name__ == "__main__":
    app.run(debug=True)
