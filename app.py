import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
from class_labels import CLASS_NAMES
import io

app = Flask(__name__)

MODEL_PATH = 'hair_disease_model.keras'
IMG_WIDTH, IMG_HEIGHT = 224, 224

# --- Remedies & Tips Database ---
DISEASE_INFO = {
    "Alopecia Areata": {
        "remedies": [
            "Apply onion juice or coconut oil to stimulate follicles.",
            "Include protein-rich foods and biotin supplements.",
            "Massage scalp daily to improve blood circulation."
        ],
        "donts": [
            "Avoid harsh shampoos or heat treatments.",
            "Don’t stress excessively — it worsens autoimmune response."
        ]
    },
    "Contact Dermatitis": {
        "remedies": [
            "Avoidance: If you identify what caused the rash, avoid or minimize exposure to it.",
            "Apply over-the-counter anti-itch creams (like hydrocortisone) or calamine lotion.",
            "Use cool, wet compresses to soothe irritation and swelling."
        ],
        "donts": [
            "Avoid harsh soaps, fragranced products, or chemicals on the affected area.",
            "Do not scratch the rash; scratching increases inflammation and risk of infection."
        ]
    },
    "Folliculitis": {
        "remedies": [
            "Using antibacterial cleansers (e.g., with benzoyl peroxide) to clean the skin.",
            "Applying warm, moist towels to the irritated skin several times a day to soothe discomfort and encourage drainage.",
            "Using anti-itch creams if irritation is severe."
        ],
        "donts": [
            "Avoid oily hair products, which can clog follicles.",
            "Do not scratch or shave the affected area vigorously."
        ]
    },
    "Head Lice": {
        "remedies": [
            "Apply medicated lice shampoo (pyrethrin or permethrin) or natural oils (neem or tea tree) as directed.",
            "Comb hair with a fine-toothed nit comb daily to remove all nits (eggs) and lice.",
            "Wash bedding, towels, and clothing in hot water and dry on high heat."
        ],
        "donts": [
            "Avoid sharing combs, brushes, hats, or towels.",
            "Do not delay treatment; lice multiply fast and are easily spread."
        ]
    },
    "Psoriasis": {
        "remedies": [
            "Use coal tar or salicylic acid shampoo to reduce scaling.",
            "Apply coconut oil, aloe vera, or a mild moisturizer to soothe dry skin.",
            "Stay hydrated, manage stress, and ensure adequate Vitamin D exposure (with caution)."
        ],
        "donts": [
            "Avoid scratching or picking at flakes, as this can worsen the plaques (Koebner phenomenon).",
            "Stay away from known stress and environmental triggers (like cold, dry weather)."
        ]
    },
    "Healthy Hair": {
        "remedies": [
            "Maintain a balanced diet rich in vitamins (A, C, D, E, B-vitamins) and minerals (Iron, Zinc).",
            "Wash hair regularly with mild, sulfate-free shampoo.",
            "Gently massage and oil your scalp weekly to improve blood circulation."
        ],
        "donts": [
            "Avoid harsh chemicals (bleach, relaxers) or excessive heat styling.",
            "Do not use tight hairstyles (braids, ponytails) that pull on the roots."
        ]
    },
    "Lichen Planus": {
        "remedies": [
            "Use a mild, sulfate-free shampoo to gently cleanse the scalp.",
            "Apply cool compresses to the scalp to relieve severe itching and inflammation.",
            "Practice stress management techniques like meditation or exercise, as stress can worsen symptoms."
        ],
        "donts": [
            "Avoid scratching, rubbing, or injuring the scalp, as this can trigger new flare-ups.",
            "Limit heat styling (blow drying, flat ironing) and chemical treatments (coloring/perming)."
        ]
    },
    "Male Pattern Baldness": {
        "remedies": [
            "Perform daily scalp massages to increase blood circulation to the follicles.",
            "Use over-the-counter treatments like Minoxidil (Rogaine) as directed.",
            "Consume a diet rich in protein, iron, zinc, and Omega-3 fatty acids."
        ],
        "donts": [
            "Avoid harsh chemical-laden shampoos that strip the scalp.",
            "Don't ignore the problem; early intervention with treatments is most effective."
        ]
    },
    "Seborrheic Dermatitis": {
        "remedies": [
            "Wash affected areas regularly with medicated shampoo containing **ketoconazole**, **selenium sulfide**, or **zinc pyrithione**.",
            "Soften and remove thick scales by applying mineral or olive oil for an hour before washing.",
            "Use fragrance-free, alcohol-free hair and skin care products."
        ],
        "donts": [
            "Avoid scratching or picking at the flakes, which increases the risk of infection.",
            "Do not use heavy, oily styling products during a flare-up, as they can worsen the condition."
        ]
    },
    "Telogen Effluvium": {
        "remedies": [
            "Identify and address the underlying cause (e.g., stress, nutritional deficiency, medication change).",
            "Ensure a high intake of protein, iron, and B vitamins (especially Biotin).",
            "Be extremely gentle with hair: use a wide-tooth comb and avoid tight hairstyles."
        ],
        "donts": [
            "Avoid excessive heat styling, chemical treatments, or rigorous brushing while shedding is active.",
            "Do not engage in extreme or restrictive diets, as rapid weight loss is a common trigger."
        ]
    },
    "Tinea Capitis": {
        "remedies": [
            "This condition requires **oral antifungal medication** prescribed by a doctor; topical creams alone will not cure it.",
            "Use a medicated shampoo (like **selenium sulfide**) as an adjunct to the oral medication to prevent spreading.",
            "Wash all bedding, towels, and clothes in hot water to kill the fungus."
        ],
        "donts": [
            "Do not stop oral medication early, even if symptoms improve (must complete the full course).",
            "Avoid sharing combs, hats, towels, or other personal items to prevent spreading the highly contagious fungus."
        ]
    }
}
# --- Load Model ---
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None

# --- Image Preprocessing ---
def preprocess_image(image_file):
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Please upload an image.")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No image selected.")

        try:
            processed = preprocess_image(file)
            predictions = model.predict(processed)
            predicted_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            predicted_class = CLASS_NAMES[predicted_index]

            # Fetch remedies & tips
            disease_data = DISEASE_INFO.get(predicted_class, {})
            result = {
                "class_name": predicted_class,
                "confidence": f"{confidence:.2f}%",
                "remedies": disease_data.get("remedies", []),
                "donts": disease_data.get("donts", [])
            }

            return render_template('index.html', result=result)

        except Exception as e:
            return render_template('index.html', error=f"Prediction error: {e}")

    return render_template('index.html')

if __name__ == '__main__':
    # Set host to '0.0.0.0' for deployment flexibility
    # Set debug=False for production
    app.run(debug=True)
