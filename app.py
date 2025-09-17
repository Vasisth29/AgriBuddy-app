from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from utils import get_recommendations, production_df # Import production_df as well

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- LOAD MODELS AND INDICES AT STARTUP ---
try:
    model = load_model('models/soil_model.h5')
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
        class_names = {v: k for k, v in class_indices.items()}
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load CNN model or class_indices. {e}")
    model, class_names = None, None

# --- NEW: Corrected list of States and UTs ---
STATE_NAMES_ENGLISH = [
    'Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
    'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli and Daman and Diu', 'Delhi', 'Goa',
    'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka',
    'Kerala', 'Ladakh', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
    'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]

# Corrected Hindi names mapping
STATE_NAMES_HINDI = {
    'Andaman and Nicobar Islands': 'अंडमान और निकोबार द्वीप समूह', 'Andhra Pradesh': 'आंध्र प्रदेश',
    'Arunachal Pradesh': 'अरुणाचल प्रदेश', 'Assam': 'असम', 'Bihar': 'बिहार', 'Chandigarh': 'चंडीगढ़',
    'Chhattisgarh': 'छत्तीसगढ़', 'Dadra and Nagar Haveli and Daman and Diu': 'दादरा और नगर हवेली और दमन और दीव',
    'Delhi': 'दिल्ली', 'Goa': 'गोवा', 'Gujarat': 'गुजरात', 'Haryana': 'हरियाणा', 'Himachal Pradesh': 'हिमाचल प्रदेश',
    'Jammu and Kashmir': 'जम्मू और कश्मीर', 'Jharkhand': 'झारखंड', 'Karnataka': 'कर्नाटक', 'Kerala': 'केरल',
    'Ladakh': 'लद्दाख', 'Lakshadweep': 'लक्षद्वीप', 'Madhya Pradesh': 'मध्य प्रदेश', 'Maharashtra': 'महाराष्ट्र',
    'Manipur': 'मणिपुर', 'Meghalaya': 'मेघालय', 'Mizoram': 'मिजोरम', 'Nagaland': 'नागालैंड', 'Odisha': 'ओडिशा',
    'Puducherry': 'पुडुचेरी', 'Punjab': 'पंजाब', 'Rajasthan': 'राजस्थान', 'Sikkim': 'सिक्किम',
    'Tamil Nadu': 'तमिलनाडु', 'Telangana': 'तेलंगाना', 'Tripura': 'त्रिपुरा', 'Uttar Pradesh': 'उत्तर प्रदेश',
    'Uttarakhand': 'उत्तराखंड', 'West Bengal': 'पश्चिम बंगाल'
}

# Create a list of dictionaries for the template, ensuring it only includes states present in the dataset
available_states_in_data = sorted(production_df['State_Name'].str.strip().unique())
STATES_FOR_DROPDOWN = [
    {'english': eng, 'hindi': STATE_NAMES_HINDI.get(eng, eng)}
    for eng in STATE_NAMES_ENGLISH if eng in available_states_in_data
]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    recommendations = None
    error_message = None
    selected_state = None

    if request.method == 'POST':
        img_file = request.files.get('image')
        state_name = request.form.get('state')
        selected_state = state_name

        if not img_file or not img_file.filename or not state_name:
            error_message = "Please upload a soil image and select your state."
            return render_template('index.html', states=STATES_FOR_DROPDOWN, error_message=error_message)

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(img_path)
        predicted_class = predict_soil(img_path)

        recommendations, error_message = get_recommendations(state_name, predicted_class)

        if recommendations:
            result = {
                'soil_type': predicted_class.replace('_', ' ').title(),
                'state': state_name.title()
            }

    return render_template(
        'index.html',
        states=STATES_FOR_DROPDOWN,
        result=result,
        recommendations=recommendations,
        error_message=error_message,
        selected_state=selected_state
    )

def predict_soil(img_path):
    if model is None or class_names is None:
        return "Model not loaded"
    
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        
        return class_names.get(class_idx, "Unknown Soil")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Could not process image"

if __name__ == "__main__":
    app.run(debug=True)