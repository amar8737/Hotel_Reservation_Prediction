import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from config.paths_config import MODEL_DIR


app = Flask(__name__)

# --- Load the Model ---
# Construct the full path to the model file
try:
    model_path = os.path.join(MODEL_DIR, 'lgbm_model.pkl')
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    # Handle the error appropriately, maybe exit or use a dummy model
    model = None 

# --- Define Routes ---

@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, makes a prediction, and returns the result."""
    if model is None:
        return render_template('index.html', prediction_text="Error: Model is not loaded.")

    try:
        # 1. Define the exact feature names and order from your training logs
        model_columns = [
            'lead_time', 'no_of_special_requests', 'avg_price_per_room',
            'arrival_month', 'arrival_date', 'market_segment_type',
            'no_of_week_nights', 'no_of_weekend_nights', 'type_of_meal_plan',
            'room_type_reserved'
        ]

        # 2. Collect form data into a dictionary, casting to the correct data types
        features_dict = {
            'lead_time': int(request.form['lead_time']),
            'no_of_special_requests': int(request.form['no_of_special_requests']),
            'avg_price_per_room': float(request.form['avg_price_per_room']),
            'arrival_month': int(request.form['arrival_month']),
            'arrival_date': int(request.form['arrival_date']),
            'market_segment_type': int(request.form['market_segment_type']),
            'no_of_week_nights': int(request.form['no_of_week_nights']),
            'no_of_weekend_nights': int(request.form['no_of_weekend_nights']),
            'type_of_meal_plan': int(request.form['type_of_meal_plan']),
            'room_type_reserved': int(request.form['room_type_reserved'])
        }

        # 3. Create a Pandas DataFrame from the dictionary
        final_features = pd.DataFrame([features_dict])

        # 4. Ensure the DataFrame columns are in the same order as the model expects
        final_features = final_features[model_columns]

        # 5. Make prediction and get probabilities
        prediction = model.predict(final_features)
        prediction_proba = model.predict_proba(final_features)

        # 6. Format the output for the user
        output = 'Cancelled' if prediction[0] == 1 else 'Not Cancelled'
        confidence = prediction_proba[0][prediction[0]] * 100

        return render_template('index.html',
                               prediction_text=f'Booking Status: {output}',
                               confidence_text=f'Confidence: {confidence:.2f}%')

    except (KeyError, ValueError) as e:
        # Handle cases where a form field is missing or has the wrong data type
        error_message = f"Invalid input. Please ensure all fields are filled correctly. Error: {e}"
        return render_template('index.html', prediction_text=error_message)


# --- Run the App ---

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)