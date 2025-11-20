
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the trained model and scaler
best_gb_model = joblib.load('gradient_boosting_model.joblib')
scaler = joblib.load('scaler.joblib')

# 2. Define categorical and numerical column names used during training
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_cols = {
    'Sex': ['F', 'M'],
    'ChestPainType': ['ASY', 'ATA', 'NAP', 'TA'],
    'RestingECG': ['LVH', 'Normal', 'ST'],
    'ExerciseAngina': ['N', 'Y'],
    'ST_Slope': ['Down', 'Flat', 'Up']
}

# Define the full list of features in the exact order the model expects
# This needs to match the columns of X_train used for model training
model_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']

# 3. Streamlit App Layout
st.set_page_config(page_title="Heart Disease Prediction App created by Ananth Sundarrajan", layout='centered')
st.title("Heart Disease Prediction")
st.write("Enter the patient's details to predict the likelihood of heart disease.")

# Input fields for user data
st.header('Patient Information')

# Numerical inputs
age = st.slider('Age (in years)', 18, 100, 50)
resting_bp = st.slider('Resting Systolic Blood Pressure (mmHg)', 80, 200, 120)
cholesterol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
max_hr = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
oldpeak = st.slider('Oldpeak (ST depression induced by exercise relative to rest found on your ECG)', 0.0, 6.2, 1.0)

# Categorical inputs
sex = st.selectbox('Sex', options=categorical_cols['Sex'])
chest_pain_type = st.selectbox('Chest Pain Type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)', options=categorical_cols['ChestPainType'])
resting_ecg = st.selectbox('Resting ECG Results (Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes criteria)', options=categorical_cols['RestingECG'])
exercise_angina = st.selectbox('Exercise Induced Angina', options=categorical_cols['ExerciseAngina'])
st_slope = st.selectbox('ST_Slope (the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping])', options=categorical_cols['ST_Slope'])

# 4. Prediction Logic
if st.button('Predict Heart Disease'):
    # Collect inputs into a dictionary
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Apply one-hot encoding to categorical features
    # Create dummy columns for all possible categorical values to ensure consistency
    for col, categories in categorical_cols.items():
        for cat in categories:
            if col != 'Sex' or cat != 'F': # For Sex, 'F' is the reference, so Sex_M is created if Sex is M
                if col != 'ChestPainType' or cat != 'ASY':
                    if col != 'RestingECG' or cat != 'LVH':
                        if col != 'ExerciseAngina' or cat != 'N':
                            if col != 'ST_Slope' or cat != 'Down':
                                input_df[f'{col}_{cat}'] = (input_df[col] == cat).astype(int)

    # For columns where drop_first=True was used during training, we need to handle reference categories explicitly.
    # 'Sex_M' is created if Sex is 'M'. 'F' is the reference.
    input_df['Sex_M'] = (input_df['Sex'] == 'M').astype(int)

    # The following categorical columns are dropped after creating their one-hot encoded counterparts.
    # The specific one-hot encoded columns (e.g., ChestPainType_ATA) correspond to the non-reference categories.
    # The reference categories (e.g., ASY for ChestPainType) are implicitly handled by the absence of their one-hot column.
    input_df = input_df.drop(columns=list(categorical_cols.keys()))

    # Ensure all model_features are present, fill missing one-hot encoded columns with 0
    for feature in model_features:
        if feature not in input_df.columns:
            # This primarily handles cases for one-hot encoded columns that were not created
            # because their corresponding categorical value wasn't selected by the user
            # e.g., if 'ChestPainType_TA' does not exist, create it and set to 0
            input_df[feature] = 0

    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Reorder columns to match the model's expected input order
    final_input_df = input_df[model_features]

    # Make prediction
    prediction = best_gb_model.predict(final_input_df)
    prediction_proba = best_gb_model.predict_proba(final_input_df)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'Based on the provided information, Heart Disease is Predicted. (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'Based on the provided information, Heart Disease is NOT Predicted. (Probability: {prediction_proba[0]:.2f})')
