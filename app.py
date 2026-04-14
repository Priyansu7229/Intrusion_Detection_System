import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Re-define the model architecture
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(78, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(15, activation='softmax') # Assuming 15 classes, adjust if needed
])

# Load the saved model weights
model.load_weights('IDS_model_weights.weights.h5')

# Load the scaler
loaded_scaler = joblib.load('scaler.save')

# Load the label encoder
loaded_label_encoder = joblib.load('label_encoder_final.save')

# Streamlit App
st.title('Network Intrusion Detection System (NIDS) CNN Model')
st.write('Upload a CSV file containing network traffic features for intrusion detection.')

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        input_df = pd.read_csv(uploaded_file)
        st.write('Uploaded Data Preview:')
        st.write(input_df.head())

        # Scale the features
        input_scaled = loaded_scaler.transform(input_df.values)

        # Reshape for CNN model (add channel dimension)
        input_cnn = input_scaled.reshape(input_scaled.shape[0], input_scaled.shape[1], 1)

        # Make predictions
        predictions = model.predict(input_cnn)
        predicted_classes_encoded = np.argmax(predictions, axis=1)
        predicted_labels = loaded_label_encoder.inverse_transform(predicted_classes_encoded)

        st.write('Prediction Results:')
        result_df = input_df.copy()
        result_df['Predicted_Attack_Type'] = predicted_labels
        st.write(result_df[['Flow Duration', 'Destination Port', 'Predicted_Attack_Type']].head())

        # Display value counts for predicted labels
        st.write('Predicted Attack Type Distribution:')
        st.write(pd.Series(predicted_labels).value_counts())

    except Exception as e:
        st.error(f'Error processing file: {e}')
