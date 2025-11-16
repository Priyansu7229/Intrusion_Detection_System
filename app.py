
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the saved model and preprocessors
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('best_cnn_nslkdd.h5')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    categorical_encoders = joblib.load('categorical_encoders.pkl')
    return model, scaler, label_encoder, categorical_encoders

model, scaler, label_encoder, categorical_encoders = load_artifacts()

# Column names for NSL-KDD (41 features, excluding 'label')
columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
           "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
           "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
           "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
           "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
           "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
           "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]

# Re-create numeric and categorical column lists based on the categorical_encoders keys
# This ensures consistency with how data was actually processed during training.
app_categorical_cols = list(categorical_encoders.keys())
app_numeric_cols = [col for col in columns if col not in app_categorical_cols]

st.title("NSL-KDD Intrusion Detection")
st.write("Enter the network traffic features to predict the attack type.")

# Create input widgets for each feature
input_data = {}

# Group inputs into columns for better layout
num_cols = 3
rows = (len(columns) + num_cols - 1) // num_cols

for r in range(rows):
    cols = st.columns(num_cols)
    for c_idx in range(num_cols):
        col_idx = r * num_cols + c_idx
        if col_idx < len(columns):
            feature_name = columns[col_idx]
            with cols[c_idx]:
                if feature_name in app_numeric_cols:
                    # Default values for numeric columns
                    if feature_name == 'duration':
                        input_data[feature_name] = st.number_input(f"Enter {feature_name}", min_value=0, value=0)
                    elif feature_name in ['src_bytes', 'dst_bytes']:
                        input_data[feature_name] = st.number_input(f"Enter {feature_name}", min_value=0, value=0)
                    elif 'rate' in feature_name or 'diff' in feature_name:
                        input_data[feature_name] = st.number_input(f"Enter {feature_name}", min_value=0.0, max_value=1.0, value=0.0, format="%.4f")
                    else:
                        # For other numeric columns, provide a generic number input
                        input_data[feature_name] = st.number_input(f"Enter {feature_name}", value=0)
                elif feature_name in app_categorical_cols:
                    le = categorical_encoders[feature_name]
                    # Use a default value from the encoder's classes
                    default_value = le.inverse_transform([0])[0] if len(le.classes_) > 0 else "unknown"
                    input_data[feature_name] = st.selectbox(f"Select {feature_name}", le.classes_, index=list(le.classes_).index(default_value) if default_value in le.classes_ else 0)
                else:
                    # Fallback for unexpected types, should not be reached if lists are correct
                    input_data[feature_name] = st.text_input(f"Enter {feature_name}", "")


if st.button("Predict Attack Type"):
    # Convert input data to a Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocessing steps mirroring the training script
    # Separate numeric and categorical data
    input_numeric = input_df[app_numeric_cols]
    input_categorical = input_df[app_categorical_cols]

    # Scale numeric features
    input_scaled_numeric = scaler.transform(input_numeric)

    # Encode categorical features using the loaded encoders
    encoded_categorical_data = []
    for col in app_categorical_cols:
        le = categorical_encoders[col]
        encoded_categorical_data.append(le.transform(input_categorical[[col]].values.ravel()))
    
    # Combine scaled numeric and encoded categorical features
    processed_input = np.hstack([input_scaled_numeric, np.array(encoded_categorical_data).T])

    # Reshape for 1D-CNN (samples, num_features, 1)
    num_features = processed_input.shape[1]
    reshaped_input = processed_input.reshape((1, num_features, 1))

    # Make prediction
    prediction = model.predict(reshaped_input)
    predicted_class_index = np.argmax(prediction)

    # Inverse transform the predicted label to get the original attack type
    predicted_label = label_encoder.inverse_transform([predicted_class_index])

    st.success(f"Predicted Attack Type: {predicted_label[0]}")
