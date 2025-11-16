
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
    # Load the exact numeric and categorical column lists for consistent preprocessing
    numeric_cols_order = joblib.load('numeric_cols.pkl')
    categorical_cols_order = joblib.load('categorical_cols.pkl')
    # Original column names (excluding 'label') from the dataset, for consistent input ordering
    original_columns_order = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
           "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
           "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
           "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
           "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
           "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
           "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
    return model, scaler, label_encoder, categorical_encoders, numeric_cols_order, categorical_cols_order, original_columns_order

model, scaler, label_encoder, categorical_encoders, app_numeric_cols, app_categorical_cols, all_feature_columns_order = load_artifacts()

st.title("NSL-KDD Intrusion Detection")
st.write("Enter the network traffic features to predict the attack type.")

# Create input widgets for each feature
input_data = {}

# Group inputs into columns for better layout
num_cols_per_row = 3

for i, feature_name in enumerate(all_feature_columns_order):
    col_index = i % num_cols_per_row
    if col_index == 0:
        cols = st.columns(num_cols_per_row)

    with cols[col_index]:
        if feature_name in app_numeric_cols:
            # Provide more sensible default values for numeric features
            if feature_name in ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']:
                input_data[feature_name] = st.number_input(f"Enter {feature_name}", min_value=0, value=0, key=f"input_{feature_name}")
            elif 'rate' in feature_name or 'diff' in feature_name:
                input_data[feature_name] = st.number_input(f"Enter {feature_name}", min_value=0.0, max_value=1.0, value=0.0, format="%.4f", key=f"input_{feature_name}")
            else:
                input_data[feature_name] = st.number_input(f"Enter {feature_name}", value=0, key=f"input_{feature_name}")
        elif feature_name in app_categorical_cols:
            le = categorical_encoders[feature_name]
            options = list(le.classes_)
            # Attempt to set a default value if possible, otherwise use the first option
            default_index = 0
            if 'normal' in options: # Example for a common default
                default_index = options.index('normal')
            elif len(options) > 0:
                default_index = 0
            
            input_data[feature_name] = st.selectbox(f"Select {feature_name}", options, index=default_index, key=f"input_{feature_name}")
        else:
            # Fallback for unexpected types (should not be reached if lists are correct)
            input_data[feature_name] = st.text_input(f"Enter {feature_name}", "", key=f"input_{feature_name}")


if st.button("Predict Attack Type"): # Added a key to the button to prevent potential Streamlit warning
    # Convert input data to a Pandas DataFrame, ensuring column order
    input_df = pd.DataFrame([input_data])

    # Separate numeric and categorical data based on the loaded column orders
    input_numeric_df = input_df[app_numeric_cols]
    input_categorical_df = input_df[app_categorical_cols]

    # Scale numeric features
    input_scaled_numeric = scaler.transform(input_numeric_df)

    # Encode categorical features using the loaded encoders
    encoded_categorical_data_list = []
    for col in app_categorical_cols:
        le = categorical_encoders[col]
        # Handle unseen categories gracefully: if an unseen category is provided,
        # try to use a fallback or raise a more informative error.
        try:
            encoded_categorical_data_list.append(le.transform(input_categorical_df[[col]].values.ravel()))
        except ValueError as e:
            st.error(f"Error: Unseen category '{input_categorical_df[col].iloc[0]}' in column '{col}'. Please select a valid option. Details: {e}")
            st.stop() # Stop execution if an error occurs
            
    # Stack them horizontally and reshape for 1D-CNN (samples, num_features, 1)
    # Ensure the order is (scaled_numeric, encoded_categorical) matching training
    processed_input = np.hstack([input_scaled_numeric, np.array(encoded_categorical_data_list).T])

    num_features = processed_input.shape[1]
    reshaped_input = processed_input.reshape((1, num_features, 1))

    # Make prediction
    prediction = model.predict(reshaped_input)
    predicted_class_index = np.argmax(prediction)

    # Inverse transform the predicted label to get the original attack type
    predicted_label = label_encoder.inverse_transform([predicted_class_index])

    st.success(f"Predicted Attack Type: {predicted_label[0]}")

    st.subheader('Prediction Probabilities:')
    # Display top 5 probabilities
    top_5_indices = np.argsort(prediction[0])[-5:][::-1]
    top_5_labels = label_encoder.inverse_transform(top_5_indices)
    top_5_probs = prediction[0][top_5_indices]
    
    for i in range(len(top_5_labels)): # Changed from prediction_prob to prediction
        st.write(f"- {top_5_labels[i]}: {top_5_probs[i]:.4f}")
