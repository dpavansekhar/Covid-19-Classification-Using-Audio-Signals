import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

# Load the saved model
def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("cnn_model.weights.h5")
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return loaded_model

model = load_model()

# Function to extract features from a given audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y, top_db=20)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rms_energy = np.mean(librosa.feature.rms(y=y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)

    features = list(mfccs_mean) + [zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff, rms_energy]
    
    return features

# Function to predict if the given audio signal contains COVID-19 cough or not
def predict_covid(file_path):
    features = extract_features(file_path)
    features = np.array(features).reshape(1, -1)

    # Load the StandardScaler used during training
    df = pd.read_csv('Features.csv')
    df = df.drop(['file_name'], axis=1)
    scaler = StandardScaler()
    scaler.fit(np.array(df.iloc[:, :-1], dtype=float))

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    predicted_class = np.argmax(prediction, axis=1)

    if predicted_class[0] == 1:
        return "COVID-19 Positive"
    else:
        return "COVID-19 Negative"

# Streamlit web interface
st.title("COVID-19 Cough Detection Application")

st.write("""
This Application uses a machine learning model to predict if a given audio signal contains COVID-19 cough or not.
""")

uploaded_file = st.file_uploader("Choose an audio file...", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    prediction = predict_covid("temp.wav")
    st.write(f"Prediction: {prediction}")
