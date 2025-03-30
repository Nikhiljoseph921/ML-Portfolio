import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import requests
import io
from sklearn.preprocessing import RobustScaler

# GitHub raw URLs for model and scaler
MODEL_URL = "https://raw.githubusercontent.com/Nikhiljoseph921/ML-Portfolio/main/DEPLOYMENT/linear.pkl"
SCALER_URL = "https://raw.githubusercontent.com/Nikhiljoseph921/ML-Portfolio/main/DEPLOYMENT/scaler.pkl"

# Function to load pickle file from GitHub URL
def load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
    return pickle.load(io.BytesIO(response.content))

# Load model and scaler
try:
    model = load_pickle_from_url(MODEL_URL)
    scaler = load_pickle_from_url(SCALER_URL)
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")

# Custom CSS for dark mode
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    .input-text {
        font-size: 18px;
        font-weight: bold;
        color: white;
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        color: #ffcc00;
        text-shadow: 0px 0px 10px rgba(255,204,0,0.8);
    }
    .stNumberInput>div>div>input {
        background-color: #262626;
        color: white;
        border-radius: 8px;
        border: 1px solid #444;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #ff4d4d;
        color: white;
        font-size: 18px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #00ccff;'>⚡ Power Consumption Prediction  </h1>", unsafe_allow_html=True)

# User Input: Temperature and Humidity
st.markdown("<p class='input-text'>Enter Temperature (AT) (°C)</p>", unsafe_allow_html=True)
temperature = st.number_input('', min_value=0.0, max_value=50.0, step=0.1)

st.markdown("<p class='input-text'>Enter Humidity (RH) (%)</p>", unsafe_allow_html=True)
humidity = st.number_input('', min_value=0.0, max_value=100.0, step=0.1)

# Prepare the input data
input_data = pd.DataFrame({'Temperature': [temperature], 'Humidity': [humidity]})
scaled_data = scaler.transform(input_data)

# Prediction Function
def plot_prediction_chart(prediction):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Adjusting y-axis limits dynamically
    y_max = max(prediction * 1.2, 10)  # Ensures space above the bar
    
    bars = ax.bar(['Predicted Power Consumption'], [prediction], color='royalblue', alpha=0.6, edgecolor='black')

    # Display the predicted value on top of the bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{prediction:.2f} kW", 
                ha='center', fontsize=14, fontweight='bold', color='yellow')

    ax.set_ylim(0, y_max)  # Set the y-axis limit
    ax.set_ylabel('Power Consumption (kW)', fontsize=12, color='white')
    ax.set_title('Power Consumption Prediction', fontsize=14, color='white')
    
    # Enhance aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor("#262626")  # Dark background for better visibility

    st.pyplot(fig)

# Make Prediction
if st.button('⚡ Predict Power Consumption ⚡'):
    prediction = model.predict(scaled_data)
    
    st.markdown(f"<p class='prediction-text'>⚡ Predicted Power Consumption: {prediction[0]:.2f} kW ⚡</p>", unsafe_allow_html=True)

    # Show Graph
    plot_prediction_chart(prediction[0])