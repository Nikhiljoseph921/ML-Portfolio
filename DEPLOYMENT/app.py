import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Streamlit UI
st.title("  AI-Driven Power Forecasting with Linear Regression")

# Load dataset automatically
default_file = "power.csv"
df = pd.read_csv(default_file)

st.write("📊 Dataset:")
st.dataframe(df.head())

# Preprocessing: Rename columns for consistency
df.rename(columns={'AT': 'Temperature', 'PE': 'PowerConsumption', 'RH': 'Humidity'}, inplace=True)

# Display summary statistics
st.write("📌 **Dataset Summary**")
st.dataframe(df.describe())
st.write("\n\n\n\n")

# Check for missing values
st.write("🔍 **Missing Values**")
st.dataframe(df.isnull().sum())

# Correlation Heatmap
st.write("🔗 **Correlation Matrix**")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
st.pyplot(plt)
st.write("\n\n\n\n")

# Scatter plot: Temperature vs Power Consumption
st.markdown("## 🌡️ Power Consumption vs Temperature")
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Temperature', y='PowerConsumption')
plt.xlabel("Temperature (AT)")
plt.ylabel("Power Consumption (PE)")
st.pyplot(plt)
st.write("\n\n\n\n")

# Scatter plot: Humidity vs Power Consumption
st.markdown("## 💦 Power Consumption vs Humidity")
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Humidity', y='PowerConsumption')
st.pyplot(plt)
st.write("\n\n\n\n")

# Define target variable
y = df["PowerConsumption"]
X = df[['Temperature', 'Humidity']]

# Check for multicollinearity using Variance Inflation Factor (VIF)
X_with_const = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(len(X.columns))]
st.write("📊 **Variance Inflation Factor (VIF)**")
st.dataframe(vif_data)
st.write("\n\n\n\n")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Show model coefficients
st.write(f"📏 **Model Intercept:** {model.intercept_}")
st.write(f"📊 **Model Coefficients:** {model.coef_}")

# Predict on test data
y_pred = model.predict(X_test)

# Model Performance Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write(f"🎯 **R² Score:** {r2:.4f}")
st.write(f"📉 **Mean Absolute Error (MAE):** {mae:.4f}")
st.write("\n\n\n\n")

# Scatter plot: Actual vs Predicted
st.markdown("## 🔬 Actual vs Predicted Power Consumption")
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, color='blue', s=15, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Power Consumption")
plt.ylabel("Predicted Power Consumption")
plt.legend()
st.pyplot(plt)
st.write("\n\n\n\n")

# Save trained model
joblib.dump(model, "linear_model.pkl")

# Load model for deployment
model = joblib.load("linear_model.pkl")

# Make predictions using full dataset
predictions = model.predict(X)
df["Prediction"] = predictions

st.markdown("## ✅ Predictions")
st.dataframe(df)
st.write("\n\n\n\n")

# Download predictions
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
st.write("\n\n\n\n")

# User Input for Prediction
st.markdown("## 📝 Make a Prediction")
temp_input = st.number_input("Enter Temperature:", min_value=0.0, max_value=100.0, value=25.0)
humidity_input = st.number_input("Enter Humidity:", min_value=0.0, max_value=100.0, value=50.0)
st.write("\n\n\n\n")

if st.button("# Predict Power Consumption"):
    user_input = pd.DataFrame({'Temperature': [temp_input], 'Humidity': [humidity_input]})
    predicted_power = model.predict(user_input)[0]
    st.write(f"⚡ **Predicted Power Consumption:** {predicted_power:.2f}")
    
    # Visualization
    st.markdown("## 📉 Visualization of Prediction")
    plt.figure(figsize=(8, 6))
    plt.bar(["Predicted Power Consumption"], [predicted_power], color='green')
    plt.ylabel("Power Consumption")
    plt.text(0, predicted_power, f"{predicted_power:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='white', bbox=dict(facecolor='black', alpha=0.5))
    st.pyplot(plt)
    st.write("\n\n\n\n")
    
    

