import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Load your dataset (replace with your actual dataset path)
df = pd.read_csv('power.csv')

# Prepare the data
df.rename(columns={'AT': 'Temperature', 'RH': 'Humidity', 'PE': 'PowerConsumption'}, inplace=True)
X = df[['Temperature', 'Humidity']]
y = df['PowerConsumption']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler using pickle
with open('linear.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved as linear.pkl and scaler.pkl.")

