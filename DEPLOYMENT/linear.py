#!/usr/bin/env python
# coding: utf-8

# # 1.Linear

# In[2]:


import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import joblib



# In[3]:


# Load dataset 
df = pd.read_csv('power.csv')

# Display first few rows
print(df.head())


# In[4]:


# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)


# In[5]:


# Get the description of the dataset
description = df.describe()
print(description)


# In[6]:


# Shows a tuple with row and column size
df.shape 


# In[7]:


# Display information about the dataframe, including the index dtype and column dtypes, non-null values, and memory usage
df.info()


# In[8]:


# Returns the first few rows of the DataFrame df
df.head()


# ## Three Interseting Insights from dataset

# ### 1. Relationship Between Atmospheric Pressure (AP) and Relative Humidity (RH)
# Atmospheric pressure (AP) might be inversely related to relative humidity (RH). Higher atmospheric pressure could indicate dry air, resulting in lower humidity levels.

# In[9]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='AP', y='RH')
plt.title('Atmospheric Pressure vs Relative Humidity')
plt.xlabel('Atmospheric Pressure (AP)')
plt.ylabel('Relative Humidity (RH)')
plt.show()


# ### 2. Distribution of Power Consumption (PE)
# The distribution of power consumption (PE) can reveal whether it's skewed or normally distributed. 

# In[10]:


plt.figure(figsize=(8, 6))
sns.histplot(df['PE'], kde=True, bins=30)
plt.title('Distribution of Power Consumption (PE)')
plt.xlabel('Power Consumption (PE)')
plt.ylabel('Frequency')
plt.show()


# ### 3. Power Consumption (PE) vs Temperature (AT)
# Analyzing the relationship between temperature (AT) and power consumption (PE) could reveal patterns, such as higher power consumption during extreme temperatures

# In[11]:


# Rename columns for consistency
df.rename(columns={'AT': 'Temperature', 'PE': 'PowerConsumption'}, inplace=True)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Temperature', y='PowerConsumption')
plt.title('Power Consumption (PE) vs Temperature (AT)')
plt.xlabel('Temperature (AT)')
plt.ylabel('Power Consumption (PE)')
plt.show()


# #### Correlation Matrix to Find Relationships Between Variables
# A correlation matrix can help identify strong linear relationships between the features. 

# In[12]:


plt.figure(figsize=(8, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[13]:


# Rename 'RH' to 'Humidity' for consistency
df.rename(columns={'RH': 'Humidity'}, inplace=True)

# Keep only Temperature, Humidity, and Power Output
df = df[['Temperature', 'Humidity', 'PowerConsumption']]
print(df.head())


# In[14]:


# Define target variable (y)
y = df['PowerConsumption']

# Remove outliers (using robustscaler  method)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[['Temperature', 'Humidity']])
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[15]:


# Define independent variables (X) and target variable (y)
X = df[['Temperature', 'Humidity']]
y = df['PowerConsumption']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# A high correlation between independent variables (Temperature & Humidity) can cause multicollinearity issues.
# we can check Variance Inflation Factor (VIF) and remove highly correlated features.

# In[16]:


# Add constant for VIF calculation
X_with_const = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(len(X.columns))]
print(vif_data)


# A VIF > 5 suggests multicollinearity so there is no multicolinearity

# In[17]:


# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")  # Corresponds to Temperature and Humidity


# In[18]:


# Predict power consumption on test data
y_pred = model.predict(X_test)

# Compare actual vs predicted values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())


# R² Score and Mean Absolute Error (MAE) to check the model's accuracy.

# In[19]:


# Calculate R² Score and MAE
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2}")
print(f"Mean Absolute Error: {mae}")


# In[20]:

# Save the trained model
joblib.dump(model, 'linear_model.pkl') 


# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual', s=15)

# Plot a reference line (y = x), which shows the ideal case where actual = predicted
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Prediction Line')

plt.xlabel('Actual Power Consumption')
plt.ylabel('Predicted Power Consumption')
plt.title('Actual vs Predicted Power Consumption')
plt.legend()
plt.show()
