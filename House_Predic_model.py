# Importing Dependencies
pip install numpy pandas joblib gradio scikit-learn xgboost matplotlib seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving & loading models
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_california_housing

# Load Dataset (Replacing Deprecated load_boston)
house_price_dataset = fetch_california_housing()
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

# Adding target column
house_price_dataframe['price'] = house_price_dataset.target * 100  # Scaling prices for better readability

# Checking for missing values
print("Missing Values in Dataset:\n", house_price_dataframe.isnull().sum())

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(house_price_dataframe.drop(['price'], axis=1))
Y = house_price_dataframe['price']

# Splitting the data into Training and Test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Optimized XGBoost Model (Hyperparameter Tuning)
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=5)
xgb_model.fit(X_train, Y_train)

# Save the trained model
joblib.dump(xgb_model, "house_price_model.pkl")

# Evaluation on Training Data
training_data_prediction = xgb_model.predict(X_train)
train_r2 = metrics.r2_score(Y_train, training_data_prediction)
train_mae = metrics.mean_absolute_error(Y_train, training_data_prediction)

print(f"🔹 Training R² Score: {train_r2:.4f}")
print(f"🔹 Training Mean Absolute Error: {train_mae:.4f}")

# Evaluation on Test Data
test_data_prediction = xgb_model.predict(X_test)
test_r2 = metrics.r2_score(Y_test, test_data_prediction)
test_mae = metrics.mean_absolute_error(Y_test, test_data_prediction)

print(f"🔹 Test R² Score: {test_r2:.4f}")
print(f"🔹 Test Mean Absolute Error: {test_mae:.4f}")

# Visualizing Actual vs Predicted Prices
plt.figure(figsize=(7,5))
plt.scatter(Y_test, test_data_prediction, alpha=0.6, color='blue')
plt.xlabel("Actual Prices ($1000s)")
plt.ylabel("Predicted Prices ($1000s)")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Function to Predict House Prices
def predict_house_price(features):
    try:
        # Convert input features into numpy array & scale it
        features_scaled = scaler.transform([features])

        # Load trained model
        model = joblib.load("house_price_model.pkl")

        # Make prediction
        predicted_price = model.predict(features_scaled)[0]
        return f"🏡 Predicted House Price: ${predicted_price * 1000:.2f}"

    except Exception as e:
        return f"⚠️ Error: {str(e)}. Please enter valid numerical values."

# Test the Prediction Function
sample_input = X_test[0]
print(predict_house_price(sample_input))
