import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Sample data creation (replace this with loading your dataset)
data = pd.DataFrame({
    'model': ['BMW 320i', 'BMW X5', 'BMW 330i', 'BMW X3'],
    'year': [2015, 2018, 2017, 2016],
    'mileage': [50000, 30000, 45000, 60000],
    'price': [20000, 35000, 25000, 22000]
})

# Define features and target
X = data.drop('price', axis=1)
y = data['price']

# Define preprocessing steps
numeric_features = ['year', 'mileage']
categorical_features = ['model']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model
joblib.dump(model, 'bmw_price_predictor.joblib')

# Load the model for prediction
loaded_model = joblib.load('bmw_price_predictor.joblib')

# Example features for prediction
example_features = pd.DataFrame({
    'model': ['BMW 320i', 'BMW X5'],
    'year': [2016, 2019],
    'mileage': [55000, 25000]
})

# Predict prices
price_predictions = loaded_model.predict(example_features)
print("Predicted Prices:", price_predictions)
