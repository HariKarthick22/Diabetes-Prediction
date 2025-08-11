import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# ✅ 1. Load Pima Indians Diabetes Dataset (with NO header)
url = '/Users/harikarthick/Desktop/Health /diabetes2.csv'
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=cols, header=None)  # ✅ Fix: Prevent string header being read as data

# ✅ 2. Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# ✅ 3. Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ 5. Build a simple MLP model (no CNN!)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(8,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output: 0 or 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ 6. Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# ✅ 7. Save model and scaler
model.save("diabetes_mlp_model.h5")
joblib.dump(scaler, "diabetes_scaler.pkl")

# ✅ 8. Evaluate performance (optional)
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Model saved! Test Accuracy: {acc*100:.2f}%")
