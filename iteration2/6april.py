import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sympy import symbols, Eq, solve
import random

# Read data from CSV file using pandas
df = pd.read_csv('6april.csv')  

# Define the secret value and a related public value
secret_value = 5.67
public_value = secret_value * secret_value

# Define symbols for the zero-knowledge proof
x = symbols('x')

# Define the equation for the zero-knowledge proof
eq = Eq(x * x, public_value)

# Solve the equation to find possible secret values
possible_secret_values = solve(eq, x)

# Verify the zero-knowledge proof
verified = False
for val in possible_secret_values:
    if val == secret_value:
        verified = True
        break

if verified:
    print("Zero-knowledge proof verified successfully.")
else:
    print("Zero-knowledge proof verification failed.")

# Once the proof is verified (for demonstration purposes), proceed with model training
if verified:
    # Updated encrypt_data function with random noise
    def encrypt_data(data):
        encrypted_data = []
        for x in data:
            noise = random.uniform(-0.5, 0.5)  # Generate random noise between -0.5 and 0.5
            encrypted_value = int(x[0]) + secret_value + noise  # Add noise to the encrypted value
            encrypted_data.append([encrypted_value])
        return encrypted_data

    # Decrypt the data after model training
    def decrypt_data(data):
        if isinstance(data, float):
            return data - secret_value  # Handle single value
        else:
            return [x - secret_value for x in data]  # Handle list of values

    # Mask the input data using a more secure encryption (with random noise)
    X_encrypted = encrypt_data(df[['size']].values.tolist())

    # Print the encrypted data
    print("Encrypted data:")
    print(X_encrypted)

    y = df['price']

    # Train a linear regression model using the encrypted dataset
    model = LinearRegression()
    model.fit(X_encrypted, y)

    # Print the coefficients of the trained model
    print("Trained model coefficients:", model.coef_[0])

    # Predict prices for new house sizes (example)
    new_sizes_encrypted = encrypt_data([[1800], [2200]])  # Encrypt new data
    predicted_prices_encrypted = model.predict(new_sizes_encrypted)
    predicted_prices = decrypt_data(predicted_prices_encrypted)  # Decrypt predictions

    # Print the encrypted predictions
    print("Predicted prices for new house sizes (encrypted):")
    print(predicted_prices_encrypted)

    # Print the decrypted prices
    print("Predicted prices for new house sizes (decrypted):\n", predicted_prices)

else:
    print("Cannot proceed with model training due to failed zero-knowledge proof.")





