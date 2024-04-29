# Import required libraries AC
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sympy import symbols, Eq, solve
import random
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.metrics import mean_squared_error



app = Flask(__name__)

# Load test and train datasets AC
df = pd.read_csv('Salary_dataset_train.csv')
df1 = pd.read_csv('Salary_dataset_test.csv')
def encrypt_data(data, secret_value):
# Encrypt data by adding random noise and secret value.AC
    encrypted_data = []
    for x in data:
        noise = random.uniform(-0.5, 0.5)
        encrypted_value = int(x[0]) + secret_value + noise
        encrypted_data.append([encrypted_value])
    return encrypted_data


@app.route('/')
def index():
# Render the main page AC
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
# Process the data to perform encrypted linear regression and generate plot AC
    def encrypt_data1(data, secret_value):
        encrypted_data = []
        for x in data:
            noise = random.uniform(-0.5, 0.5)  # Generate random noise between -0.5 and 0.5 AC
            try:
# Convert to float and then add noise and secret value AC
                encrypted_value = float(x[0]) + secret_value + noise
                encrypted_data.append([encrypted_value])
            except ValueError:
# Handle non-numeric values AC
                print(f"Warning: Non-numeric value encountered: {x[0]}")
                continue
        print(noise)
        return encrypted_data
    


# Decrypt data function AC
    def decrypt_data(data):
        if isinstance(data, float):
            return data - secret_value  # Handle single value AC
        else:
            return [x - secret_value for x in data]  # Handle list of values AC
        

    secret_value = round(random.SystemRandom().uniform(0, 1000), 2) # Secret value for ZKP AC
    public_value = secret_value ** 2 # Secret Value squared CA
    
# Original training data (x) AC
    original_data = df[['YearsExperience']].values.tolist()

# Zero-knowledge proof verification AC
    x = symbols('x')
    eq = Eq(x * x, public_value)
    possible_secret_values = solve(eq, x)
    verified = secret_value in possible_secret_values
# If the proof is verified: AC
    if verified:
# Encrypt the training data AC
        X_encrypted = encrypt_data(df[['YearsExperience']].values.tolist(), secret_value)
# Encrypt the training data (y) AC
        y_encrypted = []
        for salary in df['Salary'].values.tolist():
            encrypted_salary = encrypt_data([[salary]], secret_value)[0][0]  # Encrypt each salary individually AC
            y_encrypted.append(encrypted_salary)
            
# Train the linear regression model AC
        model = LinearRegression()
        model.fit(X_encrypted, y_encrypted)

# Encrypt the test data (x) AC
        test_data = df1[['YearsExperience']].values.tolist()
        new_years_encrypted = encrypt_data1(test_data, secret_value)

        if new_years_encrypted:
# Predict salaries for new data AC
            predicted_salary_encrypted = model.predict(new_years_encrypted)
            predicted_salary = decrypt_data(predicted_salary_encrypted)  # Decrypt predictions AC

# Decrypt the original salary data (for displaying purposes) AC
            y_decrypted = decrypt_data(y_encrypted)
            
# Calculate Mean Squared Error (MSE) and R-squared (R2 score) AC
            mse = mean_squared_error(y_encrypted, model.predict(X_encrypted))
            r2 = model.score(X_encrypted, y_encrypted)

# Generate the scatterplot AC
            plt.figure(figsize=(10, 6))
# Update plot title with MSE and R2 score AC
            plt.title(f'Linear Regression on Encrypted Data\nMSE: {mse:.2f}, R2: {r2:.2f}')
            plt.scatter([x[0] for x in X_encrypted], y_encrypted, color='blue', label='Encrypted Data')
            plt.plot([x[0] for x in X_encrypted], model.predict(X_encrypted), color='red', linewidth=2, label='Regression Line')
            plt.xlabel('Encrypted years of experience')
            plt.ylabel('Encrypted salary')
            plt.legend()
        
# Convert plot to PNG image to send to HTML AC
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
# Sends variables to be displayed on HTML AC
            return render_template("index.html", mse=mse, r2=r2, predicted_sal_e=predicted_salary_encrypted, predicted_sal=predicted_salary, original_data=original_data, encrypted_data=X_encrypted, plot_url=plot_url)

            
        else:
            error_message = "Error: Empty or invalid test data."
            return render_template("index.html", error=error_message)
    else:
        error_message = "Zero-knowledge proof verification failed."
        return render_template("index.html", error=error_message)
    
if __name__ == '__main__':
    app.run(debug=True)
