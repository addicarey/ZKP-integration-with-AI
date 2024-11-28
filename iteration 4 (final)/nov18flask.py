# Import required libraries AC
import pandas as pd
import numpy as np
import random
import hashlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging
import hmac
import secrets
from flask import Flask, render_template, request, jsonify
import io
import base64
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error

app = Flask(__name__)



# Initialize logging with formatted output AC
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Parameters for affine OPE AC
scale_factor = 2.4  # Adjust to scale values AC
offset = secrets.randbelow(1000)  # Randomly generated offset for encryption AC

# Load datasets (test and train) AC
df = pd.read_csv('Salary_dataset_train.csv')
df1 = pd.read_csv('Salary_dataset_test.csv')


@app.route('/')
def index():
# Render the main page AC
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    

    def standardize(data):
        # Standardize the data AC
        return (data - np.mean(data)) / np.std(data)

    def encrypt_data(data):
        # Encrypt by using affine OPE style encryption AC
        encrypted_data = (data * scale_factor) + offset
        return encrypted_data

    def decrypt_data(encrypted_data):
        # Decrypt by reversing the scaling and offset AC
        decrypted_data = (encrypted_data - offset) / scale_factor
        return decrypted_data

    def generate_commitment(data):
        # Generate a cryptographic commitment to the data AC
        data_bytes = data.tobytes()
        commitment = hashlib.sha256(data_bytes).hexdigest()
        return commitment

    # Generate a secret key for HMAC commitments AC
    secret_key = secrets.token_bytes(32)  # 256-bit key for strong security AC

    def generate_hmac_commitment(data, secret_key, salt):
        # Generate a cryptographic HMAC commitment to the data with a secret key and salt AC
        data_bytes = data.tobytes()
        salt_bytes = salt.encode('utf-8')  # Convert salt to bytes AC
        hmac_object = hmac.new(secret_key, data_bytes + salt_bytes, hashlib.sha256)  # Combine data and salt AC
        commitment = hmac_object.hexdigest()
        return commitment

    def generate_proof(X, y, X_encrypted, y_encrypted, secret_key, salt):
        # Generate a ZKP with HMAC commitments to enhance verification security AC
        commitment_X = generate_hmac_commitment(X, secret_key, salt)
        commitment_y = generate_hmac_commitment(y, secret_key, salt)
        
        # Log the HMAC commitments AC
        logging.info(f'\nCommitments Generated (HMAC):')
        logging.info(f'  - HMAC Commitment for X: {commitment_X}')
        logging.info(f'  - HMAC Commitment for y: {commitment_y}')
        
        proof = {
            'commitment_X': commitment_X,
            'commitment_y': commitment_y
        }
        
        logging.info(f'\nProof Details:')
        logging.info(f'\n  - Encrypted X Sample: {X_encrypted[:3]}...')
        logging.info(f'\n  - Encrypted y Sample: {y_encrypted[:3]}...')
        
        return proof

    def verify_proof(proof, X, y, secret_key, salt):
        # Verify the HMAC proof AC
        commitment_X_check = generate_hmac_commitment(X, secret_key, salt)
        commitment_y_check = generate_hmac_commitment(y, secret_key, salt)
        
        # Validate commitments AC
        is_valid = (commitment_X_check == proof['commitment_X']) and \
                (commitment_y_check == proof['commitment_y'])
        
        if is_valid:
            logging.info('\n[Success] HMAC-based proof verification succeeded.')
        else:
            logging.error('\n[Error] HMAC-based proof verification failed.')
        
        return is_valid


    logging.info(f'\n[ZKP Setup]')


    # Original data standardization AC
    X_train = standardize(df[['YearsExperience']].values)
    y_train = standardize(df[['Salary']].values)

    # Salt value for HMAC AC
    salt = secrets.token_hex(16)  # Generate a 16-byte salt AC

    logging.info(f'\n[HMAC Salt]')
    logging.info(f'  - Salt: {salt}')

    # Encrypt the original data AC
    X_encrypted = encrypt_data(X_train)
    y_encrypted = encrypt_data(y_train).flatten()

    # Generate and verify proof using the HMAC commitment AC
    proof = generate_proof(X_train, y_train, X_encrypted, y_encrypted, secret_key, salt)

    # Verify the proof before training the model AC
    if verify_proof(proof, X_train, y_train, secret_key, salt):
        # Proceed with training if proof verified AC
        logging.info('\n[Model Training]')
        
        # Train a LR model on the encrypted data AC
        model_encrypted = LinearRegression()
        model_encrypted.fit(X_encrypted, y_encrypted)
        logging.info(f'  - Trained model on encrypted data.')

        # Train a LR on the original (decrypted) data AC
        model_original = LinearRegression()
        model_original.fit(X_train, y_train)
        logging.info(f'  - Trained model on original data.')

        # Encrypt the test data AC
        X_test = standardize(df1[['YearsExperience']].values)
        X_test_encrypted = encrypt_data(X_test)

        # Predict values for the test data using the model trained on encrypted data AC
        y_pred_encrypted = model_encrypted.predict(X_test_encrypted)
        y_pred_decrypted = decrypt_data(y_pred_encrypted)

        # Predict values for the test data using the model trained on original data AC
        y_pred_original = model_original.predict(X_test)

        # Evaluate model performance on the training data (encrypted) AC
        mse_encrypted = mean_squared_error(y_encrypted, model_encrypted.predict(X_encrypted))
        r2_encrypted = model_encrypted.score(X_encrypted, y_encrypted)
    
        # Evaluate model performance on the training data (original) AC
        mse_original = mean_squared_error(y_train, model_original.predict(X_train))
        r2_original = model_original.score(X_train, y_train)

        # Plot the results for the model trained on encrypted data AC
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f'Linear Regression on Encrypted Data\nMSE: {mse_encrypted:.2f}, R2: {r2_encrypted:.2f}')
        plt.scatter(X_encrypted, y_encrypted, color='blue', label='Encrypted Data')
        plt.plot(X_encrypted, model_encrypted.predict(X_encrypted), color='red', linewidth=2, label='Regression Line')
        plt.xlabel('Encrypted Experience (in months)')
        plt.ylabel('Encrypted Salary (in thousands)')
        plt.legend()

        # Plot the results for the model trained on original data AC
        plt.subplot(1, 2, 2)
        plt.title(f'Linear Regression on Original Data\nMSE: {mse_original:.2f}, R2: {r2_original:.2f}')
        plt.scatter(X_train, y_train, color='green', label='Original Data')
        plt.plot(X_train, model_original.predict(X_train), color='red', linewidth=2, label='Regression Line')
        plt.xlabel('Experience (in months)')
        plt.ylabel('Salary (in thousands)')
        plt.legend()

        # Predictions comparison for decrypted vs. original model predictions AC
        logging.info('\n[Comparison of Model Predictions]')
        for i in range(5):  # Display a few samples
            decrypted_prediction = y_pred_decrypted[i].item() if np.isscalar(y_pred_decrypted[i]) else y_pred_decrypted[i][0]
            original_prediction = y_pred_original[i].item() if np.isscalar(y_pred_original[i]) else y_pred_original[i][0]
            logging.info(f'Encrypted Model Prediction (Decrypted): {decrypted_prediction:.2f}, Original Model Prediction: {original_prediction:.2f}')

        # Calculate additional metrics for encrypted data model AC
        mae_encrypted = mean_absolute_error(y_encrypted, model_encrypted.predict(X_encrypted))
        mape_encrypted = mean_absolute_percentage_error(y_encrypted, model_encrypted.predict(X_encrypted))
        rmse_encrypted = np.sqrt(mean_squared_error(y_encrypted, model_encrypted.predict(X_encrypted)))

        # Calculate additional metrics for original data model AC
        mae_original = mean_absolute_error(y_train, model_original.predict(X_train))
        mape_original = mean_absolute_percentage_error(y_train, model_original.predict(X_train))
        rmse_original = np.sqrt(mean_squared_error(y_train, model_original.predict(X_train)))

        # Display metrics AC
        logging.info(f'\n[Model Evaluation Metrics - Encrypted Data]')
        logging.info(f'  - MAE: {mae_encrypted:.2f}')
        logging.info(f'  - MAPE: {mape_encrypted:.2f}%')
        logging.info(f'  - RMSE: {rmse_encrypted:.2f}')
        logging.info(f'  - MSE: {mse_encrypted:.2f}')
        logging.info(f'  - R2 Score: {r2_encrypted:.2f}')

        logging.info(f'\n[Model Evaluation Metrics - Original Data]')
        logging.info(f'  - MAE: {mae_original:.2f}')
        logging.info(f'  - MAPE: {mape_original:.2f}%')
        logging.info(f'  - RMSE: {rmse_original:.2f}')
        logging.info(f'  - MSE: {mse_original:.2f}')
        logging.info(f'  - R2 Score: {r2_original:.2f}')

         # Model Metrics Plot AC
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics = ['MSE', 'MAE', 'RMSE', 'R²']
        original_metrics = [mse_original, mae_original, rmse_original, r2_original]
        encrypted_metrics = [mse_encrypted, mae_encrypted, rmse_encrypted, r2_encrypted]

        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, original_metrics, width, label='Original')
        ax.bar(x + width/2, encrypted_metrics, width, label='Encrypted')

        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        plt.tight_layout()

        # Save to display on HTML AC
        metrics_img = io.BytesIO()
        plt.savefig(metrics_img, format='png')
        metrics_img.seek(0)
        metrics_plot_url = base64.b64encode(metrics_img.getvalue()).decode()
        plt.close()

        # Show the plots
        plt.tight_layout()
        #plt.show()
        
        # Convert plot to PNG image to send to HTML AC
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        # Sends variables to be displayed on HTML AC
        # Convert data to lists for easier passing to Jinja2 template
        test_data_original = X_test.flatten().tolist()
        test_data_encrypted = X_test_encrypted.flatten().tolist()
        predicted_salaries_encrypted = y_pred_encrypted.tolist()
        predicted_salaries_decrypted = y_pred_decrypted.tolist()
    
        # Pass data and metrics to the template
        return render_template(
            "index.html",
            plot_url=plot_url,
            mae_encrypted=mae_encrypted,
            mape_encrypted=mape_encrypted,
            rmse_encrypted=rmse_encrypted,
            mse_encrypted=mse_encrypted,
            r2_encrypted=r2_encrypted,
            mae_original=mae_original,
            mape_original=mape_original,
            rmse_original=rmse_original,
            mse_original=mse_original,
            r2_original=r2_original,
            original_data=test_data_original,
            encrypted_data=test_data_encrypted,
            predicted_sal_e=predicted_salaries_encrypted,
            predicted_sal=predicted_salaries_decrypted,
            scale_factor=scale_factor,
            offset=offset,
            metrics_plot_url=metrics_plot_url,
            commitment_X=proof["commitment_X"],  # Pass commitment for X
            commitment_y=proof["commitment_y"],  # Pass commitment for y
            salt=salt,  # Pass the salt used
        )



    else:
        # If proof failed, model training won't occur AC
        logging.error('Proof verification failed: The encrypted data does not match the original data.')
        error_message = "ZKP Verification Failed"
        return render_template("index.html", error=error_message)
  
if __name__ == '__main__':
    app.run(debug=True)
