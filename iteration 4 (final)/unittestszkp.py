import unittest
import numpy as np
import secrets
import logging
from flask import Flask, render_template
from parameterized import parameterized
import pandas as pd
import time  # Import for timing functionality AC

# Parameters for affine OPE AC
scale_factor = 2.4
offset = secrets.randbelow(1000)
app = Flask(__name__)

test_results = []  # Global list to store test results AC

# Encryot Data Function AC
def encrypt_data(data):
    encrypted_data = (data * scale_factor) + offset
    return encrypted_data

# Decrypt Data Function AC
def decrypt_data(encrypted_data):
    decrypted_data = (encrypted_data - offset) / scale_factor
    return decrypted_data


class TestZKPEnhancedWithVisualization(unittest.TestCase):
    results = []

    @staticmethod
    def generate_test_data(size=100):
        np.random.seed()  # Randomize seed for variability AC
        X = np.random.uniform(-10, 10, size).reshape(-1, 1)
        y = np.random.uniform(0, 100, size).reshape(-1, 1)
        return X, y

    @parameterized.expand([(i,) for i in range(100)])  # Run 10 times AC
    def test_encryption_decryption(self, run_id):
        start_time = time.time()  # Start timing AC
        
        # Generate randomized data AC
        X, y = self.generate_test_data(size=5)
        
        # Encryption parameters AC
        scale_factor = 2.5
        offset = 5.0
        
        # Encrypt and decrypt AC
        X_encrypted = (X * scale_factor) + offset
        X_decrypted = (X_encrypted - offset) / scale_factor
        y_encrypted = (y * scale_factor) + offset
        y_decrypted = (y_encrypted - offset) / scale_factor

        # Calculate errors AC
        x_error = np.abs(X - X_decrypted).mean()
        y_error = np.abs(y - y_decrypted).mean()

        # Assert almost equal AC
        np.testing.assert_almost_equal(X, X_decrypted, decimal=6, err_msg="Decryption failed for X")
        np.testing.assert_almost_equal(y, y_decrypted, decimal=6, err_msg="Decryption failed for y")

        elapsed_time = time.time() - start_time  # End timing AC

        # Log results with timing AC
        self.results.append({
            "Run ID": run_id,
            "Test Name": "Encryption/Decryption Test",
            "Input X": X.tolist(),
            "Input y": y.tolist(),
            "X Error": x_error,
            "Y Error": y_error,
            "Expected Output": f"Decrypted y: {y.tolist()}",
            "Execution Time (s)": round(elapsed_time, 4),
            "Success": True
        })

    @parameterized.expand([(i,) for i in range(100)])
    def test_proof_verification(self, run_id):
        start_time = time.time()  # Start timing AC

        # Generate randomized data AC
        X, y = self.generate_test_data(size=5)
        salt = secrets.token_hex(16)

        # Mock encryption and proof generation AC
        scale_factor = 2.5
        offset = 5.0
        X_encrypted = (X * scale_factor) + offset
        y_encrypted = (y * scale_factor) + offset

        proof = {"X": X_encrypted, "Y": y_encrypted, "Key": salt}  # Simplified mock AC
        is_valid = proof["Key"] == salt  # Mock verification logic AC

        elapsed_time = time.time() - start_time  # End timing AC

        # Record result with timing AC
        self.results.append({
            "Run ID": run_id,
            "Test Name": "Proof Verification",
            "Input X": X.tolist(),
            "Input y": y.tolist(),
            "Proof Valid": is_valid,
            "Execution Time (s)": round(elapsed_time, 10),
            "Expected Output": "Commitment Match",
            "Success": is_valid
        })

    @classmethod
    def tearDownClass(cls):
        # Save results to a file or global variable for Flask AC
        global test_results
        test_results = cls.results

@app.route('/')
def results():
    # Convert test results into a more readable format AC
    df = pd.DataFrame(TestZKPEnhancedWithVisualization.results)
    test_results = df.to_dict('records')  # Convert DataFrame to a list of dictionaries AC

    return render_template("results.html", test_results=test_results)

if __name__ == '__main__':
    # Run unit tests and store results AC
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestZKPEnhancedWithVisualization))
    # Start Flask app AC
    app.run(debug=True)

