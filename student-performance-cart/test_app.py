"""
test_app.py
===========
Automated test suite verifying the Flask web application, CART ML inference,
data validation, and all web routes.
"""

import unittest
import os
import sys
import json
import joblib
import numpy as np

# Ensure student-performance-cart is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, get_model_and_metrics


class TestCartStudentApp(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_model_and_metrics_files_exist(self):
        """Verify model.pkl, metrics.json, dataset.csv, and static plots exist."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'model.pkl')))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'metrics.json')))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'dataset.csv')))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'static', 'images', 'decision_tree.png')))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'static', 'images', 'confusion_matrix.png')))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'static', 'images', 'feature_importance.png')))

    def test_model_inference_logic(self):
        """Test direct inference using Joblib loaded model."""
        import pandas as pd
        model, metrics = get_model_and_metrics()
        cols = ['attendance', 'study_hours', 'internal_marks', 'assignment_score']
        
        # Test High Achiever (Pass expected)
        sample_pass = pd.DataFrame([[90.0, 6.0, 85.0, 90.0]], columns=cols)
        pred_pass = model.predict(sample_pass)[0]
        self.assertEqual(pred_pass, 1, "High scoring student should be predicted PASS (1)")

        # Test At-Risk Student (Fail expected)
        sample_fail = pd.DataFrame([[45.0, 0.8, 28.0, 35.0]], columns=cols)
        pred_fail = model.predict(sample_fail)[0]
        self.assertEqual(pred_fail, 0, "Low scoring student should be predicted FAIL (0)")

    def test_home_route(self):
        """Test GET / returns 200 and has CART title."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("Student Performance Prediction", content)
        self.assertIn("Gini Impurity", content)
        self.assertIn("Start Prediction", content)

    def test_predict_get_route(self):
        """Test GET /predict renders form properly."""
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn('name="attendance"', content)
        self.assertIn('name="study_hours"', content)
        self.assertIn('name="internal_marks"', content)
        self.assertIn('name="assignment_score"', content)

    def test_predict_post_pass(self):
        """Test POST /predict with high student attributes results in PASS."""
        payload = {
            'attendance': '88.5',
            'study_hours': '5.0',
            'internal_marks': '82.0',
            'assignment_score': '85.0'
        }
        response = self.client.post('/predict', data=payload)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("PASS", content)
        self.assertIn("Confidence", content)
        self.assertIn("Predict Another Student", content)

    def test_predict_post_fail(self):
        """Test POST /predict with low student attributes results in FAIL."""
        payload = {
            'attendance': '45.0',
            'study_hours': '1.0',
            'internal_marks': '28.0',
            'assignment_score': '35.0'
        }
        response = self.client.post('/predict', data=payload)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("FAIL", content)
        self.assertIn("Confidence", content)
        self.assertIn("Predict Another Student", content)

    def test_predict_post_validation_error(self):
        """Test POST /predict rejects invalid negative or out-of-range values."""
        payload = {
            'attendance': '150.0',  # Out of range (>100)
            'study_hours': '25.0',   # Out of range (>12)
            'internal_marks': '75.0',
            'assignment_score': '80.0'
        }
        response = self.client.post('/predict', data=payload)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("Attendance must be between 0% and 100%", content)

    def test_result_get_redirect(self):
        """Test GET /result redirects to /predict."""
        response = self.client.get('/result')
        self.assertEqual(response.status_code, 302)

    def test_insights_route(self):
        """Test GET /insights displays tree, matrix, and metrics."""
        response = self.client.get('/insights')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("decision_tree.png", content)
        self.assertIn("confusion_matrix.png", content)
        self.assertIn("feature_importance.png", content)
        self.assertIn("Testing Accuracy", content)
        self.assertIn("Gini Impurity", content)


if __name__ == '__main__':
    unittest.main()
