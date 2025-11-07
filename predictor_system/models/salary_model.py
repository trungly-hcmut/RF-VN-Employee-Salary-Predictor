import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

class SalaryPredictionModel:
    """Model for salary prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.is_loaded = False
        
        # Try to load the model
        try:
            self.model = joblib.load('salary_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            self.is_loaded = True
        except Exception:
            # Model not loaded, will use mock predictions
            pass
    
    def predict_salary(self, features):
        """
        Predict salary based on input features
        
        Args:
            features (dict): Dictionary with feature values
            
        Returns:
            float: Predicted salary
        """
        if self.is_loaded:
            # Encode categorical features
            input_dict = {
                'YearsExperience': features['experience'],
                'Education': self.label_encoders['Education'].transform([features['education']])[0],
                'JobRole': self.label_encoders['JobRole'].transform([features['job_role']])[0],
                'Location': self.label_encoders['Location'].transform([features['location']])[0],
                'Age': features['age'],
                'Gender': self.label_encoders['Gender'].transform([features['gender']])[0],
            }
            input_df = pd.DataFrame([input_dict])
            input_scaled = self.scaler.transform(input_df)
            return self.model.predict(input_scaled)[0]
        else:
            # Mock prediction for demo
            return self._mock_predict(features)
    
    def predict_batch(self, df):
        """
        Predict salaries for a batch of inputs
        
        Args:
            df (DataFrame): DataFrame with feature columns
            
        Returns:
            DataFrame: Original DataFrame with predictions added
        """
        df_copy = df.copy()
        
        if self.is_loaded:
            # Encode categorical columns
            for col in ['Education', 'JobRole', 'Location', 'Gender']:
                df_copy[col] = self.label_encoders[col].transform(df_copy[col])
            
            X = df_copy[['YearsExperience', 'Education', 'JobRole', 'Location', 'Age', 'Gender']]
            X_scaled = self.scaler.transform(X)
            df_copy['PredictedSalary'] = self.model.predict(X_scaled).astype(int)
        else:
            # Mock batch predictions
            mock_salaries = []
            for _, row in df_copy.iterrows():
                features = {
                    'experience': row['YearsExperience'],
                    'education': row['Education'],
                    'job_role': row['JobRole'],
                    'location': row['Location'],
                    'age': row['Age'],
                    'gender': row['Gender']
                }
                mock_salaries.append(int(self._mock_predict(features)))
            
            df_copy['PredictedSalary'] = mock_salaries
        
        return df_copy
    
    def train_model(self, df, params, test_size=0.2):
        """
        Train a new salary prediction model
        
        Args:
            df (DataFrame): Training data
            params (dict): Model parameters
            test_size (float): Test split ratio
            
        Returns:
            dict: Model performance metrics
        """
        # Prepare data
        X = df.drop('Salary', axis=1)
        y = df['Salary']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=params['random_state']
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fix max_features parameter
        max_features = params['max_features']
        if max_features == 'auto':
            max_features = 'sqrt'  # 'auto' is deprecated, use 'sqrt' instead
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=max_features,  # Using the fixed parameter
            random_state=params['random_state']
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Save model artifacts
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.is_loaded = True
        
        # Return metrics and feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': feature_importance
        }
    
    def save_model(self):
        """Save model artifacts to disk"""
        if self.is_loaded:
            joblib.dump(self.model, 'salary_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            joblib.dump(self.label_encoders, 'label_encoders.pkl')
            return True
        return False
    
    def _mock_predict(self, features):
        """Generate mock predictions for demo purposes"""
        base_salary = 50000
        exp_factor = features['experience'] * 5000
        
        edu_factor = 0
        if features['education'] == 'Bachelor':
            edu_factor = 20000
        elif features['education'] == 'Master':
            edu_factor = 40000
        elif features['education'] == 'PhD':
            edu_factor = 60000
        
        role_factor = 0
        if features['job_role'] == 'Data Scientist':
            role_factor = 30000
        elif features['job_role'] == 'Software Engineer':
            role_factor = 25000
        elif features['job_role'] == 'Manager':
            role_factor = 35000
        elif features['job_role'] == 'Analyst':
            role_factor = 15000
        elif features['job_role'] == 'HR':
            role_factor = 10000
        
        loc_factor = 0
        if features['location'] == 'New York':
            loc_factor = 30000
        elif features['location'] == 'San Francisco':
            loc_factor = 35000
        elif features['location'] == 'Austin':
            loc_factor = 15000
        elif features['location'] == 'Remote':
            loc_factor = 10000
        elif features['location'] == 'India':
            loc_factor = 5000
        
        age_factor = (features['age'] - 22) * 1000
        
        # Add some random noise
        import random
        noise = random.uniform(-10000, 10000)
        
        # Calculate mock salary
        mock_salary = base_salary + exp_factor + edu_factor + role_factor + loc_factor + age_factor + noise
        return mock_salary