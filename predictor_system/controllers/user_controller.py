import pandas as pd
from models.salary_model import SalaryPredictionModel

class UserController:
    """Controller for user-specific operations"""
    
    @staticmethod
    def predict_salary(features):
        """
        Predict salary based on input features
        
        Args:
            features (dict): Dictionary with feature values
            
        Returns:
            float: Predicted salary
        """
        model = SalaryPredictionModel()
        return model.predict_salary(features)
    
    @staticmethod
    def predict_batch(file):
        """
        Predict salaries for a batch of inputs from a CSV file
        
        Args:
            file: Uploaded CSV file
            
        Returns:
            dict: Result with success status, message, and DataFrame with predictions
        """
        try:
            df = pd.read_csv(file)
            
            # Check required columns
            expected_cols = ['YearsExperience', 'Education', 'JobRole', 'Location', 'Age', 'Gender']
            if not all(col in df.columns for col in expected_cols):
                return {
                    "success": False,
                    "message": f"CSV must contain columns: {expected_cols}"
                }
            
            # Make predictions
            model = SalaryPredictionModel()
            result_df = model.predict_batch(df)
            
            return {
                "success": True,
                "message": "Predictions completed successfully",
                "data": result_df
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing CSV file: {e}"
            }