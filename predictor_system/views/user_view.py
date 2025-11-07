import streamlit as st
import time
from controllers.auth_controller import AuthController
from controllers.user_controller import UserController
from config.settings import EDUCATION_LEVELS, JOB_ROLES, LOCATIONS, GENDERS

def display_salary_prediction():
    """Display the salary prediction page"""
    # Display user info in sidebar
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    st.sidebar.info(f"Role: {st.session_state.role}")
    if hasattr(st.session_state, 'user_type') and st.session_state.user_type:
        st.sidebar.info(f"Account type: {st.session_state.user_type}")
    
    # Navigation buttons in sidebar
    if st.sidebar.button("Back to Dashboard"):
        st.session_state.page = 'dashboard'
        st.rerun()
        
    if st.sidebar.button("Logout"):
        AuthController.logout()
        st.rerun()

    st.title('Employee Salary Prediction')
    st.write('Predict employee salary using manual input or batch CSV upload.')

    try:
        # --- Manual Input ---
        st.header('Manual Prediction')
        with st.form('manual_form'):
            experience = st.slider('Years of Experience', 0, 20, 5)
            education = st.selectbox('Education Level', EDUCATION_LEVELS)
            job_role = st.selectbox('Job Role', JOB_ROLES)
            location = st.selectbox('Location', LOCATIONS)
            age = st.slider('Age', 22, 60, 30)
            gender = st.selectbox('Gender', GENDERS)
            submitted = st.form_submit_button('Predict Salary')

            if submitted:
                # Create features dictionary
                features = {
                    'experience': experience,
                    'education': education,
                    'job_role': job_role,
                    'location': location,
                    'age': age,
                    'gender': gender
                }
                
                # Get prediction from controller
                salary_pred = UserController.predict_salary(features)
                st.success(f'Predicted Salary: ₹{int(salary_pred):,.0f}')

        # --- Batch Prediction (Only for Recruiters) ---
        if hasattr(st.session_state, 'user_type') and st.session_state.user_type == "Recruiter":
            st.header('Batch Prediction (CSV Upload)')
            st.write('Upload a CSV file with columns: YearsExperience, Education, JobRole, Location, Age, Gender')
            file = st.file_uploader('Upload CSV', type=['csv'])
            if file is not None:
                # Process batch prediction
                result = UserController.predict_batch(file)
                if result["success"]:
                    st.dataframe(result["data"])
                    # Download link
                    csv = result["data"].to_csv(index=False).encode('utf-8')
                    st.download_button('Download Predictions as CSV', csv, 'salary_predictions.csv', 'text/csv')
                else:
                    st.error(result["message"])
                    st.info("Please ensure your CSV file is properly formatted.")
        elif hasattr(st.session_state, 'user_type') and st.session_state.user_type != "Recruiter":
            st.info("Batch prediction is only available for Recruiter accounts.")

        # --- Model Info ---
        st.header('Model & Dataset Info')
        st.write('Trained on 500 synthetic employee records. Model: Random Forest Regressor.')
        st.write('Features: YearsExperience, Education, JobRole, Location, Age, Gender')
        st.write('All salaries are shown in Indian Rupees (₹).')
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("The application is running in a limited mode. Some features may not be available.")