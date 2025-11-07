import streamlit as st
import time
import pandas as pd
from controllers.auth_controller import AuthController
from controllers.admin_controller import AdminController

def display_admin_sidebar():
    """Display the admin navigation sidebar"""
    st.sidebar.success(f"Logged in as: {st.session_state.username} (Admin)")
    
    st.sidebar.title("Admin Navigation")
    
    if st.sidebar.button("Dashboard"):
        st.session_state.page = 'dashboard'
        st.rerun()
        
    if st.sidebar.button("User Management"):
        st.session_state.page = 'admin_user_management'
        st.rerun()
        
    if st.sidebar.button("Data Visualization"):
        st.session_state.page = 'admin_data_visualization'
        st.rerun()
        
    if st.sidebar.button("Model Management"):
        st.session_state.page = 'admin_model_management'
        st.rerun()
        
    if st.sidebar.button("Salary Prediction Tool"):
        st.session_state.page = 'salary_prediction'
        st.rerun()
        
    if st.sidebar.button("Logout"):
        AuthController.logout()
        st.rerun()

def display_dashboard():
    """Display the admin or user dashboard"""
    st.header(f"Welcome, {st.session_state.username}!")
    
    if st.session_state.role == "Admin":
        st.subheader("Admin Dashboard")
        st.write("Welcome to the admin dashboard. Select a management area below:")
        
        # Create three columns for admin options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("User Management")
            st.write("Manage user accounts and permissions")
            if st.button("User Management", key="user_mgmt_btn", use_container_width=True):
                st.session_state.page = 'admin_user_management'
                st.rerun()
        
        with col2:
            st.info("Data Visualization")
            st.write("View charts and analytics on employee data")
            if st.button("Data Visualization", key="data_viz_btn", use_container_width=True):
                st.session_state.page = 'admin_data_visualization'
                st.rerun()
        
        with col3:
            st.info("Model Management")
            st.write("Adjust model parameters and retrain")
            if st.button("Model Management", key="model_mgmt_btn", use_container_width=True):
                st.session_state.page = 'admin_model_management'
                st.rerun()
        
        st.markdown("---")
        
        # Quick access to salary prediction tool
        if st.button("Access Salary Prediction Tool", use_container_width=True):
            st.session_state.page = 'salary_prediction'
            st.rerun()
    else:
        # Check if user_type exists and is not None
        user_type = st.session_state.get('user_type', 'User')
        st.subheader(f"{user_type} Dashboard")
        st.write(f"This is the {user_type.lower() if user_type else 'user'} dashboard.")
        
        # Button for users to access the salary prediction app
        if st.button("Access Salary Prediction Tool"):
            st.session_state.page = 'salary_prediction'
            st.rerun()
    
    if st.button("Logout"):
        AuthController.logout()
        st.rerun()

def display_admin_user_management():
    """Display the admin user management page"""
    # Admin navigation sidebar
    display_admin_sidebar()
    
    st.title("User Management")
    st.write("Manage user accounts and permissions")
    
    # Display current users
    st.subheader("Current Users")
    
    # Get user data from controller
    user_data = AdminController.get_all_users()
    user_df = pd.DataFrame(user_data)
    st.dataframe(user_df, use_container_width=True)
    
    # Add new user section
    st.subheader("Add New User")
    
    with st.form("add_user_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", ["Admin", "User"])
        
        # Only show user type if role is User
        new_user_type = None
        if new_role == "User":
            new_user_type = st.selectbox("User Type", ["Student", "Software Engineer", "Recruiter"])
        
        submit_button = st.form_submit_button("Add User")
        
        if submit_button:
            result = AdminController.add_user(new_username, new_password, new_role, new_user_type)
            if result["success"]:
                st.success(result["message"])
                st.code(f"{new_username}: {result['user_data']}")
            else:
                st.error(result["message"])

def display_admin_data_visualization():
    """Display the admin data visualization page"""
    # Admin navigation sidebar
    display_admin_sidebar()
    
    st.title("Data Visualization")
    st.write("Explore employee salary data through visualizations")
    
    try:
        # Get data from controller
        df = AdminController.get_employee_data()
        
        # Display the data
        st.subheader("Employee Salary Data")
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Total records: {len(df)}")
        
        # Data summary
        st.subheader("Data Summary")
        st.write(df.describe())
        
        # Visualization options
        st.subheader("Visualizations")
        
        viz_type = st.selectbox(
            "Select Visualization", 
            ["Salary Distribution", "Salary by Experience", "Salary by Education", 
             "Salary by Job Role", "Salary by Location", "Correlation Heatmap"]
        )
        
        # Create and display visualization
        fig = AdminController.create_visualization(df, viz_type)
        st.pyplot(fig)
        
        # Custom visualization
        st.subheader("Custom Visualization")
        st.write("Create your own visualization by selecting variables to plot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("X-axis", df.columns)
        
        with col2:
            y_var = st.selectbox("Y-axis", df.columns, index=1)
        
        plot_type = st.selectbox("Plot Type", ["Scatter", "Bar", "Line", "Box"])
        
        # Create and display custom visualization
        fig = AdminController.create_visualization(df, "Custom Plot", x_var, y_var, plot_type)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading or visualizing data: {e}")
        st.info("Please ensure the employee_salary_data.csv file exists and is properly formatted.")

def display_admin_model_management():
    """Display the admin model management page"""
    # Admin navigation sidebar
    display_admin_sidebar()
    
    st.title("Model Management")
    st.write("Adjust model parameters and retrain the salary prediction model")
    
    # Get data from controller
    df = AdminController.get_employee_data()
        
    # Display data sample
    st.subheader("Training Data Sample")
    st.dataframe(df.head(), use_container_width=True)
    
    # Model parameters section
    st.subheader("Random Forest Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 50, 500, 100, 10)
        max_depth = st.slider("Max Depth", 5, 50, 10, 1)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2, 1)
    
    with col2:
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1, 1)
        # Updated options for max_features
        max_features = st.selectbox("Max Features", ["sqrt", "log2", None])
        random_state = st.number_input("Random State", 0, 100, 42, 1)
    
    # Train-test split parameters
    st.subheader("Train-Test Split")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    
    # Button to train model
    if st.button("Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            # Prepare model parameters
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,
                'random_state': random_state,
                'test_size': test_size
            }
            
            # Train model and get metrics
            metrics = AdminController.train_model(df, params)
            
            # Display metrics
            st.subheader("Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Squared Error", f"{metrics['mse']:.2f}")
            
            with col2:
                st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
            
            with col3:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            st.dataframe(metrics['feature_importance'], use_container_width=True)
            
            # Plot feature importance
            fig = AdminController.create_visualization(
                metrics['feature_importance'], 
                "Custom Plot", 
                "Importance", 
                "Feature", 
                "Bar"
            )
            st.pyplot(fig)
            
            # Automatically save the model
            if AdminController.save_model():
                st.success("Model trained and saved successfully! It will now be used for predictions.")
            else:
                st.warning("Model was trained but could not be saved to disk. It will be used for predictions in this session only.")

    # Data upload section for adding new training data
    st.subheader("Add Training Data")
    st.write("Upload a CSV file with additional training data")
    
    upload_file = st.file_uploader("Upload CSV", type=['csv'])
    if upload_file is not None:
        try:
            new_data = pd.read_csv(upload_file)
            st.write("Preview of uploaded data:")
            st.dataframe(new_data.head(), use_container_width=True)
            
            if st.button("Add to Training Data"):
                # In a real app, you would validate and merge with existing data
                st.success("Data would be added to training dataset in a real application.")
                st.info(f"Uploaded {len(new_data)} records.")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")