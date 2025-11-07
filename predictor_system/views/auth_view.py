import streamlit as st
import time
from controllers.auth_controller import AuthController
from config.settings import USERS, USER_TYPES

def display_role_selection():
    """Display the role selection page"""
    st.header("Select Your Role")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Admin", use_container_width=True):
            st.session_state.role = "Admin"
            st.session_state.page = 'login'
            st.rerun()
    
    with col2:
        if st.button("User", use_container_width=True):
            st.session_state.role = "User"
            st.session_state.page = 'login'
            st.rerun()
    
    st.markdown("---")
    st.write("Please select a role to continue to the login page.")
    
    # Display available test accounts
    with st.expander("Available Test Accounts"):
        st.markdown("### Admin Accounts")
        st.markdown("- Username: `admin1`, Password: `admin123`")
        st.markdown("- Username: `admin2`, Password: `admin456`")
        st.markdown("- Username: `admin3`, Password: `admin789`")
        
        st.markdown("### User Accounts")
        st.markdown("- Username: `student1`, Password: `stud123` (Student)")
        st.markdown("- Username: `engineer1`, Password: `eng123` (Software Engineer)")
        st.markdown("- Username: `recruiter1`, Password: `rec123` (Recruiter)")

def display_login_page():
    """Display the login page"""
    selected_role = st.session_state.role
    st.header(f"Login as {selected_role}")
    
    # Back button
    if st.button("← Back to Role Selection"):
        st.session_state.page = 'role_selection'
        st.session_state.role = None
        st.rerun()
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        # Only show user type selection for regular users if not predefined
        user_type = None
        if selected_role == "User":
            # Check if username exists and has a predefined user_type
            if username in USERS and USERS[username]["role"] == "User":
                user_type = USERS[username]["user_type"]
                st.info(f"Account type: {user_type}")
            else:
                user_type = st.selectbox("User Type", USER_TYPES)
        
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            # Pass the selected role as an optional parameter
            login_result = AuthController.login(username, password, selected_role, user_type)
            if login_result["success"]:
                st.success("Login successful!")
                
                # Show message if user selected the wrong role
                if "message" in login_result:
                    st.info(login_result["message"])
                
                # Simulate loading
                with st.spinner("Redirecting..."):
                    time.sleep(1)
                
                st.rerun()
            else:
                st.error(login_result["message"])