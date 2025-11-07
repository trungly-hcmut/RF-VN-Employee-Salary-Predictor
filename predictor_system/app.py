import streamlit as st
from views.auth_view import display_role_selection, display_login_page
from views.admin_view import display_dashboard, display_admin_user_management, display_admin_data_visualization, display_admin_model_management
from views.user_view import display_salary_prediction
from config.settings import set_page_config

def main():
    # Set page configuration
    set_page_config()
    
    # Initialize session state variables if they don't exist
    if 'page' not in st.session_state:
        st.session_state.page = 'role_selection'
    if 'role' not in st.session_state:
        st.session_state.role = None
    
    # Display different pages based on session state
    if st.session_state.page == 'role_selection':
        display_role_selection()
    elif st.session_state.page == 'login':
        display_login_page()
    elif st.session_state.page == 'dashboard':
        display_dashboard()
    elif st.session_state.page == 'salary_prediction':
        display_salary_prediction()
    elif st.session_state.page == 'admin_user_management':
        display_admin_user_management()
    elif st.session_state.page == 'admin_data_visualization':
        display_admin_data_visualization()
    elif st.session_state.page == 'admin_model_management':
        display_admin_model_management()

if __name__ == "__main__":
    main()