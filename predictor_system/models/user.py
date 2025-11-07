from config.settings import USERS

class UserModel:
    """User model for authentication and user management"""
    
    @staticmethod
    def validate_login(username, password, role=None, user_type=None):
        """
        Validates user credentials against the predefined USERS dictionary
        
        Args:
            username (str): Username
            password (str): Password
            role (str, optional): Role to validate against
            user_type (str, optional): User type for new users
            
        Returns:
            dict: Validation result with success status, message, role, and user_type
        """
        if username in USERS:
            user_data = USERS[username]
            
            # Check if the role matches if a role was specified
            if role and user_data["role"] != role:
                return {
                    "success": False, 
                    "message": f"This account is a {user_data['role']} account, not a {role} account.",
                    "role": user_data["role"]
                }
            
            # Check password
            if user_data["password"] == password:
                # For User role, check user_type if applicable
                if user_data["role"] == "User":
                    actual_user_type = user_data.get("user_type", user_type)
                    return {
                        "success": True, 
                        "role": user_data["role"], 
                        "user_type": actual_user_type
                    }
                return {
                    "success": True, 
                    "role": user_data["role"], 
                    "user_type": None
                }
            else:
                return {"success": False, "message": "Invalid password"}
        else:
            # For non-existing users, allow login with any password for demo purposes
            # In a real application, you would reject unknown users
            if username and password:
                return {
                    "success": True, 
                    "role": role or "User",  # Default to User if no role specified
                    "user_type": user_type
                }
            return {"success": False, "message": "Invalid username or password"}
    
    @staticmethod
    def get_all_users():
        """Return all users as a list of dictionaries"""
        user_data = []
        for username, details in USERS.items():
            user_dict = {
                "Username": username,
                "Role": details["role"],
                "User Type": details.get("user_type", "N/A")
            }
            user_data.append(user_dict)
        return user_data