# Authentication utilities
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai
import streamlit as st
import json
import tempfile
import os

from config import PROJECT_ID, LOCATION

def setup_google_auth(key_path=None):
    """Set up Google Cloud authentication and initialize Vertex AI."""
    # Try to get credentials from Streamlit secrets first
    if hasattr(st, 'secrets') and 'google_credentials' in st.secrets:
        try:
            st.sidebar.write("Found Google credentials in secrets")
            
            # Get the JSON string
            json_str = st.secrets.google_credentials["json"]
            st.sidebar.write(f"JSON string length: {len(json_str)} chars")
            st.sidebar.write(f"JSON starts with: {json_str[:50]}...")
            st.sidebar.write(f"JSON ends with: ...{json_str[-50:]}")
            
            # Check if JSON string has escaped quotes or backslashes
            if '\\"' in json_str or '\\\\' in json_str:
                st.sidebar.warning("JSON contains escaped characters that might cause issues")
            
            # Create a temporary file for the credentials
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp:
                temp.write(json_str)
                temp_key_path = temp.name
                st.sidebar.write(f"Wrote credentials to: {temp_key_path}")
            
            # Check the contents of the temporary file
            with open(temp_key_path, 'r') as f:
                file_contents = f.read()
                # Show just the beginning and end to avoid exposing the whole key
                st.sidebar.write(f"File starts with: {file_contents[:50]}...")
                st.sidebar.write(f"File ends with: ...{file_contents[-50:]}")
                
                # Try parsing the JSON to see if it's valid
                try:
                    import json
                    parsed = json.loads(file_contents)
                    st.sidebar.write("JSON is valid and can be parsed")
                    # Check for required fields
                    required_fields = ["type", "project_id", "private_key", "client_email"]
                    missing = [f for f in required_fields if f not in parsed]
                    if missing:
                        st.sidebar.error(f"Missing required fields: {', '.join(missing)}")
                except json.JSONDecodeError as e:
                    st.sidebar.error(f"Invalid JSON: {str(e)}")
            
            # Use the temporary credentials file
            try:
                credentials = Credentials.from_service_account_file(
                    temp_key_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                st.sidebar.write("Successfully loaded credentials from file")
            except Exception as e:
                st.sidebar.error(f"Error loading credentials from file: {str(e)}")
                st.sidebar.error(f"Error type: {type(e).__name__}")
                import traceback
                st.sidebar.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Clean up the temporary file
            os.unlink(temp_key_path)
            
            # Initialize Vertex AI with credentials
            try:
                vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
                st.sidebar.success("✅ Authenticated using Streamlit secrets")
            except Exception as e:
                st.sidebar.error(f"Error initializing Vertex AI: {str(e)}")
                st.sidebar.error(f"Error type: {type(e).__name__}")
                import traceback
                st.sidebar.error(f"Traceback: {traceback.format_exc()}")
                raise
                
            return credentials
            
        except Exception as e:
            st.sidebar.error(f"Error using secret credentials: {str(e)}")
            import traceback
            st.sidebar.error(f"Traceback: {traceback.format_exc()}")
            # Continue to fallback methods if secrets failed
    
    # Fall back to file-based credentials if specified
    if key_path:
        try:
            st.sidebar.write(f"Trying to use key file at: {key_path}")
            if not os.path.exists(key_path):
                st.sidebar.error(f"Key file not found: {key_path}")
            else:
                st.sidebar.write(f"Key file exists")
                
            credentials = Credentials.from_service_account_file(
                key_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            if credentials.expired:
                credentials.refresh(Request())
                
            # Initialize Vertex AI with credentials
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
            st.sidebar.success("✅ Authenticated using key file")
            return credentials
        except Exception as e:
            st.sidebar.error(f"Error using credentials file: {str(e)}")
            import traceback
            st.sidebar.error(f"Traceback: {traceback.format_exc()}")
    
    # Use application default credentials as last resort
    try:
        st.sidebar.write("Trying application default credentials")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        st.sidebar.success("✅ Authenticated using application default credentials")
        return None
    except Exception as e:
        st.sidebar.error(f"Failed to authenticate: {str(e)}")
        import traceback
        st.sidebar.error(f"Traceback: {traceback.format_exc()}")
        st.error("Authentication failed. Please check your credentials.")
        return None