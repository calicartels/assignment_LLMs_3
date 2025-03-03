# test_secrets.py
import streamlit as st
import json

st.write("Checking secrets...")
if hasattr(st, 'secrets'):
    st.write("Secrets are available")
    if 'google_credentials' in st.secrets:
        st.write("Google credentials are available")
        try:
            # Try parsing the JSON
            creds = json.loads(st.secrets.google_credentials["json"])
            st.write("Successfully parsed credentials JSON")
            st.write(f"Project ID: {creds['project_id']}")
        except Exception as e:
            st.write(f"Error parsing credentials: {str(e)}")
    else:
        st.write("No google_credentials in secrets")
else:
    st.write("No secrets available")