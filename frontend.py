import streamlit as st
import requests

# Set the FastAPI URL
api_url = "http://127.0.0.1:8000/query"

st.title("PartSelect Agent")

# Input field for user to enter their query
query = st.text_input("Enter your query:", "")

# Button to send the query to the API
if st.button("Submit"):
    if query:
        # Prepare the data to be sent to the API
        payload = {"query": query}
        
        # Make a POST request to the FastAPI backend
        response = requests.post(api_url, json=payload)
        
        # Display the API response
        if response.status_code == 200:
            result = response.json().get("result", [])
            if result:
                st.success("API Response:")
                st.write(result)
            else:
                st.warning("No results found.")
        else:
            st.error(f"API returned an error: {response.status_code}")
    else:
        st.warning("Please enter a query before submitting.")
