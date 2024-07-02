import streamlit as st

from script import Context, generate_response

# Initialize context for the user
if 'user_context' not in st.session_state:
    st.session_state['user_context'] = Context()

# Streamlit app layout
st.title("Food Nutrition Chatbot")

user_id = st.text_input("User ID", value="default_user")
user_input = st.text_input("Ask a question about food nutrition:")

if st.button("Submit"):
    if user_input:
        response = generate_response(user_id, user_input)
        st.write(f"Response: {response}")
    else:
        st.write("Please enter a query.")

