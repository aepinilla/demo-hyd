import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.title("Simple Chatbot")

# Define avatars
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize model
GPT_MODEL = "gpt-4o"

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# Chat input
if user_message := st.chat_input("How can I help?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_message)
    
    # Generate and display assistant response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        
        try:
            # Get response from OpenAI
            full_response = ""
            for chunk in client.chat.completions.create(
                model=GPT_MODEL,
                messages=st.session_state.messages,
                stream=True,
            ):
                content = chunk.choices[0].delta.content or ""
                full_response += content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
