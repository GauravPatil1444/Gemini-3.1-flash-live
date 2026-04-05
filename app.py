import streamlit as st
from websocket import create_connection
import json

st.set_page_config(page_title="Gemini Live Chat", page_icon="🎤")
st.title("Gemini Multimodal Live")

# Use session state to keep track of the chat history in the UI
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Audio input widget
audio_file = st.audio_input("Record your message")

if audio_file:
    try:
        # Connect to your FastAPI proxy
        ws = create_connection("ws://localhost:8000/ws/gemini")
        
        # Send the audio file bytes
        ws.send_binary(audio_file.read())
        
        with st.spinner("Gemini is thinking..."):
            # Listen for transcripts (User + Gemini)
            while True:
                try:
                    # Set a timeout so we don't wait forever
                    ws.settimeout(5.0) 
                    response = json.loads(ws.recv())
                    
                    if "user" in response:
                        st.session_state.chat_log.append(f"🎤 You: {response['user']}")
                    if "gemini" in response:
                        st.session_state.chat_log.append(f"🤖 Gemini: {response['gemini']}")
                        break # Exit loop after getting Gemini's response
                except Exception:
                    break # Break if timeout or connection closes
        ws.close()
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")

# Display the chat history
for message in st.session_state.chat_log:
    st.write(message)