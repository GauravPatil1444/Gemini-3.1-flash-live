import streamlit as st
import asyncio
import websockets
import numpy as np
import threading
import json
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

WS_URL = "ws://localhost:8000/ws/audio"

st.title("🎤 Voice AI Assistant")

# ================== SESSION STATE INIT ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ws" not in st.session_state:
    st.session_state.ws = None

if "receiver_started" not in st.session_state:
    st.session_state.receiver_started = False

if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()


# ================== BACKGROUND RECEIVER ==================
def receiver_thread():
    loop = st.session_state.loop
    asyncio.set_event_loop(loop)

    async def receive():
        ws = st.session_state.ws
        while True:
            try:
                msg = await ws.recv()

                if isinstance(msg, str):
                    data = json.loads(msg)
                    st.session_state.messages.append(data)

            except Exception as e:
                print("Receiver error:", e)
                break

    loop.run_until_complete(receive())


# ================== AUDIO PROCESSOR ==================
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        if st.session_state.ws is None:
            # Create websocket connection
            st.session_state.ws = asyncio.run(websockets.connect(WS_URL))

            # Start receiver thread ONCE
            if not st.session_state.receiver_started:
                st.session_state.receiver_started = True
                thread = threading.Thread(target=receiver_thread, daemon=True)
                thread.start()

    def recv(self, frame):
        audio = frame.to_ndarray().flatten().astype(np.int16)

        try:
            asyncio.run(
                st.session_state.ws.send(audio.tobytes())
            )
        except Exception as e:
            print("Send error:", e)

        return frame


# ================== WEBRTC ==================
webrtc_ctx = webrtc_streamer(
    key="voice-ai",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx.state.playing:
    st.success("🎙️ Listening... Speak now!")


# ================== UI ==================
st.write("### 🧾 Conversation")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['text']}")
    else:
        st.markdown(f"**🤖 Gemini:** {msg['text']}")