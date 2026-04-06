import asyncio
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

app = FastAPI()

from dotenv import load_dotenv
import os
# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")



# 🔐 API key
client = genai.Client(api_key=api_key)

MODEL = "gemini-3.1-flash-live-preview"

CONFIG = types.LiveConnectConfig(
    response_modalities=[types.Modality.AUDIO],
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    system_instruction=types.Content(
        parts=[types.Part(text="You are a helpful AI assistant.")]
    ),
)


@app.websocket("/ws/audio")
async def websocket_audio(ws: WebSocket):
    await ws.accept()
    print("🔌 Client connected")

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:

            # -------- RECEIVE FROM CLIENT (MIC AUDIO) -------- #
            async def receive_from_client():
                while True:
                    data = await ws.receive_bytes()

                    await session.send_realtime_input(
                        audio={
                            "data": data,
                            "mime_type": "audio/pcm"
                        }
                    )

            # -------- SEND TO CLIENT (AUDIO + TEXT) -------- #
            async def send_to_client():
                while True:
                    turn = session.receive()

                    async for response in turn:
                        content = response.server_content
                        if not content:
                            continue

                        # 🎧 AUDIO
                        if content.model_turn:
                            for part in content.model_turn.parts:
                                if part.inline_data:
                                    await ws.send_bytes(part.inline_data.data)

                                if part.text:
                                    await ws.send_json({
                                        "type": "text",
                                        "role": "model",
                                        "text": part.text
                                    })

                        # 🧠 USER TRANSCRIPT
                        if content.input_transcription and content.input_transcription.text:
                            await ws.send_json({
                                "type": "text",
                                "role": "user",
                                "text": content.input_transcription.text
                            })

                        # 🤖 MODEL TRANSCRIPT
                        if content.output_transcription and content.output_transcription.text:
                            await ws.send_json({
                                "type": "text",
                                "role": "model",
                                "text": content.output_transcription.text
                            })

            async with asyncio.TaskGroup() as tg:
                tg.create_task(receive_from_client())
                tg.create_task(send_to_client())

    except WebSocketDisconnect:
        print("❌ Client disconnected")

    except Exception as e:
        print("❌ Error:", e)