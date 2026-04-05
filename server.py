import asyncio
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

app = FastAPI()

# SECURITY: Use an environment variable instead of hardcoding
API_KEY = "AIzaSyD8vwPrOfAzI3HIC_dlwRcAQuTxBK2rheY"
MODEL_ID = "gemini-3.1-flash-live-preview"

client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})

@app.websocket("/ws/gemini")
async def gemini_proxy(websocket: WebSocket):
    await websocket.accept()
    print("Streamlit client connected.")

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        )
    )

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            
            async def receive_from_gemini():
                """Forwards Gemini's text and audio transcripts to Streamlit."""
                try:
                    async for response in session.receive():
                        content = response.server_content
                        if not content: continue

                        data_to_send = {"type": "transcript"}
                        
                        # Capture User Transcript
                        if content.input_transcription:
                            data_to_send["user"] = content.input_transcription.text
                        
                        # Capture Gemini Transcript
                        if content.output_transcription:
                            data_to_send["gemini"] = content.output_transcription.text

                        # If we have text, send it to Streamlit
                        if "user" in data_to_send or "gemini" in data_to_send:
                            await websocket.send_json(data_to_send)
                except Exception as e:
                    print(f"Error receiving from Gemini: {e}")

            async def send_to_gemini():
                """Forwards Streamlit audio chunks to Gemini."""
                try:
                    while True:
                        # Receive binary audio from Streamlit
                        audio_data = await websocket.receive_bytes()
                        
                        # STRIP WAV HEADER: Gemini expects raw PCM, not a WAV file
                        if audio_data.startswith(b'RIFF'):
                            audio_data = audio_data[44:]

                        # FIX: Pass dict directly, NOT in a list [dict]
                        await session.send_realtime_input(
                            audio={"data": audio_data, "mime_type": "audio/pcm"}
                        )
                except WebSocketDisconnect:
                    print("Streamlit disconnected.")
                except Exception as e:
                    print(f"Error sending to Gemini: {e}")

            # Run both loops concurrently
            await asyncio.gather(receive_from_gemini(), send_to_gemini())

    except Exception as e:
        print(f"Global Session Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)