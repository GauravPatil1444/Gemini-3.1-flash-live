import asyncio
from google import genai
import pyaudio
from google.genai import types
client = genai.Client(api_key="AIzaSyD8vwPrOfAzI3HIC_dlwRcAQuTxBK2rheY")

# --- pyaudio config ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()

# --- Live API config ---
MODEL = "gemini-3.1-flash-live-preview"
# CONFIG = {
#     "response_modalities": ["AUDIO"],
#     "system_instruction": "You are a helpful and friendly AI assistant.",
# }
from google.genai import types

CONFIG = types.LiveConnectConfig(
    # Keep AUDIO enabled to prevent the 1011 Internal Error
    response_modalities=[types.Modality.AUDIO],
    
    # This enables the "User Transcription" you were looking for
    input_audio_transcription=types.AudioTranscriptionConfig(),
    
    # This ensures Gemini's response is also available as text
    output_audio_transcription=types.AudioTranscriptionConfig(),
    
    system_instruction=types.Content(
        parts=[types.Part(text="You are a helpful AI assistant. Responses should be concise.")]
    ),
)

audio_queue_output = asyncio.Queue()
event_queue = asyncio.Queue()
audio_queue_mic = asyncio.Queue(maxsize=5)
audio_stream = None

async def listen_audio():
    """Listens for audio and puts it into the mic audio queue."""
    global audio_stream
    mic_info = pya.get_default_input_device_info()
    audio_stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
    )
    kwargs = {"exception_on_overflow": False} if __debug__ else {}
    while True:
        data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
        await audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})

async def send_realtime(session):
    """Sends audio from the mic audio queue to the GenAI session."""
    while True:
        msg = await audio_queue_mic.get()
        await session.send_realtime_input(audio=msg)
async def receive_audio(session):
    while True:
        turn = session.receive()
        async for response in turn:
            content = response.server_content
            if not content:
                continue

            # 1. Check for Model Text (if the model sends text parts)
            if content.model_turn:
                for part in content.model_turn.parts:
                    if part.text:
                        print(f"Gemini (Text Part): {part.text}")
            
            # 2. Check for User Transcription (What YOU said)
            if content.input_transcription and content.input_transcription.text:
                print(f"User: {content.input_transcription.text}")

            # 3. Check for Gemini Transcription (What the MODEL said)
            if content.output_transcription and content.output_transcription.text:
                print(f"Gemini: {content.output_transcription.text}")
async def play_audio():
    """Plays audio from the speaker audio queue."""
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
    )
    while True:
        bytestream = await audio_queue_output.get()
        await asyncio.to_thread(stream.write, bytestream)


async def run():
    """Main function to run the audio loop."""
    try:
        async with client.aio.live.connect(
            model=MODEL, config=CONFIG
        ) as live_session:
            print("Connected to Gemini. Start speaking!")
            async with asyncio.TaskGroup() as tg:
                tg.create_task(send_realtime(live_session))
                tg.create_task(listen_audio())
                tg.create_task(receive_audio(live_session))
                tg.create_task(play_audio())
    except asyncio.CancelledError:
        pass
    finally:
        if audio_stream:
            audio_stream.close()
        pya.terminate()
        print("\nConnection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")