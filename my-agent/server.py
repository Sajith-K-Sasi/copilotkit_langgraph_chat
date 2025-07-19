from fastapi import FastAPI, HTTPException, UploadFile, File

from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from elevenlabs.client import ElevenLabs
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent

from io import BytesIO
import os
from dotenv import load_dotenv


app = FastAPI(title="FastAPI CopilotKit LangGraph ElevenLabs Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Initialize the CopilotKit SDK
from agent import graph
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="agent",
            description="An example agent to use as a starting point for your own agent.",
            graph=graph,
        )
    ],
    actions=[],
)
add_fastapi_endpoint(app, sdk, "/copilotkit_remote", max_workers=10)




## ElevenLabs

# Load environment variables
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


@app.get("/tts")
async def text_to_speech(text: str):

    response = elevenlabs_client.text_to_speech.stream(
        text=text,
        voice_id="IKne3meq5aSn9XLyUdCD",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        )

    audio_stream = BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)
    
    return StreamingResponse(
        audio_stream,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=output.mp3"},
    )


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    
    # Read audio data
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="No audio data provided.")

    transcription = elevenlabs_client.speech_to_text.convert(
        file=audio_data,
        model_id="scribe_v1", # Model to use, for now only "scribe_v1" is supported
        tag_audio_events=True, # Tag audio events like laughter, applause, etc.
        language_code="eng", # Language of the audio file. If set to None, the model will detect the language automatically.
        diarize=True, # Whether to annotate who is speaking
    )

    return transcription


def main():
    """Run the uvicorn server."""
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
 
if __name__ == "__main__":
    main()

