import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import soundfile as sf
import tempfile
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None

def get_model():
    global classifier
    if classifier is None:
        print("🔄 Loading model...")
        classifier = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
            device=-1,
            model_kwargs={"force_download": True}
        )
        print("✅ Model loaded")
    return classifier

LABEL_MAP = {
    "neutral": "neu",
    "happy": "hap",
    "sad": "sad",
    "angry": "ang"
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

@app.post("/predict")
async def predict(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if file.content_type != "audio/wav":
        raise HTTPException(
            status_code=400,
            detail="Only WAV audio is supported"
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        # Read WAV safely using BytesIO
        audio_buffer = io.BytesIO(contents)
        audio, sr = sf.read(audio_buffer, dtype="float32")

        print(f"✅ Audio loaded: sr={sr}, shape={audio.shape}")

        if sr != 16000:
            raise HTTPException(
                status_code=400,
                detail=f"Sample rate must be 16kHz, got {sr}Hz"
            )

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono

        print(f"🎤 Processing audio: ndim={audio.ndim}, length={len(audio)}")

        model = get_model()
        results = model(audio)
        top = results[0]

        raw_label = top["label"].lower()
        confidence = float(top["score"])
        emotion = LABEL_MAP.get(raw_label, raw_label[:3])

        print(f"✅ Prediction: emotion={emotion}, confidence={confidence}")

        return {
            "emotion": emotion,
            "confidence": confidence
        }

    except Exception as e:
        print(f"❌ ML ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")