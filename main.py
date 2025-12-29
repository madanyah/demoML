import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import torch 
import librosa
import tempfile

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
            model="superb/wav2vec2-base-superb-er"
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

@app.get("/")
async def root():
    return {"status": "OK", "message": "ML server running"}

@app.post("/predict")
async def predict(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid audio file")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)

        model = get_model()
        results = model(audio)
        top = results[0]

        raw_label = top["label"].lower()
        confidence = float(top["score"])

        emotion = LABEL_MAP.get(raw_label, raw_label[:3])

        return {
            "emotion": emotion,
            "confidence": confidence
        }

    except Exception as e:
        print("❌ ML ERROR:", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
