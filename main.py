from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import librosa
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔄 Loading emotion detection model...")
classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)
print("✅ Model loaded!")

LABEL_MAP = {
    "neutral": "neu",
    "happy": "hap",
    "sad": "sad",
    "angry": "ang"
}

@app.get("/")
async def root():
    return {"status": "OK", "message": "ML server running"}

@app.post("/predict")
async def predict(file: UploadFile):
    tmp_path = None

    try:
        # =========================
        # SAVE FILE (WEBM / WAV)
        # =========================
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # =========================
        # LOAD AUDIO (LIBROSA 🔥)
        # =========================
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)

        # =========================
        # PREDICT
        # =========================
        results = classifier(audio)
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
