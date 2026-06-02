import os
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Charger les variables d'environnement
load_dotenv()

# Client Groq (charge au demarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")

# Initialiser l'app FastAPI
app = FastAPI(title="SenSante API", version="4.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modele ML
print("Chargement du modele...")
model = joblib.load("models/model.pkl")
print(f"Modele charge : {model.classes_.tolist()}")

# Schemas Pydantic
class PatientData(BaseModel):
    age: int = Field(..., description="Age du patient")
    sexe: str = Field(..., description="M ou F")
    temperature: float = Field(..., description="Temperature en Celsius")
    tension: int = Field(..., description="Tension systolique")
    toux: int = Field(..., description="0 ou 1")
    fatigue: int = Field(..., description="0 ou 1")
    maux_tete: int = Field(..., description="0 ou 1")
    region: str = Field(..., description="Region du Senegal")

class DiagnosticResult(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    classes: list = Field(..., description="Toutes les classes")
    probabilites: list = Field(..., description="Toutes les probabilites")

class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)

class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en francais")
    modele_llm: str = Field(default="llama-3.1-8b-instant", description="Modele LLM utilise")

# Routes
@app.get("/health")
def health():
    """Verifier que l'API fonctionne."""
    return {"status": "ok", "version": "4.0"}

@app.post("/predict", response_model=DiagnosticResult)
def predict(patient: PatientData):
    """Predire le diagnostic a partir des donnees patient."""
    sexe_encoded = 1 if patient.sexe == "M" else 0

    regions = ["Dakar", "Thies", "Saint-Louis", "Ziguinchor", "Autre"]
    region_encoded = regions.index(patient.region) if patient.region in regions else 4

    features = np.array([[
        patient.age,
        sexe_encoded,
        patient.temperature,
        patient.tension,
        patient.toux,
        patient.fatigue,
        patient.maux_tete,
        region_encoded
    ]])

    prediction = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    idx = list(model.classes_).index(prediction)

    return DiagnosticResult(
        diagnostic=prediction,
        probabilite=round(float(probas[idx]), 2),
        classes=model.classes_.tolist(),
        probabilites=[round(float(p), 2) for p in probas]
    )

# System prompt pour le LLM
SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours une consultation medicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en francais avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun"
        )

    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(explication=explication)
    from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Servir le frontend comme fichier statique
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    """Servir la page d'accueil."""
    return FileResponse("frontend/index.html")