import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Définition de la structure des données d'entrée (Features du Churn)
class CustomerData(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# 2. Initialisation de l'application FastAPI
app = FastAPI(
    title="Service de Prédiction de Churn Client",
    description="API REST MLOps pour exposer la fonction predict() du modèle.",
    version="1.0.0"
)

# Chemin vers le modèle enregistré
MODEL_PATH = "model.joblib"

# 3. Chargement du modèle au démarrage de l'API
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Modèle chargé avec succès pour l'API !")
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors du chargement du modèle : {str(e)}")
else:
    model = None
    print("⚠️ Attention : Aucun fichier 'model.joblib' trouvé. Exécutez le pipeline ML d'abord.")

# 4. Route racine (Simple vérification de santé)
@app.get("/")
def home():
    return {"status": "online", "message": "L'API de prédiction est opérationnelle."}

# 5. Route POST pour la prédiction
@app.post("/predict", summary="Prédire si un client va partir (Churn)")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle d'IA n'est pas disponible sur le serveur.")
    
    try:
        # Convertir l'objet Pydantic en format liste/tableau accepté par Scikit-Learn
        features = [[
            data.CreditScore,
            data.Age,
            data.Tenure,
            data.Balance,
            data.NumOfProducts,
            data.HasCrCard,
            data.IsActiveMember,
            data.EstimatedSalary
        ]]
        
        # Exécution de la prédiction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None
        
        return {
            "prediction": int(prediction),
            "churn_status": "Partant (Churn)" if prediction == 1 else "Fidèle",
            "probability": float(probability) if probability is not None else "N/A"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
