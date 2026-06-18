from pipeline_prefect import all_flow
from datetime import timedelta

if __name__ == "__main__":
    print("📢 Enregistrement et lancement des services de déploiement Prefect...")
    
    # .serve() publie le schéma sur le serveur ET crée un worker local
    all_flow.serve(
        name="ml-pipeline-all",
        tags=["production", "mlops"],
        interval=timedelta(days=1), # Planification automatique toutes les 24 heures
        description="Pipeline complet exécuté quotidiennement comprenant les checks de code et l'entraînement ML."
    )
