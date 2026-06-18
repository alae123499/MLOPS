import argparse
from pipeline_prefect import (
    all_flow,
    train_flow,
    evaluate_flow,
    code_flow,
    api_flow,       # 🌟 Importation du nouveau flow de l'API
    mlflow_ui_flow  # 📊 ÉTAPE 03 : Importation du flow MLflow UI
)

def main():
    parser = argparse.ArgumentParser(
        description="Run Prefect ML Pipeline"
    )

    parser.add_argument(
        "--flow",
        required=True,
        # 📊 Ajout du choix 'mlflow_ui' dans la liste des commandes valides
        choices=["all", "train", "evaluate", "code", "api", "mlflow_ui"],  
        help="Sélectionnez le flow Prefect à exécuter."
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow() 
    elif args.flow == "train":
        train_flow()

    elif args.flow == "evaluate":
        evaluate_flow()

    elif args.flow == "code":
        code_flow()

    elif args.flow == "api":  # 🌟 Exécution du flow API
        api_flow()

    elif args.flow == "mlflow_ui":  # 📊 ÉTAPE 03 : Exécution du serveur MLflow
        mlflow_ui_flow()

if __name__ == "__main__":
    main()
