import subprocess
from prefect import flow, task
from model_pipeline import *

# ======================
# 🟢 TASKS ML
# ======================


@task
def task_prepare():
    X, y = prepare_data("Churn_Modelling.csv")
    return X, y


@task
def task_split(X, y):
    return split_data(X, y)


@task
def task_train(X_train, y_train):
    return train_model(X_train, y_train)


@task
def task_evaluate(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


@task
def task_save(model):
    save_model(model)


@task
def task_load():
    return load_model()


# ======================
# 🟢 CODE QUALITY TASKS (Optimisées pour vos fichiers)
# ======================


@task
def install_dependencies():
    print("📦 Téléchargement et installation des dépendances...")
    # Optionnel : subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)


@task
def format_code():
    print("✨ 1. Formatage du code avec Black...")
    res = subprocess.run(
        ["black", "model_pipeline.py", "main.py", "pipeline_prefect.py"],
        capture_output=True,
        text=True,
    )
    print(res.stdout)


@task
def code_quality():
    print("🔍 2. Analyse de la qualité du code avec Flake8...")
    res = subprocess.run(
        ["flake8", "model_pipeline.py", "main.py", "pipeline_prefect.py"],
        capture_output=True,
        text=True,
    )
    print(res.stdout)


@task
def security_check():
    print("🛡️ 3. Scan de sécurité avec Bandit...")
    res = subprocess.run(
        ["bandit", "-r", "model_pipeline.py", "main.py", "pipeline_prefect.py"],
        capture_output=True,
        text=True,
    )
    print(res.stdout)


@task
def unit_tests():
    print("🧪 4. Exécution des tests unitaires avec Pytest...")
    res = subprocess.run(["pytest"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError("❌ Les tests unitaires ont échoué ! Arrêt du pipeline.")


# ======================
# 🟢 FLOWS
# ======================


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    code_quality()
    security_check()
    unit_tests()


@flow(name="train")
def train_flow():
    X, y = task_prepare()
    X_train, X_test, y_train, y_test = task_split(X, y)
    model = task_train(X_train, y_train)
    task_save(model)


@flow(name="evaluate")
def evaluate_flow():
    model = task_load()
    X, y = task_prepare()
    X_train, X_test, y_train, y_test = task_split(X, y)
    acc, cm = task_evaluate(model, X_test, y_test)
    print("Accuracy:", acc)


@flow(name="all")
def all_flow():
    # Étape 04 : Appel des étapes liées au code avant le ML
    code_flow()

    # Suite du pipeline ML
    print("🚀 Début du pipeline Machine Learning...")
    X, y = task_prepare()
    X_train, X_test, y_train, y_test = task_split(X, y)
    model = task_train(X_train, y_train)
    acc, cm = task_evaluate(model, X_test, y_test)
    task_save(model)
# =====================================================================
# 🟢 ÉTAPE 04 (ATELIER 4) : TÂCHE ET FLOW POUR L'API FASTAPI
# =====================================================================

@task(name="start_fastapi_server")
def start_fastapi_server():
    import subprocess
    print("🌐 [Task] Lancement du serveur d'inférence FastAPI...")
    
    # On utilise Popen pour lancer uvicorn sans bloquer le script Prefect
    cmd = ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    subprocess.Popen(cmd)
    print("🚀 API FastAPI démarrée avec succès ! Visitez http://127.0.0.1:8000/docs")


@flow(name="api")
def api_flow():
    """Flow Prefect dédié au démarrage et à la gestion du service REST."""
    start_fastapi_server()
# =====================================================================
# 🔵 ÉTAPE 03 (ATELIER 5) : COMMANDE AUTOMATISÉE MLFLOW
# =====================================================================

@task(name="start_mlflow_ui")
def start_mlflow_ui():
    import subprocess
    print("📊 [Task] Lancement de l'interface graphique MLflow (Backend SQLite)...")
    
    # Commande pour lancer le serveur sans bloquer le script
    cmd = [
        "mlflow", "ui", 
        "--backend-store-uri", "sqlite:///mlflow.db", 
        "--host", "0.0.0.0", 
        "--port", "5000"
    ]
    subprocess.Popen(cmd)
    print("🚀 Serveur MLflow disponible sur http://127.0.0.1:5000")

@flow(name="mlflow_ui")
def mlflow_ui_flow():
    """Flow Prefect dédié au déploiement de l'interface de tracking MLflow."""
    start_mlflow_ui()
