import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# 🟢 1. DATA PREPARATION
def prepare_data(file_path):
    df = pd.read_csv(file_path)

    # encodage
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # suppression colonnes inutiles
    df = df.drop(["Surname", "Geography"], axis=1)

    # split features/target
    X = df.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
    y = df["Exited"]

    return X, y


# 🟢 2. SPLIT DATA
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=1)


# 🟢 3. TRAIN MODEL (Intégré avec MLflow)
def train_model(X_train, y_train, X_test=None, y_test=None):
    # Définition des hyperparamètres
    n_estimators = 100
    random_state = 42

    # Configuration de l'URI de tracking MLflow locale
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Churn_Prediction_Experiment")

    # Démarrage du Run d'expérimentation MLflow
    with mlflow.start_run() as run:
        print(f"📊 [MLflow] Run démarré : {run.info.run_id}")
        
        # Initialisation et entraînement du modèle
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Enregistrement des hyperparamètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        
        # Si les données de test sont fournies, on calcule et loggue la métrique directement
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)
            print(f"📈 [MLflow] Métrique enregistrée - Accuracy: {acc:.4f}")

        # Enregistrement du modèle directement dans le registre d'artefacts MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("✅ [MLflow] Modèle, paramètres et artefacts enregistrés avec succès !")
        
    return model 


# 🟢 4. EVALUATION
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, cm


# 🟢 5. SAVE MODEL
def save_model(model, filename="model.joblib"):
    joblib.dump(model, filename)


# 🟢 6. LOAD MODEL
def load_model(filename="model.joblib"):
    return joblib.load(filename)
