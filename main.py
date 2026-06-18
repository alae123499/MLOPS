from model_pipeline import (
    prepare_data,
    split_data,
    train_model,
    evaluate_model,
    save_model,
)

# 🟢 1. Charger dataset
file_path = "Churn_Modelling.csv"

X, y = prepare_data(file_path)

# 🟢 2. Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# 🟢 3. Train model
model = train_model(X_train, y_train)

# 🟢 4. Evaluate
accuracy, cm = evaluate_model(model, X_test, y_test)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# 🟢 5. Save model
save_model(model)

print("Model saved successfully ✔")
