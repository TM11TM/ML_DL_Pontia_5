from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluar_modelo(y_real, y_predicho, y_proba=None):
    resultados = {
        "Accuracy": accuracy_score(y_real, y_predicho),
        "Precision": precision_score(y_real, y_predicho),
        "Recall": recall_score(y_real, y_predicho),
        "F1-Score": f1_score(y_real, y_predicho)
    }
    if y_proba is not None:
        resultados["AUC"] = roc_auc_score(y_real, y_proba)
    return resultados

def evaluar_modelos(modelos, X_test, y_test):
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

        resultados = evaluar_modelo(y_test, y_pred, y_proba)

        print(f"Resultados para {nombre}:")
        for metrica, valor in resultados.items():
            print(f"  {metrica}: {valor:.4f}")
        print("-" * 40)
