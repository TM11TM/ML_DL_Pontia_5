### model.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


def entrenar_modelos(X_train, y_train):
    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "Árbol de Decisión": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False),
        "Red Neuronal": MLPClassifier(max_iter=300)
    }

    modelos_entrenados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        modelos_entrenados[nombre] = modelo
    return modelos_entrenados
