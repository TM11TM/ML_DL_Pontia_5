import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocesar_datos(X_hotel_train, X_hotel_test):
    # Codificación one-hot para columnas categóricas
    X_hotel_train = pd.get_dummies(X_hotel_train)
    X_hotel_test = pd.get_dummies(X_hotel_test)

    # Alinear columnas entre train y test
    X_hotel_train, X_hotel_test = X_hotel_train.align(X_hotel_test, join='left', axis=1, fill_value=0)

    # Escalado
    scaler = StandardScaler()
    X_hotel_train_scaled = scaler.fit_transform(X_hotel_train)
    X_hotel_test_scaled = scaler.transform(X_hotel_test)

    return X_hotel_train_scaled, X_hotel_test_scaled
