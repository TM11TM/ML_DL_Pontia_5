from data_loader import cargar_datos, dividir_datos
from preprocess import preprocesar_datos
from model import entrenar_modelos
from metrics import evaluar_modelos
import config

def main():
    df_hotel_preprocessed = cargar_datos(config.PATH_DATASET_HOTEL)
    X_hotel_train, X_hotel_test, y_hotel_train, y_hotel_test = dividir_datos(df_hotel_preprocessed, test_size=0.2, random_state=42)

     # Preprocesar (one-hot + escalado)
    X_hotel_train_encoded, X_hotel_test_encoded = preprocesar_datos(X_hotel_train, X_hotel_test)
    
    y_hotel_train = y_hotel_train.astype(int)
    y_hotel_test = y_hotel_test.astype(int)
    
    X_hotel_train_encoded, X_hotel_test_encoded = preprocesar_datos(X_hotel_train, X_hotel_test)
    
    modelos = entrenar_modelos(X_hotel_train_encoded, y_hotel_train)
 
    evaluar_modelos(modelos, X_hotel_test_encoded, y_hotel_test)

if __name__ == "__main__":
    main()
