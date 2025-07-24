import pandas as pd
from sklearn.model_selection import train_test_split

def cargar_datos(path):
    df_hotel = pd.read_csv(path) 
    df_hotel.drop_duplicates(inplace=True)
    df_hotel_preprocessed = df_hotel.copy()
    
    nulos = df_hotel_preprocessed.isna().sum()
    
    
    # Eliminamos las siguientes columnas con más de un 50% de nulos
    columnas_a_eliminar = nulos[nulos > (df_hotel_preprocessed.shape[0] * 0.5)].index
    if len(columnas_a_eliminar) > 0:
        df_hotel_preprocessed = df_hotel_preprocessed.drop(columns=columnas_a_eliminar)

    #Observamos la variable dependienteç
    target_column = 'is_canceled'
    df_hotel_preprocessed[target_column] = df_hotel_preprocessed[target_column].astype(str)
    
    return df_hotel_preprocessed
 
def dividir_datos(df_hotel_preprocessed, target_column="is_canceled", test_size=0.2, random_state=42):
    X_hotel = df_hotel_preprocessed.drop(columns=target_column)
    y_hotel = df_hotel_preprocessed[target_column]

    X_hotel_train, X_hotel_test, y_hotel_train, y_hotel_test = train_test_split(
        X_hotel, y_hotel, test_size=test_size, random_state=random_state
    )

    return X_hotel_train, X_hotel_test, y_hotel_train, y_hotel_test