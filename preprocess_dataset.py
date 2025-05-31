import pandas as pd

def preprocess_dataset(nav_path, msci_path, maestro_path):
    """
    Load and clean the input datasets.
    
    Parameters:
    - nav_path (str): Path to the NAVs .pickle file.
    - msci_path (str): Path to the MSCI .csv file.
    - maestro_path (str): Path to the master .csv file.

    Returns:
    - maestro (pd.DataFrame): Cleaned metadata for each fund.
    - fondos (pd.DataFrame): NAV time series with valid funds.
    """
    df_fondos = pd.read_pickle(nav_path)
    maestro = pd.read_csv(maestro_path, dtype={'allfunds_id': str})
    
    # Filtrar los fondos que tienen datos de NAVs
    allfunds_ids = df_fondos.columns.tolist()
    maestro = maestro[maestro['allfunds_id'].isin(allfunds_ids)]

    # Eliminar fondos sin datos en los últimos 60 días
    last_60_days = df_fondos.tail(60)
    invalid_columns = last_60_days.columns[last_60_days.isna().all()].tolist()
    fondos = df_fondos.drop(columns=invalid_columns)

    # Actualizar allfunds_ids y filtrar maestro de nuevo
    allfunds_ids = fondos.columns.tolist()
    maestro = maestro[maestro['allfunds_id'].isin(allfunds_ids)]

    # Rellenar valores nulos
    fondos = fondos.ffill().bfill()
    
    return maestro, fondos
