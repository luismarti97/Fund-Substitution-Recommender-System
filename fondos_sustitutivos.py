import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import preprocess_dataset

st.set_page_config(page_title='Sustitución de Fondos de Inversión', page_icon=':chart_with_upwards_trend:', layout='wide')

st.markdown("""
<style>
.selected-fund {
    background-color: #ffeb3b;
    color: #000;
    font-weight: bold;
    padding: 10px;
    border-radius: 5px;
}
.sustitute-fund {
    background-color: #e0f7fa;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# Cargar y preprocesar los datos con caching
@st.cache_data
def load_data():
    maestro, fondos = preprocess_dataset('navs (1).pickle', 'MSCI (2).csv', 'maestro (2).csv')
    fondos_clasificados = pd.read_csv('clusterized_funds.csv', dtype={'allfunds_id': str})
    fondos_clasificados_normalizados = fondos_clasificados.copy()
    
    # Establecer allfunds_id como índice
    fondos_clasificados.set_index('allfunds_id', inplace=True)
    
    columnas_numericas = ['rentabilidad_acumulada', 'volatilidad', 'ongoing_charges', 'management_fee', 'income']
    columnas_binarias = ['clean_share']
    
    fondos = fondos.ffill()
    fondos = fondos.bfill()
    
    scaler = StandardScaler()
    fondos_clasificados_normalizados[columnas_numericas] = scaler.fit_transform(fondos_clasificados[columnas_numericas])
    
    return maestro, fondos, fondos_clasificados, fondos_clasificados_normalizados

maestro, fondos, fondos_clasificados, fondos_clasificados_normalizados = load_data()

def obtener_fondos_sustitutivos(isin, criterios, num_sustitutivos=15):
    columnas_numericas = ['rentabilidad_acumulada', 'volatilidad', 'ongoing_charges', 'management_fee']
    fondo = fondos_clasificados_normalizados[fondos_clasificados_normalizados['isin'] == isin]
    if fondo.empty:
        return None, 'Este fondo ha sido descartado al hacer el ejercicio por falta de datos históricos.'
    
    cluster = fondo['cluster'].values[0]
    fondos_cluster = fondos_clasificados_normalizados[fondos_clasificados_normalizados['cluster'] == cluster]
    
    if len(fondos_cluster) < 10:
        return None, 'Este fondo no tiene sustitutivos.'
    
    fondos_sustitutivos = fondos_cluster[~fondos_cluster['isin'].isin([isin])]
    
    # Calcular distancia euclidiana
    fondos_sustitutivos['distance'] = np.sqrt(
        np.sum((fondos_sustitutivos[columnas_numericas] - fondo[columnas_numericas].values[0])**2, axis=1)
    )
    
    fondos_sustitutivos = fondos_sustitutivos.sort_values('distance').head(num_sustitutivos)
    
    # Aplicar criterios de exclusión de manera unificada
    if criterios:
        mask = pd.Series([True] * len(fondos_sustitutivos), index=fondos_sustitutivos.index)
        for criterio, valor in criterios.items():
            if criterio == 'asset_type':
                mask &= fondos_sustitutivos[criterio] != valor
            elif criterio == 'currency':
                mask &= fondos_sustitutivos['currency'] != valor
            elif criterio == 'geo_zone':
                mask &= fondos_sustitutivos['geo_zone'] != valor
            elif criterio == 'management_fee':
                valor_management_fee = float(valor)
                mask &= fondos_sustitutivos['management_fee'] <= valor_management_fee
            elif criterio == 'ongoing_charges':
                valor_ongoing_charges = float(valor)
                mask &= fondos_sustitutivos['ongoing_charges'] <= valor_ongoing_charges
            elif criterio == 'rentabilidad_acumulada':
                valor_rentabilidad_acumulada = float(valor)
                mask &= fondos_sustitutivos['rentabilidad_acumulada'] >= valor_rentabilidad_acumulada
            elif criterio == 'volatilidad':
                valor_volatilidad = float(valor)
                mask &= fondos_sustitutivos['volatilidad'] <= valor_volatilidad
        
        fondos_sustitutivos = fondos_sustitutivos[mask]
    
    if fondos_sustitutivos.empty:
        return None, 'No se encontraron fondos sustitutivos que cumplan con los criterios.'
    
    return fondos_sustitutivos, None

st.title('Sustitución de Fondos de Inversión')

st.markdown("""
Esta aplicación permite encontrar fondos de inversión sustitutos basados en criterios específicos y características de los fondos actuales.
""")

st.sidebar.header('Parámetros del Fondo')
isin = st.sidebar.text_input('Introduce el ISIN del fondo:', key='isin_input')

fondo_seleccionado = fondos_clasificados_normalizados[fondos_clasificados_normalizados['isin'] == isin]

if not fondo_seleccionado.empty:
    criterios_exclusivos = ['asset_type', 'currency', 'geo_zone', 'management_fee', 'ongoing_charges', 'rentabilidad_acumulada', 'volatilidad']
    
    criterio_1 = st.sidebar.selectbox('Criterio que no te gusta 1:', criterios_exclusivos, key='criterio_1_select')
    criterio_1_valor = st.sidebar.text_input(f'Valor del {criterio_1} que no te gusta:', value=fondo_seleccionado[criterio_1].values[0], key='criterio_1_valor', disabled=True)
    criterio_2 = st.sidebar.selectbox('Criterio que no te gusta 2:', criterios_exclusivos, key='criterio_2_select')
    criterio_2_valor = st.sidebar.text_input(f'Valor del {criterio_2} que no te gusta:', value=fondo_seleccionado[criterio_2].values[0], key='criterio_2_valor', disabled=True)
else:
    st.sidebar.warning('Introduce un ISIN válido para seleccionar criterios.')
    criterio_1 = None
    criterio_1_valor = None
    criterio_2 = None
    criterio_2_valor = None

if st.sidebar.button('Buscar Fondos Sustitutivos') and criterio_1 and criterio_2:
    criterios = {
        criterio_1: criterio_1_valor,
        criterio_2: criterio_2_valor
    }
    
    fondos_sustitutivos, criterios_aplicados = obtener_fondos_sustitutivos(isin, criterios)
    
    if fondos_sustitutivos is None:
        st.error(criterios_aplicados)
    else:
        st.subheader('Fondos Sustitutivos')
        st.write(fondos_sustitutivos)
        
        st.subheader('Comparación de Fondos')
        comparacion = pd.concat([fondo_seleccionado, fondos_sustitutivos]).set_index('isin')
        st.write(comparacion)
else:
    st.sidebar.info('Por favor, introduce un ISIN y selecciona los criterios para buscar fondos sustitutivos.')

st.markdown("""
---
*Desarrollado por [Luis Marti Avila]*
""")
