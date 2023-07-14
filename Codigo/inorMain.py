#Bibliotecas utilizadas

import streamlit as st
import streamlit.components.v1 as html
import requests
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import io
from PIL import Image
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid


# Archivos externos algoritmos y despliegues

from pagInicio import iniciaInicio
from pagMetricas import iniciaMetricas
from pagApriori import iniciaApriori
from pagAscend import iniciaAscend
from pagK import iniciaK
from pagRLMP import iniciaRLMP
from pagRLC import iniciaRLC
from pagAP import iniciaAP
from pagAC import iniciaAC
from pagBP import iniciaBP
from pagBC import iniciaBC
from pagCreador import iniciaCreador






# --------------- DECLARACIONES y CONFIGURACIONES ------------------



# Imagenes - objetos
inor_logo = Image.open("images/icono7.png")
inor_titulo = Image.open("images/iconoLargo.png")


# Configuración ventana navegador:
st.set_page_config(page_title="Inor Computation", page_icon=inor_logo,layout="wide")

# Uso de archivo CSS local para diseño
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# llamado función
local_css("style/style.css")






# ----------------- FUNCIONALIDAD -------------------------------



# Menú lateral
add_text = st.sidebar.subheader("Menú")
with st.sidebar:
    choose = option_menu("inor Computation", ["Inicio", "Métricas de Distancia","Reglas Asociación - Apriori", "Clustering Jerárquico - Ascendente", "Clustering Particional - K-means",
    										  "Regresión Lineal Múltiple - Pronóstico", "Regresión Logística - Clasificación", "Árbol decisión - Pronóstico","Árbol decisión - Clasificación",
    										  "Bosques aleatorios - Pronóstico","Bosques aleatorios - Clasificación","Creador"],
                         icons=['house', 'binoculars', 'rulers', 'diagram-3', 'bricks','graph-up','file-binary','tree','tree-fill','bootstrap','bootstrap-fill','person lines fill'],
                         menu_icon="cpu", default_index=0,
                         orientation='vertical',
                         styles={
        "container": {"padding": "5!important", "background-color": "#808080"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#5D92DA"},
    }
    )

if 'choose' not in st.session_state:
	st.session_state.choose= "Inicio"

def borrado():
	if st.session_state.choose != choose:
		for key in st.session_state.keys():
			if key != "choose":
				del st.session_state[key]
	st.session_state.choose=choose


# ---- Títulos para todas las páginas ----
with st.container():
	titulo, espacio, logo = st.columns((50,2,10))
	with titulo:
		st.write("")
		st.image(inor_titulo)
	with logo:
		#insert image
		st.image(inor_logo)
	st.write("---")


# ---- Flujo del menú ----
if choose=="Inicio":
	borrado()
	iniciaInicio()
elif choose=="Métricas de Distancia":
	borrado()
	iniciaMetricas()
elif choose=="Reglas Asociación - Apriori":
	borrado()
	iniciaApriori()
elif choose=="Clustering Jerárquico - Ascendente":
	borrado()
	iniciaAscend()
elif choose=="Clustering Particional - K-means":
	borrado()
	iniciaK()
elif choose=="Regresión Lineal Múltiple - Pronóstico":
	borrado()
	iniciaRLMP()
elif choose=="Regresión Logística - Clasificación":
	borrado()
	iniciaRLC()
elif choose=="Árbol decisión - Pronóstico":
	borrado()
	iniciaAP()
elif choose=="Árbol decisión - Clasificación":
	borrado()
	iniciaAC()
elif choose=="Bosques aleatorios - Pronóstico":
	borrado()
	iniciaBP()
elif choose=="Bosques aleatorios - Clasificación":
	borrado()
	iniciaBC()
elif choose=="Creador":
	borrado()
	iniciaCreador()
