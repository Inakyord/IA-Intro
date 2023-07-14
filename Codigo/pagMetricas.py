#Bibliotecas utilizadas

import streamlit as st
import io
import requests
from PIL import Image
import pandas as pd                         # Para la manipulación y análisis de datos
import numpy as np                          # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datos
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance



# --------------- DECLARACIONES y CONFIGURACIONES ------------------


# Imagenes - objetos
matriz = Image.open("images/metricas/matriz.png")
eucli = Image.open("images/metricas/eucli.png")
cheb = Image.open("images/metricas/cheb.png")
man = Image.open("images/metricas/man.png")
mink = Image.open("images/metricas/mink.png")
fEucli = Image.open("images/metricas/formEucli.png")
fCheb = Image.open("images/metricas/formCheb.png")
fMan = Image.open("images/metricas/formMan.png")
fMink = Image.open("images/metricas/formMink.png")

# Config demo:
archivoDemo = "archivosPrueba/Hipoteca.csv"
cabecera = 0


# ----------------- FUNCIONES DESPLIEGUE-------------------------------


def iniciaMetricas():
	st.header("Métricas de distancia")
	st.text("""
		Las medidas de distancia no son muy útiles por si solas, pero son importantísimas
		para otros algoritmos. Por ejemplo en los algoritmos de clusterización
		(agrupación), el cálculo entre puntos se realiza con medidas de distancia, de aquí
		que sean importantes de comprender y saber los diferentes tipos que tenemos.
		En general las más importantes y las que se revisan en esta práctica son:
		euclidiana, chebyshev, manhattan y minkowski.
		"""
	)
	st.text("""
		El utilizar la medida adecuada nos llevará a obtener modelos más precisos
		y por ende mejores resultados. Las medidas de distancia nos permitirán identificar
		objetos con características similares o no similares. Pero la similaridad no es igual
		entre las diferentes medidas. Antes de meternos a explicar cada una de las
		diferentes distancias hay que comprender la forma en la que se muestran los
		resultados de distancias entre dos objetos y esto es mediante las matrices de
		distancias (cuadradas, pero sólo necesitamos un triángulo [inferior o superior]).
		"""
	)
	st.subheader("Matriz de distancias")
	st.image(matriz)
	st.write("---")
	st.header("Métricas:")
	st.write("---")
	st.subheader("Distancia Euclidiana")
	e1,e2 = st.columns((1,2))
	with e1:
		st.image(eucli)
	with e2:
		st.write("Longitud recta entre dos objetos (teoriema de Pitágora).")
		st.image(fEucli)
	st.write("---")
	st.subheader("Distancia de Chebyshev")
	b1,b2 = st.columns((2,1))
	with b1:
		st.write("Valor máximo absoluto entre las coordenadas de dos puntos (horizontal o vertical).")
		st.image(fCheb)
	with b2:
		st.image(cheb)
	st.write("---")
	st.subheader("Distancia de Manhattan")
	h1,h2 = st.columns((1,2))
	with h1:
		st.image(man)
	with h2:
		st.write("Distancia geométrica, en forma de cuadrícula, entre dos objetos (geometría del taxista).")
		st.image(fMan)
	st.write("---")
	st.subheader("Distancia de Minkowski")
	k1,k2 = st.columns((2,1))
	with k1:
		st.write("Distancia entre dos objetos suavizada y no recta.")
		st.image(fMink)
	with k2:
		st.image(mink)
	st.write("---")

	# Opción usuario
	with st.container():
		st.text("")
		st.subheader("Implementación algoritmo")
		st.text("Seleccione una opción:")
		st.text("")
		izq1,izq2,izq3, der1,der2,der3 = st.columns(6)
		with izq2:
			ejemplo = st.button("Ejemplo")
		with der2:
			propio = st.button("Subir datos")
		st.write("---")
		if 'ejemplo' not in st.session_state:
			st.session_state['ejemplo'] = ejemplo
		if 'propio' not in st.session_state:
			st.session_state.propio = propio
		if(ejemplo or propio):
			st.session_state.ejemplo = ejemplo
			st.session_state.propio = propio
		if(st.session_state.ejemplo):
			if 'tecnico' in st.session_state:
				del st.session_state['tecnico']
			if 'normal' in st.session_state:
				del st.session_state['normal']
			demo()
		elif(st.session_state.propio):
			propioArchivo()


def demo():
	st.header("Demo")
	st.text("En este demo se puede ver la funcionalidad del algoritmo de clustering jerárquico ascendente.")
	# Lectura archivo
	Datos = pd.read_csv(archivoDemo, header=cabecera)
	st.subheader("Los primeros 5 renglones son:")
	st.write(Datos.head(5))
	st.subheader("Información datos:")
	buffer = io.StringIO()
	Datos.info(buf=buffer)
	s = buffer.getvalue()
	st.text(s)
	st.subheader("Selección de métrica de distancia:")
	metrica = st.radio(
		"¿Con qué métrica desea realizar la medición?",
		('Euclidiana', 'Chebyshev', 'Manhattan', 'Minkowski'))
	if 'metrica' not in st.session_state:
		st.session_state['metrica'] = metrica
	if(metrica == 'Euclidiana'):
		st.subheader("Matriz de distancias Euclidianas")
		st.write(pd.DataFrame(cdist(Datos, Datos, metric='euclidean')))
	elif (metrica == 'Chebyshev'):
		st.subheader("Matriz de distancias Chebyshev")
		st.write(pd.DataFrame(cdist(Datos, Datos, metric='chebyshev')))
	elif (metrica == 'Manhattan'):
		st.subheader("Matriz de distancias Manhattan")
		st.write(pd.DataFrame(cdist(Datos, Datos, metric='cityblock')))
	elif (metrica == 'Minkowski'):
		st.subheader("Matriz de distancias Minkowski")
		st.write(pd.DataFrame(cdist(Datos, Datos, metric='minkowski', p=1.5)))
	maximo = len(Datos.index)
	objeto1 = st.number_input('Número del primer renglón a comparar',min_value=1,max_value=maximo)
	objeto2 = st.number_input('Número del segundo renglón a comparar',min_value=1,max_value=maximo)
	if 'objeto1' not in st.session_state:
		st.session_state['objeto1'] = objeto1
	if 'objeto2' not in st.session_state:
		st.session_state['objeto2'] = objeto2
	Dato1 = Datos.iloc[objeto1]
	Dato2 = Datos.iloc[objeto2]
	if(metrica == 'Euclidiana'):
		st.subheader("Distancia Euclidiana entre los datos:")
		st.write(distance.euclidean(Dato1,Dato2))
	elif (metrica == 'Chebyshev'):
		st.subheader("Distancia de Chebyshev entre los datos:")
		st.write(distance.chebyshev(Dato1,Dato2))
	elif (metrica == 'Manhattan'):
		st.subheader("Distancia Manhattan entre los datos:")
		st.write(distance.cityblock(Dato1,Dato2))
	elif (metrica == 'Minkowski'):
		st.subheader("Distancia Minkowski entre los datos:")
		st.write(distance.minkowski(Dato1,Dato2, p=1.5))

	

def propioArchivo():
	st.header("Propio")
	uploaded_file = st.file_uploader("Choose .csv file", type='csv')
	if uploaded_file is not None:
		st.write("filename: ",uploaded_file.name)
		Datos = pd.read_csv(uploaded_file, header=cabecera)
		st.subheader("Los primeros 5 renglones son:")
		st.write(Datos.head(5))
		st.subheader("Información datos:")
		buffer = io.StringIO()
		Datos.info(buf=buffer)
		s = buffer.getvalue()
		st.text(s)
		st.subheader("Selección de métrica de distancia:")
		metrica = st.radio(
			"¿Con qué métrica desea realizar la medición?",
			('Euclidiana', 'Chebyshev', 'Manhattan', 'Minkowski'))
		if 'metrica' not in st.session_state:
			st.session_state['metrica'] = metrica
		if(metrica == 'Euclidiana'):
			st.subheader("Matriz de distancias Euclidianas")
			st.write(pd.DataFrame(cdist(Datos, Datos, metric='euclidean')))
		elif (metrica == 'Chebyshev'):
			st.subheader("Matriz de distancias Chebyshev")
			st.write(pd.DataFrame(cdist(Datos, Datos, metric='chebyshev')))
		elif (metrica == 'Manhattan'):
			st.subheader("Matriz de distancias Manhattan")
			st.write(pd.DataFrame(cdist(Datos, Datos, metric='cityblock')))
		elif (metrica == 'Minkowski'):
			st.subheader("Matriz de distancias Minkowski")
			st.write(pd.DataFrame(cdist(Datos, Datos, metric='minkowski', p=1.5)))
		maximo = len(Datos.index)
		objeto1 = st.number_input('Número del primer renglón a comparar',min_value=1,max_value=maximo)
		objeto2 = st.number_input('Número del segundo renglón a comparar',min_value=1,max_value=maximo)
		if 'objeto1' not in st.session_state:
			st.session_state['objeto1'] = objeto1
		if 'objeto2' not in st.session_state:
			st.session_state['objeto2'] = objeto2
		Dato1 = Datos.iloc[objeto1]
		Dato2 = Datos.iloc[objeto2]
		if(metrica == 'Euclidiana'):
			st.subheader("Distancia Euclidiana entre los datos:")
			st.write(distance.euclidean(Dato1,Dato2))
		elif (metrica == 'Chebyshev'):
			st.subheader("Distancia de Chebyshev entre los datos:")
			st.write(distance.chebyshev(Dato1,Dato2))
		elif (metrica == 'Manhattan'):
			st.subheader("Distancia Manhattan entre los datos:")
			st.write(distance.cityblock(Dato1,Dato2))
		elif (metrica == 'Minkowski'):
			st.subheader("Distancia Minkowski entre los datos:")
			st.write(distance.minkowski(Dato1,Dato2, p=1.5))
		
