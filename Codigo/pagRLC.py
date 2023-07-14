#Bibliotecas utilizadas

import streamlit as st
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from PIL import Image
#Se importan las bibliotecas necesarias para generar el modelo de regresión logística
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from generadorPDF import *


# --------------- DECLARACIONES y CONFIGURACIONES ------------------

# Imagenes desplegadas:
deco = Image.open("images/rlc/imgRLC.png")
reglas = Image.open("images/rlc/imgClasif.png")

# Config demo:
archivoDemo = "archivosPrueba/WDBCOriginal.csv"
cabecera = 0



# ----------------- FUNCIONES DESPLIEGUE-------------------------------

def iniciaRLC():
	st.header("Regresión Logística")
	i,d = st.columns((1,2))
	with i:
		st.image(deco)
	with d:
		st.write("""
			La regresión logística es otro tipo de algoritmo de aprendizaje supervisado,
			el cual tiene como objetivo el predecir valores binarios que clasifican
			nuestros datos. Este algoritmo consiste de una transformación a la regresión
			lineal.
			"""
		)
		st.write("""
			La manera en que funciona es utilizando una función sigmoide de la forma:
			Y=1/(1+e^-x), esta función nos genera probabilidades del 0 al 1. Para ajustarla
			a nuestro modelo lo que se hace es reemplazar la X, por una regresión lineal.
			Al final se clasificara el valor en la clase cuyo procesamiento por la función
			sigmoide (modelo) nos de la probabilidad más alta.
			"""
		)
	st.subheader("Clasificación")
	st.write("""
		Dentro del aprendizaje supervisado podemos dividir los algoritmos en dos 
		grandes categorías según su objetivo y resultado de procesamiento. Los algoritmos se 
		construyen con un conjunto de datos de entrenamiento y se validan con un conjunto de 
		datos de validación. Por un lado, 
		tenemos a la clasificación que predice etiquetas de dos clases 
		(binario) o de más de dos clases (multiclases). Estas etiquetas pueden ser tanto de 
		tipo discretas como nominales. Si la precisión es aceptable su utiliza el modelo para 
		clasificar nuevos valores que no se conocen.
		"""
	)
	iz,ce,de = st.columns((1,2,1))
	with ce:
		st.image(reglas)
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
	st.text("En este demo se puede ver la funcionalidad del algoritmo de clustering particional k-means.")
	# Lectura archivo
	Datos = pd.read_csv(archivoDemo, header=cabecera)
	st.subheader("Los primeros 5 renglones son:")
	st.write(Datos.head(5))
	st.subheader("Información datos:")
	buffer = io.StringIO()
	Datos.info(buf=buffer)
	s = buffer.getvalue()
	st.text(s)
	st.subheader("Descripción de los datos: ")
	st.write(Datos.describe())
	st.subheader("Matriz de correlación:")
	CorrDatos = Datos.corr()
	st.write(CorrDatos)
	st.subheader("Mapa de calor de correlaciones:")
	figura3 = plt.figure(figsize=(14,7))
	MatrizInf = np.triu(CorrDatos)
	sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
	st.pyplot(figura3)
	st.subheader("Variables seleccionadas: ")
	cadena = ("1) Textura [Posición 3]\n"+
    	"2) Area [Posición 5]\n"+
    	"3) Smoothness [Posición 6]\n"+
    	"4) Compactness [Posición 7]\n"+
    	"5) Symmetry [Posición 10]\n"+
    	"6) FractalDimension [Posición 11]\n"
    	)
	st.write(cadena)
	st.subheader("Clasificación datos:")
	st.write(Datos.groupby('Diagnosis').size())
	X = np.array(Datos[['Texture',
                      'Perimeter',
                      'Smoothness',	
                      'Compactness',	
                      'Symmetry',	
                      'FractalDimension']])
	Y = np.array(Datos[['Diagnosis']])
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 1234,
                                                                                shuffle = True)
	Clasificacion = linear_model.LogisticRegression()
	Clasificacion.fit(X_train, Y_train)                 #Se entrena el modelo
	#Se genera la probabilidad
	Probabilidad = Clasificacion.predict_proba(X_validation)
	Predicciones = Clasificacion.predict(X_validation)
	#Validación modelo
	st.subheader("Validación del modelo")
	Y_Clasificacion = Clasificacion.predict(X_validation)
	Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
	                                   Y_Clasificacion, 
	                                   rownames=['Real'], 
	                                   colnames=['Clasificación'])
	st.write(Matriz_Clasificacion)
	#Reporte de la clasificación
	st.subheader("Modelo:")
	ts1 = np.array2string(Clasificacion.coef_, precision=6, separator=',')
	ts2 = np.array2string(Clasificacion.intercept_, precision=6, separator=',')
	ts3 = str(Clasificacion.score(X_validation, Y_validation))
	ts4 = classification_report(Y_validation, Y_Clasificacion)
	ts5 = Matriz_Clasificacion.to_string()
	reglas = ("\nECUACION:\n\n"+"Coeficientes: "+ ts1 +"\n"+
		"Intercepto: "+ ts2 +"\n\n\n\n"+
		"MATRIZ CLASIFICACION: \n\n"+ts5+"\n\n\n\n"+
		"MÉTRICAS: \n\n"+ ts4 + "\n\n"+
		"EXACTITUD: \n"+ ts3+"\n\n")
	st.text(reglas)
		


def propioArchivo():
	with st.container():
		st.text("")
		st.subheader("Tipo usuario")
		st.text("Seleccione una opción:")
		st.text("")
		izq1,izq2,izq3, der1,der2,der3 = st.columns(6)
		with izq2:
			tecnico = st.button("Técnico")
		with der2:
			normal = st.button("Normal")
		st.write("---")
		if 'tecnico' not in st.session_state:
			st.session_state['tecnico'] = tecnico
		if 'normal' not in st.session_state:
			st.session_state.normal = normal
		if(tecnico or normal):
			st.session_state.tecnico = tecnico
			st.session_state.normal = normal
	if(st.session_state.tecnico or st.session_state.normal):
		st.header("Análisis de archivo del usuario")
		st.text("En esta parte el usuario podrá ingresar su archivo para procesar.")
		# Lectura archivo
		st.subheader("Seleccione archivo:")
		uploaded_file = st.file_uploader("Escoga su archivo formato .csv", type='csv')
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
			st.subheader("Descripción de los datos: ")
			st.write(Datos.describe())
			st.subheader("Matriz de correlación:")
			CorrDatos = Datos.corr()
			st.write(CorrDatos)
			st.subheader("Mapa de calor de correlaciones:")
			figura3 = plt.figure(figsize=(14,7))
			MatrizInf = np.triu(CorrDatos)
			sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
			st.pyplot(figura3)
			columnas = Datos.columns.values.tolist()
			options = st.multiselect(
				'¿Qué columnas desea eliminar?',columnas
			)
			st.write('Se eliminarán: ',options)
			for i in options:
				del Datos[i]
			st.subheader("Matriz de datos reducida dimensionalidad:")
			st.write(Datos)
			columnas2 = Datos.columns.values.tolist()
			opcion = st.selectbox(
				'¿Cuál es la variable de clase?',columnas2
			)
			Y = np.array(Datos[[opcion]])
			X = np.array(Datos.drop(opcion,axis=1))
			X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
		                                                                                test_size = 0.2, 
		                                                                                random_state = 1234,
		                                                                                shuffle = True)
			Clasificacion = linear_model.LogisticRegression()
			Clasificacion.fit(X_train, Y_train)                 #Se entrena el modelo
			#Se genera la probabilidad
			Probabilidad = Clasificacion.predict_proba(X_validation)
			Predicciones = Clasificacion.predict(X_validation)
			#Validación modelo
			st.subheader("Validación del modelo")
			Y_Clasificacion = Clasificacion.predict(X_validation)
			Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
			                                   Y_Clasificacion, 
			                                   rownames=['Real'], 
			                                   colnames=['Clasificación'])
			st.write(Matriz_Clasificacion)
			#Reporte de la clasificación
			st.subheader("Modelo:")
			ts1 = np.array2string(Clasificacion.coef_, precision=6, separator=',')
			ts2 = np.array2string(Clasificacion.intercept_, precision=6, separator=',')
			ts3 = str(Clasificacion.score(X_validation, Y_validation))
			ts4 = classification_report(Y_validation, Y_Clasificacion)
			ts5 = Matriz_Clasificacion.to_string()
			reglas = ("\nECUACION:\n\n"+"Coeficientes: "+ ts1 +"\n"+
				"Intercepto: "+ ts2 +"\n\n\n\n"+
				"MATRIZ CLASIFICACION: \n\n"+ts5+"\n\n\n\n"+
				"MÉTRICAS: \n\n"+ ts4 + "\n\n"+
				"EXACTITUD: \n"+ ts3+"\n\n")
			st.text(reglas)

			#Descripción de resultados por técnico
			if st.session_state.tecnico:
			  text = st.text_area("Escriba las interpretaciones:")
			  listo = st.button('Listo')
			  if listo:
			    html = obtener_link("Regresión Logística",reglas,text,"RLog",True)
			    st.markdown(html, unsafe_allow_html=True)