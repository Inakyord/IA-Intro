#Bibliotecas utilizadas

import streamlit as st
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from generadorPDF import *



# --------------- DECLARACIONES y CONFIGURACIONES ------------------

# Imagenes desplegadas:
deco = Image.open("images/AC/acGrafica.png")
reglas = Image.open("images/BC/bosque.png")

# Config demo:
archivoDemo = "archivosPrueba/WDBCOriginal.csv"
cabecera = 0





# ----------------- FUNCIONES DESPLIEGUE-------------------------------

def iniciaBC():
	st.header("Bosque aleatorio - Clasificación")
	st.subheader("Árboles de clasificación")
	i,d = st.columns((1,2))
	with i:
		st.image(deco)
	with d:
		st.write("""
			Los árboles de decisión son uno de los algoritmos de 
			aprendizaje automático supervisado más utilizados en el 
			mundo real. Uno de sus mayores beneficios es el de 
			admitir valores tanto numéricos como nominales y nulos 
			para sus características por lo que tienen un gran rango de 
			aplicación dentro de la inteligencia artificial. Su objetivo es 
			el de construir una jerarquía eficiente y escalable que divide los datos en función 
			de determinadas condiciones. Para esto se utiliza la estrategia: divide y vencerás.
			En realidad es de la misma forma que nosotros tomamos nuestras decisiones día a día.
			"""
		)
		st.write("""
			En los árboles de decisión de clasificación se calcula la entropía para todas las clases 
			y atributos. Después se selecciona el mejor atributo basado en la ganancia de información 
			de cada variable. Y finalmente se itera hasta que todos los elementos sean clasificados.
			La validación y métricas de desempeño se obtienen a partir de una matriz de clasificación, 
			también conocida como matriz de confusión.
			"""
		)
	st.subheader("Bosques aleatorios")
	st.write("""
		Una forma de mejorar la generalización de los árboles de decisión es combinar 
		varios árboles, a esto se les conoce como bosques aleatorios. Su objetivo es 
		el de construir un conjunto (ensamble) de árboles de decisión combinados. Al 
		combinar lo que en realidad está pasando es que distintos árboles ven 
		distintas porciones de los datos. Por lo que, los bosques aleatorios son una 
		variación moderna, que agrupan varios árboles de decisión para producir un modelo 
		generalizado con el objetivo de reducir la tendencia al sobreajuste. Es similar a
		la forma en que ponderamos diferentes opiniones en nuestras vidas.
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
	Datos = Datos.replace({'M': 'Malignant', 'B': 'Benign'})
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
                                                                                random_state = 0,
                                                                                shuffle = True)
	Clasificacion = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, min_samples_leaf=2, random_state=0)
	Clasificacion.fit(X_train, Y_train)
	#Se etiquetan las clasificaciones
	Y_Clasificacion = Clasificacion.predict(X_validation)
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
	ts1 = Clasificacion.criterion
	ts3 = str(Clasificacion.score(X_validation, Y_validation))
	ts4 = classification_report(Y_validation, Y_Clasificacion)
	ts5 = Matriz_Clasificacion.to_string()
	reglas = ("\nDATOS DEL MODELO: \n\n"+
		"Criterio: "+ ts1 + "\n\n\n"
		"MATRIZ CLASIFICACION: \n\n"+ts5+"\n\n\n\n"+
		"MÉTRICAS: \n\n"+ ts4 + "\n\n"+
		"EXACTITUD: \n"+ ts3+"\n\n")
	st.text(reglas)
	Importancia = pd.DataFrame({'Variable': list(Datos[['Texture', 'Area', 'Smoothness', 
                                                     'Compactness', 'Symmetry', 'FractalDimension']]),
                            'Importancia': Clasificacion.feature_importances_}).sort_values('Importancia', ascending=False)
	st.write("\nImportancia de las variables:")
	st.write(Importancia)
	reglas = (reglas + "\n\n\nIMPORTANCIA: "+"\n\n"+
		Importancia.to_string())
	#imagenes arbol
	st.subheader("Selección del árbol 33 para mostrar, Estimador = 33")
	Estimador = Clasificacion.estimators_[33]
	st.subheader("Árbol creado: ")
	with st.spinner("Calculando árbol ..."):
		figura5 = plt.figure(figsize=(16,16))  
		plot_tree(Estimador, 
	          feature_names = ['Texture', 'Area', 'Smoothness', 
	                           'Compactness', 'Symmetry', 'FractalDimension'],
	          class_names = Y_Clasificacion)
		st.pyplot(figura5)
	Reporte = export_text(Estimador, 
                      feature_names = ['Texture', 'Area', 'Smoothness', 
                                       'Compactness', 'Symmetry', 'FractalDimension'])
	with st.expander("Reporte de reglas:"):
		st.text(Reporte)
	reglas = (reglas + "\n\n\n"+"REPORTE: \n\n"+Reporte)



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
		                                                                                random_state = 0,
		                                                                                shuffle = True)
			estim = st.number_input('Número de árboles estimadores: ',min_value=2, value=100)
			prof = st.number_input('Profundidad: ',min_value=2,value=8)
			inter = st.number_input('Mínimo para decisión: ',min_value=2,value=4)
			hoja = st.number_input('Mínimo para hoja: ',min_value=1,value=2)

			Clasificacion = RandomForestClassifier(n_estimators=estim, max_depth=prof, min_samples_split=inter, min_samples_leaf=hoja, random_state=0)
			Clasificacion.fit(X_train, Y_train)
			#Se etiquetan las clasificaciones
			Y_Clasificacion = Clasificacion.predict(X_validation)
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
			ts1 = Clasificacion.criterion
			ts3 = str(Clasificacion.score(X_validation, Y_validation))
			ts4 = classification_report(Y_validation, Y_Clasificacion)
			ts5 = Matriz_Clasificacion.to_string()
			reglas = ("\nDATOS DEL MODELO: \n\n"+
				"Criterio: "+ ts1 + "\n\n\n"
				"MATRIZ CLASIFICACION: \n\n"+ts5+"\n\n\n\n"+
				"MÉTRICAS: \n\n"+ ts4 + "\n\n"+
				"EXACTITUD: \n"+ ts3+"\n\n")
			st.text(reglas)
			columnas3 = (Datos.drop(opcion,axis=1)).columns.values.tolist()
			Importancia = pd.DataFrame({'Variable': columnas3,
		                            'Importancia': Clasificacion.feature_importances_}).sort_values('Importancia', ascending=False)
			st.write("\nImportancia de las variables:")
			st.write(Importancia)
			reglas = (reglas + "\n\n\nIMPORTANCIA: "+"\n\n"+
				Importancia.to_string())
			#imagenes arbol
			st.subheader("Selección del árbol n para mostrar")
			n = st.number_input('Número del árbol que mostrar: ',min_value=1,max_value=estim,value=1)
			Estimador = Clasificacion.estimators_[n]
			st.subheader("Árbol creado: ")
			with st.spinner("Calculando árbol ..."):
				figura5 = plt.figure(figsize=(16,16))  
				plot_tree(Estimador, 
			          feature_names = columnas3,
			          class_names = Y_Clasificacion)
				st.pyplot(figura5)
			Reporte = export_text(Estimador, 
		                      feature_names = columnas3)
			with st.expander("Reporte de reglas:"):
				st.text(Reporte)
			reglas = (reglas + "\n\n\n"+"REPORTE: \n\n"+Reporte)

			#Descripción de resultados por técnico
			if st.session_state.tecnico:
			  text = st.text_area("Escriba las interpretaciones:")
			  listo = st.button('Listo')
			  if listo:
			    html = obtener_link("Bosque aleatorio - Clasificación",reglas,text,"BAC",True)
			    st.markdown(html, unsafe_allow_html=True)

