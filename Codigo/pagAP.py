#Bibliotecas utilizadas

import streamlit as st
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from PIL import Image
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from generadorPDF import *



# --------------- DECLARACIONES y CONFIGURACIONES ------------------

# Imagenes desplegadas:
deco = Image.open("images/AP/apGrafica.png")
reglas = Image.open("images/rlmp/imgPronos.png")

# Config demo:
archivoDemo = "archivosPrueba/WDBCOriginal.csv"
cabecera = 0



# ----------------- FUNCIONES DESPLIEGUE-------------------------------

def iniciaAP():
	st.header("Árbol de decisión - Pronóstico")
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
			Los árboles de decisión también se pueden aplicar a problemas de regresión (pronóstico). 
			Pueden aprenden detalles finos de los datos de entrenamiento, y aprenden también del ruido.
			Si el objetivo es un valor continuo, entonces, para el nodo (m), los criterios comunes para 
			determinar las ubicaciones para futuras divisiones son el error cuadrático medio (MSE), y el error 
			absoluto medio (MAE).

			"""
		)
	st.subheader("Pronóstico")
	st.write("""
		Dentro del aprendizaje supervisado podemos dividir los algoritmos en dos 
		grandes categorías según su objetivo y resultado de procesamiento. Los algoritmos se 
		construyen con un conjunto de datos de entrenamiento y se validan con un conjunto de 
		datos de validación. Por un lado, tenemos el pronóstico que modela funciones de 
		valor continuo para predecir valores desconocidos o faltantes (interpolación o 
		extrapolación). Si la precisión es aceptable su utiliza el modelo para pronosticar
		nuevos valores que no se conocen.
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
	st.subheader("Gráfica del área por paciente:")
	with st.spinner('Armando gráfico...'):
		figura2 = plt.figure(figsize=(20, 5))
		plt.plot(Datos['IDNumber'], Datos['Area'], color='green', marker='o', label='Area')
		plt.xlabel('Paciente')
		plt.ylabel('Tamaño del tumor')
		plt.title('Pacientes con tumores cancerígenos')
		plt.grid(True)
		plt.legend()
		st.pyplot(figura2)
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
    	"6) FractalDimension [Posición 11]\n"+
    	"7) *Perimeter [Posición 4] - Para calcular el área del tumor -\n"
    	)
	st.write(cadena)
	X = np.array(Datos[['Texture',
                      'Perimeter',
                      'Smoothness',	
                      'Compactness',	
                      'Symmetry',	
                      'FractalDimension']])
	Y = np.array(Datos[['Area']])
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 1234, 
                                                                    shuffle = True)
	PronosticoAD = DecisionTreeRegressor(max_depth=8, min_samples_split=4, min_samples_leaf=2, random_state=0)
	PronosticoAD.fit(X_train, Y_train)
	#Se genera el pronóstico
	Y_Pronostico = PronosticoAD.predict(X_test)
	st.subheader("Obtención de los coeficientes, intercepto, error y Score")
	ts1 = PronosticoAD.criterion
	ts2 = np.array2string(mean_absolute_error(Y_test, Y_Pronostico), precision=6, separator=',')
	ts3 = np.array2string(mean_squared_error(Y_test, Y_Pronostico), precision=6, separator=',')
	ts4 = np.array2string(mean_squared_error(Y_test, Y_Pronostico, squared=False), precision=6, separator=',')
	ts5 = np.array2string(r2_score(Y_test, Y_Pronostico), precision=6, separator=',')
	reglas = ("\nDATOS DEL MODELO:\n\n"+
		"Criterio: "+ ts1 +"\n"+
		"MAE: "+ ts2 +"\n"+
		"MSE: "+ ts3 +"\n"+
		"RMSE: "+ ts4 +"\n"+
		"Score (Bondad de ajuste): "+ ts5 +"\n\n")
	Importancia = pd.DataFrame({'Variable': list(Datos[['Texture', 'Perimeter', 'Smoothness',	
                                            'Compactness', 'Symmetry', 'FractalDimension']]),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
	st.text(reglas)
	st.write("\nImportancia de las variables: \n")
	st.write(Importancia)
	reglas = (reglas + "\n\n\nIMPORTANCIA: "+"\n\n"+
		Importancia.to_string())
	st.subheader("Visualización:")
	figura4 = plt.figure(figsize=(20, 5))
	plt.plot(Y_test, color='green', marker='o', label='Y_test')
	plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
	plt.xlabel('Paciente')
	plt.ylabel('Tamaño del tumor')
	plt.title('Pacientes con tumores cancerígenos')
	plt.grid(True)
	plt.legend()
	st.pyplot(figura4)
	#imagenes arbol
	st.subheader("Árbol creado: ")
	with st.spinner("Calculando árbol ..."):
		figura5 = plt.figure(figsize=(16,16))  
		plot_tree(PronosticoAD, feature_names = ['Texture', 'Perimeter', 'Smoothness', 
	                                         'Compactness', 'Symmetry', 'FractalDimension'])
		st.pyplot(figura5)
	Reporte = export_text(PronosticoAD, feature_names = ['Texture', 'Perimeter', 'Smoothness', 
                                                     'Compactness', 'Symmetry', 'FractalDimension'])
	with st.expander("Reporte de reglas:"):
		st.text(Reporte)
	reglas = (reglas + Reporte)




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
			with st.spinner('Armando gráfico...'):
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
				'¿Cuál es la variable pronóstico?',columnas2
			)
			Y = np.array(Datos[[opcion]])
			X = np.array(Datos.drop(opcion,axis=1))
			X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
		                                                                    test_size = 0.2, 
		                                                                    random_state = 1234, 
		                                                                    shuffle = True)
			prof = st.number_input('Profundidad: ',min_value=2,value=8)
			inter = st.number_input('Mínimo para decisión: ',min_value=2,value=4)
			hoja = st.number_input('Mínimo para hoja: ',min_value=1,value=2)

			PronosticoAD = DecisionTreeRegressor(max_depth=prof, min_samples_split=inter, min_samples_leaf=hoja, random_state=0)
			PronosticoAD.fit(X_train, Y_train)
			#Se genera el pronóstico
			Y_Pronostico = PronosticoAD.predict(X_test)
			st.subheader("Modelo:")
			ts1 = PronosticoAD.criterion
			ts2 = np.array2string(mean_absolute_error(Y_test, Y_Pronostico), precision=6, separator=',')
			ts3 = np.array2string(mean_squared_error(Y_test, Y_Pronostico), precision=6, separator=',')
			ts4 = np.array2string(mean_squared_error(Y_test, Y_Pronostico, squared=False), precision=6, separator=',')
			ts5 = np.array2string(r2_score(Y_test, Y_Pronostico), precision=6, separator=',')
			reglas = ("\nDATOS DEL MODELO:\n\n"+
				"Criterio: "+ ts1 +"\n"+
				"MAE: "+ ts2 +"\n"+
				"MSE: "+ ts3 +"\n"+
				"RMSE: "+ ts4 +"\n"+
				"Score (Bondad de ajuste): "+ ts5 +"\n\n")
			columnas3 = (Datos.drop(opcion,axis=1)).columns.values.tolist()
			Importancia = pd.DataFrame({'Variable': columnas3,
		                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
			st.text(reglas)
			st.write("\nImportancia de las variables: \n")
			st.write(Importancia)
			reglas = (reglas + "\n\n\nIMPORTANCIA: "+"\n\n\n"+
				Importancia.to_string())
			st.subheader("Visualización:")
			figura4 = plt.figure(figsize=(20, 5))
			plt.plot(Y_test, color='green', marker='o', label='Y_test')
			plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
			plt.xlabel('Paciente')
			plt.ylabel('Tamaño del tumor')
			plt.title('Pacientes con tumores cancerígenos')
			plt.grid(True)
			plt.legend()
			st.pyplot(figura4)
			#imagenes arbol
			st.subheader("Árbol creado: ")
			with st.spinner("Calculando árbol ..."):
				figura5 = plt.figure(figsize=(16,16))  
				plot_tree(PronosticoAD, feature_names = columnas3)
				st.pyplot(figura5)
			Reporte = export_text(PronosticoAD, feature_names = columnas3)
			with st.expander("Reporte de reglas:"):
				st.text(Reporte)
			reglas = (reglas + "\n\n\n"+"REPORTE: \n\n"+Reporte)
			
			#Descripción de resultados por técnico
			if st.session_state.tecnico:
			  text = st.text_area("Escriba las interpretaciones:")
			  listo = st.button('Listo')
			  if listo:
			    html = obtener_link("Árbol de Decisión - Pronóstico",reglas,text,"ADP",True)
			    st.markdown(html, unsafe_allow_html=True)



