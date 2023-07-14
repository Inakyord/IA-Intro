#Bibliotecas utilizadas

import streamlit as st
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIL import Image
from generadorPDF import *




# --------------- DECLARACIONES y CONFIGURACIONES ------------------

# Imagenes desplegadas:
jera = Image.open("images/ascend/imgJera.png")
ascend = Image.open("images/ascend/imgAscend.png")
cluster = Image.open("images/clustering.png")
centroide = Image.open("images/centroide.png")

# Config demo:
archivoDemo = "archivosPrueba/Hipoteca.csv"
cabecera = 0



# ----------------- FUNCIONES DESPLIEGUE-------------------------------

def iniciaAscend():
	st.header("Clustering Jerárquico")
	
	# Expliación clustering
	with st.container():
		st.write("---")
		st.subheader("Clustering")
		col1,col2=st.columns(2)
		with col1:
			st.write("""
				La IA aplicada en el análisis clústeres consiste en la segmentación y delimitación de grupos
				de objetos (elementos), que son unidos por características comunes que éstos comparten
				(aprendizaje no supervisado).
				El objetivo es dividir una población heterogénea de elementos en un número de grupos
				naturales (regiones o segmentos homogéneos), de acuerdo a sus similitudes.
				"""
			)
		with col2:
			st.image(cluster)
		st.write("""
				Para hacer clustering es necesario poder medir la distancia (de las
				características) entre los diferentes objetos de un grupo. Para ello se utilizan
				principalmente las siguientes 4 distancias: euclidiana, chebyshev, manhattan y minkowski.
				Cuando se trabaja con clustering, dado que son
				algoritmos basados en distancias, es fundamental
				estandarizar los datos para que cada una de las
				variables contribuyan por igual en el análisis.
				"""
			)
		col3,col4=st.columns(2)
		with col3:
			st.write("""
				El centroide es el punto que ocupa la posición media en un cluster.
				La ubicación del centroide se calcula de manera iterativa. El centroide
				nos muestra alrededor de cuál valor está formado el grupo.
				"""
			)
		with col4:
			st.image(centroide)
		st.write("---")
		st.write("---")
	
	# Explicación jerárquico
	with st.container():
		st.header("Clustering jerárquico")
		i,d = st.columns((2,1))
		with i:
			st.write("""
				El algoritmo de Clustering Jerárquico es un algoritmo no supervisado que permite
				la segmentación y delimitación de grupos de objetos, que son unidos por
				características comunes que éstos comparten. Lo que se intenta con esto es el
				dividir una población aparentemente heterogénea en un número de regiones o
				segmentos homogéneos basándonos en sus similitudes. Hoy los algoritmos de
				clustering tienen grandes aplicaciones en áreas como el marketing, pólizas de
				seguros, urbanismo, biología, astrofísica, meteorología, etc.
				"""
			)
			st.write("""
				El Clustering Jerárquico consiste de primero medir las similitudes entre los
				diferentes vectores mediante las métricas de distancia, después ir agrupando los
				elementos más cercanos hasta cumplir con el criterio de una cantidad adecuada
				de grupos. Con esto ya tendríamos nuestros grupos, pero hay que interpretarlos
				de la mejor manera posible para definir el tipo de objetos que forman parte de él.
				Categorizar el grupo por decirlo así, de otro modo es inútil la información.
				"""
			)
		with d:
			st.image(jera)
	
	# Explicación ascendente
	with st.container():
		st.subheader("Algoritmo ascendente")
		st.write("""
			Consiste en agrupar en cada iteración aquellos 2 elementos más cercanos (clúster) -los de
			menor distancia-. De esta manera se va construyendo una estructura en forma de árbol. 
			El proceso concluye cuando se forma un único clúster (grupo).
			"""
		)
		st.write("""
			Dicho de otra forma y más explicado primero se calcula la matriz de distancia a partir de nuestra
			matriz de vectores ya estandarizada. Al inicio cada elemento es un clúster propio, pero de manera 
			iterativa se van combinando cada vez los dos clústers más cercanos entre ellos. Después de la 
			combinación se calcula el centroide del nuevo grupo y con él se recalculan las distancias de la
			matriz de similitudes. Esto se repite hasta que sólo quede un clúster.
			"""
		)
		iz,ce,de = st.columns((1,2,1))
		with ce:
			st.image(ascend)
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
	st.subheader("Agrupados por variable:")
	st.write(Datos.groupby('comprar').size())
	st.subheader("Gráficas de correlación:")
	with st.spinner('Armando gráfico...'):
		figura=sns.pairplot(Datos, hue='comprar')
		st.pyplot(figura)
	st.success("Listo!")
	st.subheader("Matriz de correlación:")
	CorrDatos = Datos.corr()
	st.write(CorrDatos)
	st.subheader("Mapa de calor de correlaciones:")
	figura3 = plt.figure(figsize=(14,7))
	MatrizInf = np.triu(CorrDatos)
	sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
	st.pyplot(figura3)
	MatrizHipoteca = np.array(Datos[['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
	st.subheader("Matriz de datos reducida dimensionalidad:")
	st.write(MatrizHipoteca)
	estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
	MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)   # Se calculan la media y desviación y se escalan los datos
	escalar = MinMaxScaler()                                      # Se instancia el objeto MinMaxScaler
	MEscalada = escalar.fit_transform(MatrizHipoteca)             # Se normalizan los datos entre 0 y 1
	figura4 = plt.figure(figsize=(17,12))
	plt.title("Casos de hipoteca, Matriz estandarizada, metrica euclidiana")
	plt.xlabel('Datos')
	plt.ylabel('Distancia')
	Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
	st.subheader("Dendrograma")
	with st.spinner('Armando gráfico...'):
		st.pyplot(figura4)
	st.success("Listo!")
	#Se crean las etiquetas de los elementos en los clústeres
	st.subheader('Número de clusters elegido: 5')
	MJerarquico = AgglomerativeClustering(n_clusters=5, linkage='complete', affinity='euclidean')
	MJerarquico.fit_predict(MEstandarizada)
	Datos = Datos.drop(columns=['comprar'])
	Datos['clusterH'] = MJerarquico.labels_
	st.subheader("Datos con su etiqueta del cluster jerárquico:")
	st.write(Datos) 
	#Cantidad de elementos en los clusters
	Datos.groupby(['clusterH'])['clusterH'].count()
	CentroidesH = Datos.groupby('clusterH').mean()
	st.subheader("Centroides de cada grupo:")
	st.write(CentroidesH)
	

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
			st.subheader("Gráficas de correlación:")
			with st.spinner('Armando mapa de calor...'):
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
			MatrizHipoteca=np.array(Datos)
			st.subheader("Matriz de datos reducida dimensionalidad:")
			st.write(MatrizHipoteca)
			estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
			MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)   # Se calculan la media y desviación y se escalan los datos        # Se normalizan los datos entre 0 y 1
			if st.session_state.tecnico:
				metrica = st.radio(
					"¿Con qué métrica desea realizar la medición?",
					('Euclidiana', 'Chebyshev', 'Manhattan', 'Minkowski'))
				if 'metrica' not in st.session_state:
					st.session_state['metrica'] = metrica
				if metrica == 'Euclidiana':
					medida = 'euclidean'
				elif metrica == 'Chebyshev':
					medida = 'chebyshev'
				elif metrica == 'Manhattan':
					medida = 'cityblock'
				elif metrica == 'Minkowski':
					medida = 'minkowski'
				st.write("Métrica elegida: "+medida)
			else:
				medida = 'euclidean'
				st.write("Métrica utilizada: "+medida)
			figura4 = plt.figure(figsize=(17,12))
			plt.title("Matriz estandarizada, metrica "+medida)
			plt.xlabel('Datos')
			plt.ylabel('Distancia')
			Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric=medida))
			st.subheader("Dendrograma")
			with st.spinner('Armando gráfico...'):
				st.pyplot(figura4)
			st.success("Listo!")
			#Se crean las etiquetas de los elementos en los clústeres
			n=st.number_input('Número de cluster: ',min_value=2,value=2)
			MJerarquico = AgglomerativeClustering(n_clusters=n, linkage='complete', affinity=medida)
			MJerarquico.fit_predict(MEstandarizada)
			Datos['clusterH'] = MJerarquico.labels_
			st.subheader("Datos con su etiqueta del cluster jerárquico:")
			st.write(Datos) 
			#Cantidad de elementos en los clusters
			Datos.groupby(['clusterH'])['clusterH'].count()
			CentroidesH = Datos.groupby('clusterH').mean()
			st.subheader("Centroides de cada grupo:")
			st.write(CentroidesH)
			reglas = CentroidesH.to_string()


			#Descripción de resultados por técnico
			if st.session_state.tecnico:
			  text = st.text_area("Escriba las interpretaciones:")
			  listo = st.button('Listo')
			  if listo:
			    html = obtener_link("Clustering Jerárquico",reglas,text,"ClusteringH",True)
			    st.markdown(html, unsafe_allow_html=True)


