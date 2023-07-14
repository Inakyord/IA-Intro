#Bibliotecas utilizadas

import streamlit as st
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIL import Image
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from generadorPDF import *



# --------------- DECLARACIONES y CONFIGURACIONES ------------------

# Imagenes desplegadas:
reglas = Image.open("images/kmeans/imgK.png")
deco = Image.open("images/kmeans/imgPart.png")
cluster = Image.open("images/clustering.png")
centroide = Image.open("images/centroide.png")

# Config demo:
archivoDemo = "archivosPrueba/Hipoteca.csv"
cabecera = 0




# ----------------- FUNCIONES DESPLIEGUE-------------------------------

def iniciaK():
	st.header("Clustering Particional")
	
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

	# Expliacación particional
	with st.container():
		st.header("Clustering Particional")
		i,d = st.columns((1,2))
		with i:
			st.image(deco)
		with d:
			st.write("""
				El Clustering Particional nos permite crear k clústeres a partir de un conjunto de
				datos, resolviendo problemas de optimización dado que minimiza la suma de
				distancias de cada elemento al centroide del clúster. Un centroide es el punto
				medio entre todos los objetos que forman parte de un mismo clúster.
				"""
			)

	# Expliación k-means
	with st.container():
		st.subheader("Algoritmo k-means")
		st.write("""
			En este
			algoritmo se establece previamente el número de clústeres que se esperan
			obtener k . Este valor es un entero arbitrario para el cual se ajustarán todos los
			datos del conjunto en ese número de grupos. La forma en que se realiza esto es
			seleccionando aleatoriamente k objetos del conjunto de datos y establecerlos
			como centroides iniciales. Para cada uno de los registros se le asignará el
			centroide más cercano (según la métrica de distancias) y con base en todos los
			objetos asignados a un centroide, se le calculará su media para mover el centroide
			a la media del clúster. Esto se repite indefinidamente en varias iteraciones hasta
			que se consiga finalmente que el centroide no se mueva y por lo tanto los
			elementos y distancias del clúster tampoco cambien.
			"""
		)
		st.write("""
			Para la selección de la mejor k posible, es decir para que las distancias sean
			mínimas pero el modelo siga siendo lo suficientemente general para poder trabajar
			con nuevos datos y que no esté sobre especializado, se utiliza normalmente el
			método del codo. Este consiste en seleccionar visualmente el punto menos suave
			de la curva que se obtiene al graficar las distancias SSE (suma de la distancia al
			cuadrado entre cada elemento de un clúster y su centroide) con el número k de
			clústeres utilizados. Este punto se considera adecuado para tener un buen modelo
			general, pero no necesariamente es el mejor número de clústeres. Es solo un
			método de aproximación no exacto.
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
	
	#Definición de k clusters para K-means
	#Se utiliza random_state para inicializar el generador interno de números aleatorios
	SSE = []
	for i in range(2, 12):
	    km = KMeans(n_clusters=i, random_state=0)
	    km.fit(MEscalada)
	    SSE.append(km.inertia_)
	#Se grafica SSE en función de k
	figura4 = plt.figure(figsize=(15, 11))
	plt.plot(range(2, 12), SSE, marker='o')
	plt.xlabel('Cantidad de clusters *k*')
	plt.ylabel('SSE')
	plt.title('Elbow Method')
	st.subheader("Método del codo:")
	st.pyplot(figura4)
	kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
	st.subheader("Método del codo calculado:")
	st.write("El valor del codo es: ",kl.elbow)
	plt.style.use('ggplot')
	kl.plot_knee()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot(kl.plot_knee())
	MParticional = KMeans(n_clusters=5, random_state=0).fit(MEscalada)
	MParticional.predict(MEscalada)
	Datos = Datos.drop(columns=['comprar'])
	Datos['clusterP'] = MParticional.labels_
	st.subheader("Datos con su etiqueta del cluster jerárquico:")
	st.write(Datos)
	#Cantidad de elementos en los clusters
	st.subheader("Número de elementos por cluster:")
	st.write(Datos.groupby(['clusterP'])['clusterP'].count())
	CentroidesP = Datos.groupby('clusterP').mean()
	st.subheader("Centroides de cada grupo:")
	st.write(CentroidesP) 
	# Gráfica de los elementos y los centros de los clusters
	st.subheader("Gráfica del clustering")
	plt.rcParams['figure.figsize'] = (10, 7)
	plt.style.use('ggplot')
	colores=['red', 'blue', 'green', 'yellow','orange']
	asignar=[]
	for row in MParticional.labels_:
	    asignar.append(colores[row])
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(MEstandarizada[:, 0], 
	           MEstandarizada[:, 1], 
	           MEstandarizada[:, 2], marker='o', c=asignar, s=60)
	ax.scatter(MParticional.cluster_centers_[:, 0], 
	           MParticional.cluster_centers_[:, 1], 
	           MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
	st.pyplot(fig)


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
		
			#Definición de k clusters para K-means
			#Se utiliza random_state para inicializar el generador interno de números aleatorios
			SSE = []
			for i in range(2, 12):
			    km = KMeans(n_clusters=i, random_state=0)
			    km.fit(MEstandarizada)
			    SSE.append(km.inertia_)
			#Se grafica SSE en función de k
			figura4 = plt.figure(figsize=(15, 11))
			plt.plot(range(2, 12), SSE, marker='o')
			plt.xlabel('Cantidad de clusters *k*')
			plt.ylabel('SSE')
			plt.title('Elbow Method')
			st.subheader("Método del codo:")
			st.pyplot(figura4)
			kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
			st.subheader("Método del codo calculado:")
			st.write("El valor del codo es: ",kl.elbow)
			plt.style.use('ggplot')
			kl.plot_knee()
			st.set_option('deprecation.showPyplotGlobalUse', False)
			st.pyplot(kl.plot_knee())
			MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
			MParticional.predict(MEstandarizada)
			Datos['clusterP'] = MParticional.labels_
			st.subheader("Datos con su etiqueta del cluster jerárquico:")
			st.write(Datos)
			#Cantidad de elementos en los clusters
			st.subheader("Número de elementos por cluster:")
			st.write(Datos.groupby(['clusterP'])['clusterP'].count())
			CentroidesP = Datos.groupby('clusterP').mean()
			st.subheader("Centroides de cada grupo:")
			st.write(CentroidesP) 
			# Gráfica de los elementos y los centros de los clusters
			st.subheader("Gráfica del clustering")
			plt.rcParams['figure.figsize'] = (10, 7)
			plt.style.use('ggplot')
			colores=['red', 'blue', 'green', 'yellow','orange','pink','black','gray','cyan']
			asignar=[]
			for row in MParticional.labels_:
			    asignar.append(colores[row])
			fig = plt.figure()
			ax = Axes3D(fig)
			ax.scatter(MEstandarizada[:, 0], 
			           MEstandarizada[:, 1], 
			           MEstandarizada[:, 2], marker='o', c=asignar, s=60)
			ax.scatter(MParticional.cluster_centers_[:, 0], 
			           MParticional.cluster_centers_[:, 1], 
			           MParticional.cluster_centers_[:, 2], marker='o', c=colores[0:kl.elbow], s=1000)
			st.pyplot(fig)
			reglas = "Número de elementos por cluster: \n\n"+(Datos.groupby(['clusterP'])['clusterP'].count()).to_string()
			reglas = reglas+"\n\nCentroides: \n\n"+CentroidesP.to_string()

			#Descripción de resultados por técnico
			if st.session_state.tecnico:
			  text = st.text_area("Escriba las interpretaciones:")
			  listo = st.button('Listo')
			  if listo:
			    html = obtener_link("Clustering Particional",reglas,text,"ClusteringP",True)
			    st.markdown(html, unsafe_allow_html=True)