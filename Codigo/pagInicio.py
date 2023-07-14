#Bibliotecas utilizadas

import streamlit as st
import requests
from PIL import Image
from streamlit_lottie import st_lottie



# --------------- DECLARACIONES y CONFIGURACIONES ------------------


# Imagenes - objetos
ia = Image.open("images/inicio/ia.png")
ml = Image.open("images/inicio/ml.png")
inor_logo = Image.open("images/icono7.png")
inor_titulo = Image.open("images/iconoLargo.png")


# Animaciones - definición método carga
def load_lottieurl(url):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()

# Animaciones - objetos
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")


# ----------------- FUNCIONES DESPLIEGUE-------------------------------


def iniciaInicio():
	with st.container():
		st.header("Inteligencia Artificial")
		left_column, right_column = st.columns((1,2))
		with right_column:
			st.write("""
				La inteligencia artificial nace de la idea de
				que todos los aspectos del aprendizaje y a su vez de la inteligencia humana se pueden replicar
				con precisión mediante una computadora. Se cree que si logramos obtener los elementos fı́sicos
				tecnológicos y un entendimiento total del funcionamiento y comportamiento del cerebro,
				podremos construir máquinas con ”inteligencia” propia. Actualmente nos encontramos desarrollando
				inteligencia artificial estrecha la cual carece de la capacidad de entender el contexto y 
				está enfocada en tareas específicas y no generales. Aquí mostramos una tabla con las 
				clasificaciones de la intelifencia artificial.
				"""
			)
			st.image(ia)
		with left_column:
			st_lottie(lottie_coding, height=300, key="coding")
			st.write("---")
			st.subheader("Inteligencia")
			st.write("""
				La inteligencia es la capacidad de decidir bien, lo que implica utilizar muchas áreas del conocimiento
				para en base al contexto poder realizar una acción que nos acerque al objetivo final eficientemente. 
				Dentro de las habilidades necesarias están: razonamiento, planificación y resolución de problemas,
				comprensión de ideas complejas, aprendizaje rápido y continuo en base a la experiencia.
				"""
			)
		st.write("---")
	with st.container():
		st.header("Aprendizaje automático (Machine Learning)")
		st.text("""
			Machine learning significa aprender de datos sin ser explı́citamente programado para
			ello, siendo la muestra de datos inclusive más importante que los mismos algoritmos que los
			procesan. El desarrollo de esta automatización ha aumentado inclusive más las expectativas
			de la creación de máquinas inteligentes. Sin embargo, esto no es como tal una inteligencia y
			ni siquiera apunta hacia la dirección de generar máquinas inteligentes por si solas. A 
			continuación se muestra una tabla con ĺas categorías y algunos de los algoritmos más
			conocidos del aprendizaje automático:
			"""
		)
		st.image(ml)
		st.write("---")
		st.write("---")
	with st.container():
		st.header("inor Computation")
		i,c,d = st.columns((2,3,2))
		with c:
			st.image(inor_logo)
		st.subheader("¿Quiénes somos?")
		st.markdown("""<tt>inor Computation</tt> es una aplicación web desarrollada en el lenguaje de 
			programación Python3 y desplegada mediante la utilización de la biblioteca de python Streamlit.
			""", unsafe_allow_html=True)
		st.write("""Se realizó como el proyecto final para la asignatura de Inteligencia Artificial de la
			carrera de Ingeniería en Computación en la Facultad de Ingeniería de la UNAM, por Iñaky Ordiales.
			En la aplicación se implementaron varios algoritmos de aprendizaje automático para poder
			ser utilizados con los datos que ingrese el usuario. Además, el usuario podrá elegir los diferentes
			parámetros a utilizar en cada algoritmo, dando una mayor flexibilidad en la obtención de resultados.
			"""
		)
		st.write("""Algo extra ...
			"""
		)
		st.subheader("Modo de uso")
		st.write("""
			Por default al ingresar a la aplicación a través del navegador se debe desplegar el menú
			lateral donde aparecerán las opciones de la aplicación: inicio, métricas, ..., creador.
			Seleccionando cada una de ellas se le mostrará la información correspondiente a dicha
			sección. En el caso de que el menú no se encuentre abierto, en la parte superior izquierda 
			dentro de la ventana del navegador aparece una flecha de la forma '>' con la que podrá
			abrir el menú. En caso de quererlo achicar, el menú tiene una cruz del lado superior derecho.
			""")
		st.write("""
			Dependiendo de la selección de contenido realizada se mostrará en pantalla información acerca 
			del tema o algoritmo. En el caso de ser un algoritmo al recorrer hacia abajo la página, se podrá
			seleccionar entre las opciones de ver un ejemplo con datos precargados o subir sus propios datos
			para analizarlos. En cualquiera de los dos caminos se le darán al usuario parámetros y opciones
			a elegir para realizar el correcto procesamiento. Finalmente se mostrará en pantalla los
			resultados.
			""")
		st.text("")
		st.markdown("""<h5> Esperemos sea de su agrado    :) </h5>
			""", unsafe_allow_html=True)
		st.image(inor_titulo)