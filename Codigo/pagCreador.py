#Bibliotecas utilizadas

import streamlit as st
import requests
from PIL import Image
from streamlit_lottie import st_lottie



# --------------- DECLARACIONES y CONFIGURACIONES ------------------


# Imagenes - objetos
foto = Image.open("images/creador/credencial.jpeg")


# Animaciones - definición método carga
def load_lottieurl(url):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()

# Animaciones - objetos
lottie_coding = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_eux2nl1l.json")






# ----------------- FUNCIONES DESPLIEGUE-------------------------------


def iniciaCreador():
	with st.container():
		st.header("Hola, soy Iñaky")
		izq,der = st.columns((1,2))
		with izq:
			st.image(foto)
		with der:
			st.write("""
				Mi nombre es Iñaky Ordiales, soy estudiante de la carrera de 
				Ingeniería en Computación de la Facultad de Ingeniería de la
				Universidad Nacional Autónoma de México. """
			)
			st.write("""
				Desde chico me han gustado las matemáticas e interesado la 
				teconología, por lo que cuando llegó el momento de decidir el
				camino para continuar mis estudios la ingeniería en computación
				fue una elección coherente. A lo largo de la carrera me he ido
				apasionando cada vez más con todo el complejo mundo de la 
				computación. """
			)
			st.write("""
				Además de la programación me gusta leer, jugar fútbol, salir a
				correr y estar con amigos. Creo que es importante tratar de 
				mantener un balance entre nuestras actividades académicas y 
				el resto de nuestra vida."""
			)

	with st.container():
		st.text("")
		st_lottie(lottie_coding, height=300, key="coding")

	with st.container():
		st.write("---")
		st.header("¡Contáctame!")
		st.write("##")

		# Documentation: https://formsubmit.co/
		contact_form = """
		<from action="https://formsubmit.co/inaky.ordiales@gmail.com" method="POST">
			<input type="hidden" name="_captcha" value="false">
			<input type="text" name="name" placeholder="Tu nombre" required>
			<input type="email" name="email" placeholder="Tu email" required>
			<textarea name="message" placeholder="Mensaje para enviar..." required></textarea>
			<button type="submit">Enviar</button>
		</form>
		"""
		left_column, right_column = st.columns(2)
		with left_column:
			st.markdown(contact_form, unsafe_allow_html=True)
		with right_column:
			st.empty()