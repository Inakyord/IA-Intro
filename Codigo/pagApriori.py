#Bibliotecas utilizadas

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori
from PIL import Image
from generadorPDF import *




# --------------- DECLARACIONES y CONFIGURACIONES ------------------

# Imagenes desplegadas:
reglas = Image.open("images/apriori/formApriori.png")
deco = Image.open("images/apriori/imgApriori.png")

# Config demo:
archivoDemo = "archivosPrueba/store_data.csv"
cabecera = None



# ----------------- FUNCIONES DESPLIEGUE-------------------------------

def iniciaApriori():
	# Expliación asociación
	with st.container():
		st.header("Reglas de asociación")
		i,d = st.columns((1,2))
		with i:
			st.image(deco)
		with d:
			st.write("""
				Uno de los tipos de algoritmos más utilizados en la actualidad son las 
				reglas de asociación. Éstos tienen su mayor aplicación en los sistemas 
				de recomendación, los cuales dependiendo de tus actividades	previas y 
				de las relaciones de ellas con las de otros usuarios, se da una serie de
				recomendaciones personalizadas para ti.
				"""
			)
			st.write("""
				En específico las reglas de asociación son parte del aprendizaje automático
				no supervisado y se basan en reglas para encontrar relaciones ocultas a simple
				vista para nosotros en los datos. Otro nombre que tiene es el de análisis de
				afinidad. Consiste en identificar patrones if – then que refieren a que si se
				realizó	algo es probable que se realice lo consiguiente. A la parte if se le 
				conoce como	antecedente y a la parte then como consecuente. Con esta misma 
				metodología es con la que debemos leer nuestras reglas resultantes. Visto desde
				un punto matemático, las reglas de asociación son una proporción probabilística 
				sobre la ocurrencia de eventos dentro de un conjunto de datos.
				"""
			)

	# Expliación a priori
	with st.container():
		st.subheader("Algoritmo a priori")
		st.write("""
			El nombre a priori viene del hecho en que se eliminan los elementos por su
			ocurrencia antes de saber en qué derivarían. Para el algoritmo a priori es
			necesario establecer tres parámetros:
			"""
		)
		st.markdown("<ul><li> <b>Soporte</b> (Cobertura): Indica cuan importante es una regla dentro del total de transacciones.</li><li><b>Confianza</b>: Indica qué tan fiable es una regla.</li><li><b>Lift</b> (Elevación, interés): Indica el nivel de relación (aumento de probabilidad) entre el antecedente y consecuente de la regla.</li></ul>",unsafe_allow_html=True)
		st.write("Los parametros se calculan de la sigiuente forma:")
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
	st.header("Demo - a priori.")
	st.text("En este demo se puede ver la funcionalidad del algoritmo apriori.")
	# Lectura archivo
	Transacciones = pd.read_csv(archivoDemo, header=cabecera)
	st.subheader("Los primeros 5 renglones son:")
	st.write(Transacciones.head(5))
	# Procesamiento
	Lista = pd.DataFrame(Transacciones.values.reshape(-1).tolist())
	Lista['Frecuencia']=1
	Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
	Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
	Lista = Lista.rename(columns={0 : 'Item'})
	st.subheader("Lista de items:")
	st.write(Lista)
	st.subheader("Gráfica de frecuencia:")
	with st.spinner('Armando gráfico...'):
		graficaRepeticion = plt.figure(figsize=(16,20), dpi=450)
		plt.ylabel('Item')
		plt.xlabel('Frecuencia')
		plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
		st.pyplot(graficaRepeticion)
	st.success('Listo!')
	# Aplicación algoritmo
	TransaccionesLista = Transacciones.stack().groupby(level=0).apply(list).tolist()
	soporteMin=0.05
	confianzaMin=0.15
	liftMin=1.01
	Reglas = apriori(TransaccionesLista, 
                   min_support=soporteMin, 
                   min_confidence=confianzaMin, 
                   min_lift=liftMin)
	# Resultados
	Resultados = list(Reglas)
	st.subheader("Se encontraron un total de "+str(len(Resultados))+" reglas:") #Total de reglas encontradas
	i=0
	with st.expander("Reporte de reglas:"):
		for item in Resultados:
		  #El primer índice de la lista
		  i+=1
		  Emparejar = item[0]
		  items = [x for x in Emparejar]
		  st.text("["+str(i)+"]")
		  st.text("Regla: " + str(item[0]))

		  #El segundo índice de la lista
		  st.text("Soporte: " + str(item[1]))

		  #El tercer índice de la lista
		  st.text("Confianza: " + str(item[2][0][2]))
		  st.text("Lift: " + str(item[2][0][3])) 
		  st.write("---")


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
			Transacciones = pd.read_csv(uploaded_file, header=cabecera)
			st.subheader("Los primeros 5 renglones son:")
			st.write(Transacciones.head(5))
			# Procesamiento
			Lista = pd.DataFrame(Transacciones.values.reshape(-1).tolist())
			Lista['Frecuencia']=1
			Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
			Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
			Lista = Lista.rename(columns={0 : 'Item'})
			st.subheader("Lista de items:")
			st.write(Lista)
			st.subheader("Gráfica de frecuencia:")
			with st.spinner('Armando gráfico...'):
				graficaRepeticion = plt.figure(figsize=(16,20), dpi=450)
				plt.ylabel('Item')
				plt.xlabel('Frecuencia')
				plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
				#plt.savefig('images/temp/temp.png', bbox_inches='tight')
				st.pyplot(graficaRepeticion)
			st.success('Listo!')
			# Aplicación algoritmo
			TransaccionesLista = Transacciones.stack().groupby(level=0).apply(list).tolist()
			soporteMin=st.number_input('Soporte mínimo: ',min_value=0.05,value=0.10)
			confianzaMin=st.number_input('Confianza mínima: ',min_value=0.05,value=0.15)
			liftMin=st.number_input('Levantamiento mínimo: ',min_value=1.1,value=1.5)
			Reglas = apriori(TransaccionesLista, 
		                   min_support=soporteMin, 
		                   min_confidence=confianzaMin, 
		                   min_lift=liftMin)
			# Resultados
			Resultados = list(Reglas)
			st.subheader("Se encontraron un total de "+str(len(Resultados))+" reglas:") #Total de reglas encontradas
			i=0
			reglas = ""
			with st.expander("Reporte de reglas:"):
				for item in Resultados:
				  #El primer índice de la lista
				  i+=1
				  Emparejar = item[0]
				  items = [x for x in Emparejar]
				  st.text("["+str(i)+"]")
				  st.text("Regla: " + str(item[0]))

				  #El segundo índice de la lista
				  st.text("Soporte: " + str(item[1]))

				  #El tercer índice de la lista
				  st.text("Confianza: " + str(item[2][0][2]))
				  st.text("Lift: " + str(item[2][0][3])) 
				  st.write("---")
				  reglas = reglas + "["+str(i)+"]\n"+"Regla: " + str(item[0])+"\n"+"Soporte: " + str(item[1])+"\n"+"Confianza: " + str(item[2][0][2])+"\n"+"Lift: " + str(item[2][0][3])+"\n"+"----------------------\n"

			#Descripción de resultados por técnico
			if st.session_state.tecnico:
			  text = st.text_area("Escriba las interpretaciones:")
			  listo = st.button('Listo')
			  if listo:
			    html = obtener_link("Reglas Apriori",reglas,text,"Apriori",True)
			    st.markdown(html, unsafe_allow_html=True)
