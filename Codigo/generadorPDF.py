import streamlit as st
from fpdf import FPDF
import base64
from PIL import Image
import os


# --------------- DECLARACIONES y CONFIGURACIONES ------------------

inor_logo = "images/noicono7.png"
inor_titulo = "images/iconoLargo.png"
fondo = "images/fondo_azul.png"


def create_download_link(val, filename):
  b64 = base64.b64encode(val)  # val looks like b'...'
  return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def obtener_link(titulo, texto, comentarios, nombre, imagen):
  pdf = FPDF()
  pdf.add_page()
  pdf.image(fondo,x=0,y=0,w=210,h=297,type='',link='')
  pdf.image(inor_titulo,x=15,y=15,h=20)
  pdf.image(inor_logo,x=160,y=10,w=40)
  pdf.set_text_color(106,106,106)
  pdf.set_font('Arial','B',16)
  pdf.set_xy(80,45)
  pdf.write(5,titulo+"\n\n\n")
  pdf.set_font('Arial','',12)
  pdf.write(5,texto+"\n\n")
  pdf.set_font('Arial','B',14)
  pdf.write(5, "\nComentarios técnicos:\n\n")
  pdf.set_font('Arial','',12)
  pdf.write(5,comentarios)
  html = create_download_link(pdf.output(dest="S").encode("latin-1"), nombre)
  pdf.close()
  return html
