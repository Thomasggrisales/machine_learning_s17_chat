import io
import platform

import streamlit as st
from PIL import Image
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG con PDF", page_icon="💬")

# Título y presentación
st.title("Generación Aumentada por Recuperación (RAG) 💬")
st.write("Versión de Python:", platform.python_version())

try:
    st.image(Image.open("Chat_pdf.png"), width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Clave de API
api_key = st.text_input("Ingresa tu Clave de OpenAI", type="password")

# Carga del PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")


@st.cache_resource(show_spinner="Creando la base de conocimiento...")
def crear_base_conocimiento(pdf_bytes: bytes, api_key: str):
    """Extrae el texto, lo divide en fragmentos y crea el índice FAISS.
    Se guarda en caché para no recalcular embeddings (ni pagarlos) en cada rerun."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texto = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_text(texto)
    if not fragmentos:
        return None, len(texto), 0

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    base = FAISS.from_texts(fragmentos, embeddings)
    return base, len(texto), len(fragmentos)


if not api_key:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
elif pdf is None:
    st.info("Por favor carga un archivo PDF para comenzar")
else:
    try:
        base, n_caracteres, n_fragmentos = crear_base_conocimiento(pdf.getvalue(), api_key)

        if base is None:
            st.error("No se pudo extraer texto del PDF (¿es un PDF escaneado sin OCR?).")
            st.stop()

        st.info(f"Texto extraído: {n_caracteres} caracteres")
        st.success(f"Documento dividido en {n_fragmentos} fragmentos")

        st.subheader("Escribe qué quieres saber sobre el documento")
        pregunta = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        if pregunta:
            # 1. Recuperación: fragmentos más parecidos a la pregunta
            docs = base.similarity_search(pregunta, k=4)
            contexto = "\n\n---\n\n".join(d.page_content for d in docs)

            # 2. Generación: el modelo responde usando solo ese contexto
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            mensajes = [
                ("system",
                 "Responde la pregunta usando únicamente el contexto proporcionado. "
                 "Si la respuesta no está en el contexto, dilo claramente. Responde en español."),
                ("human", f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"),
            ]
            with st.spinner("Pensando..."):
                respuesta = llm.invoke(mensajes)

            st.markdown("### Respuesta:")
            st.markdown(respuesta.content)

            with st.expander("Ver fragmentos recuperados"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Fragmento {i}**")
                    st.text(d.page_content)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
        import traceback
        st.code(traceback.format_exc())
