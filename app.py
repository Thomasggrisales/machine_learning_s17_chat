import io
import platform

import streamlit as st
from PIL import Image
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# 1. Configuración de la página con temática médica
st.set_page_config(page_title="Asistente Farmacológico", page_icon="💊", layout="wide")

# 2. Inicializar el historial de chat en la sesión
if "mensajes_chat" not in st.session_state:
    st.session_state.mensajes_chat = [
        {"role": "assistant", "content": "¡Hola! Soy tu asistente virtual especializado en medicamentos 💊. Por favor, sube tu guía médica en PDF en el menú lateral y pregúntame lo que necesites saber."}
    ]

@st.cache_resource(show_spinner="Analizando el documento médico...")
def crear_base_conocimiento(pdf_bytes: bytes, api_key: str):
    """Extrae el texto, lo divide en fragmentos y crea el índice FAISS."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texto = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    fragmentos = splitter.split_text(texto)
    if not fragmentos:
        return None, len(texto), 0

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    base = FAISS.from_texts(fragmentos, embeddings)
    return base, len(texto), len(fragmentos)

# 3. Barra lateral (Sidebar) para configuraciones
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100) # Icono de farmacia opcional
    st.header("⚙️ Configuración")
    
    # Clave de API
    api_key = st.text_input("Ingresa tu Clave de OpenAI", type="password", help="Tu clave se usa solo durante esta sesión.")
    
    # Carga del PDF
    st.subheader("📚 Base de Conocimiento")
    pdf = st.file_uploader("Carga la guía de medicamentos (PDF)", type="pdf")
    
    st.divider()
    st.markdown("""
    ### ⚠️ Aviso Médico Importante
    *Este chatbot tiene fines estrictamente **informativos**. No sustituye el diagnóstico, consejo o tratamiento de un profesional de la salud. **Nunca te automediques**.*
    """)
    st.caption(f"Versión de Python: {platform.python_version()}")

# 4. Pantalla Principal del Chatbot
st.title("🩺 Asistente Virtual sobre Medicamentos")
st.markdown("Consulta información sobre analgésicos, antibióticos, interacciones y usos seguros basado en el documento proporcionado.")

# 5. Lógica principal
if not api_key:
    st.info("👈 Por favor, ingresa tu clave de API de OpenAI en el menú lateral para comenzar.")
elif pdf is None:
    st.info("👈 Sube un archivo PDF con información de medicamentos en el menú lateral.")
else:
    try:
        # Procesar PDF (se cachea automáticamente si es el mismo archivo)
        base, n_caracteres, n_fragmentos = crear_base_conocimiento(pdf.getvalue(), api_key)

        if base is None:
            st.error("No se pudo extraer texto del PDF (¿es un PDF escaneado sin OCR?).")
            st.stop()

        # Mostrar el historial de mensajes en la pantalla
        for mensaje in st.session_state.mensajes_chat:
            with st.chat_message(mensaje["role"]):
                st.markdown(mensaje["content"])

        # 6. Input del usuario para el chat
        if pregunta := st.chat_input("Ej: ¿Para qué sirve el ibuprofeno? o ¿Cuáles son los riesgos de automedicarse?"):
            
            # Mostrar la pregunta del usuario inmediatamente
            with st.chat_message("user"):
                st.markdown(pregunta)
            
            # Guardar la pregunta en el historial
            st.session_state.mensajes_chat.append({"role": "user", "content": pregunta})

            # 7. Generar la respuesta (RAG)
            with st.chat_message("assistant"):
                with st.spinner("Consultando la guía de medicamentos..."):
                    # Recuperación de contexto
                    docs = base.similarity_search(pregunta, k=4)
                    contexto = "\n\n---\n\n".join(d.page_content for d in docs)

                    # Configurar el LLM
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
                    
                    # Prompt del sistema especializado en salud
                    prompt_sistema = (
                        "Eres un asistente virtual experto en farmacología y medicamentos. "
                        "Tu objetivo es responder de manera clara, empática y profesional. "
                        "REGLAS ESTRICTAS:\n"
                        "1. Responde ÚNICAMENTE usando el contexto proporcionado del documento.\n"
                        "2. Si la respuesta no está en el contexto, di: 'Lo siento, no encuentro esa información en la guía proporcionada.' No inventes datos.\n"
                        "3. NO recetes medicamentos ni des diagnósticos.\n"
                        "4. Siempre que hables de dosis, efectos secundarios o usos, añade una breve advertencia de que se debe consultar a un médico."
                    )

                    # Preparar los mensajes para el LLM (incluyendo el contexto actual)
                    mensajes = [
                        ("system", prompt_sistema),
                        ("human", f"Contexto del documento médico:\n{contexto}\n\nPregunta del usuario: {pregunta}")
                    ]

                    # Ejecutar y mostrar respuesta en streaming (efecto de escritura)
                    respuesta = llm.invoke(mensajes)
                    
                    # Renderizar respuesta
                    st.markdown(respuesta.content)
                    
                    # Opción para ver las fuentes (colapsado para no ensuciar el chat)
                    with st.expander("🔍 Ver fragmentos del documento consultados"):
                        for i, d in enumerate(docs, 1):
                            st.markdown(f"**Fragmento {i}:**\n_{d.page_content}_")

            # Guardar la respuesta del asistente en el historial
            st.session_state.mensajes_chat.append({"role": "assistant", "content": respuesta.content})

    except Exception as e:
        st.error(f"Error al procesar la solicitud: {e}")
