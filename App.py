#...librerias...
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# --- CONEXIONES ---
from pymongo import MongoClient
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import SearchRequest, NamedVector
from qdrant_client.models import Filter

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


# ====================================================
#  Cargar secrets para conectar a las bases de datos #
# ====================================================

MONGO_URI = st.secrets["mongo"]["uri"]
MONGO_DB = st.secrets["mongo"]["db"]
MONGO_COLLECTION = st.secrets["mongo"]["collection"]

NEO4J_URI = st.secrets["neo4j"]["uri"]
NEO4J_USER = st.secrets["neo4j"]["user"]
NEO4J_PASSWORD = st.secrets["neo4j"]["password"]

QDRANT_URL = st.secrets["qdrant"]["url"]
QDRANT_KEY = st.secrets["qdrant"]["api_key"]
QDRANT_COLLECTION = st.secrets["qdrant"]["collection"]


# ====== Conexiones cacheadas ======

@st.cache_resource
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    col = db[MONGO_COLLECTION]
    return col


@st.cache_resource
def get_neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver


@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
    return client


@st.cache_resource
def get_sentence_model():
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    return SentenceTransformer(model_name)


mongo_col = get_mongo_collection()
neo_driver = get_neo4j_driver()
qdrant = get_qdrant_client()
embedder = get_sentence_model()


# ================================ #
#        funciones Mongo           #
# ================================ #

def consulta_providecia(nombre, preview_chars=250):
    query = {"providencia": nombre}
    res = list(mongo_col.find(query,
                              {'tipo': True, 'anio': True, 'texto': True, '_id': False}))

    if not res:
        return []

    for doc in res:
        texto = doc.get("texto", "")
        doc["texto_preview"] = texto[:preview_chars] + "..."

    return res


def mongo_obtener_completo(nombre):
    return mongo_col.find_one({"providencia": nombre}, {"_id": False})


def consulta_tipo_sentencia(tipo, preview_chars=250):
    query = {'tipo': tipo}
    res = list(mongo_col.find(query,
                              {'providencia': True, 'tipo': True,
                               'anio': True, 'texto': True, '_id': False}))

    if not res:
        return []

    for doc in res:
        texto = doc.get("texto", "")
        doc["texto_preview"] = texto[:preview_chars] + "..."
    mensage = f"Se encontraron {len(res)} resultados de sentencias de tipo: '{tipo}'\n"
    return res, mensage


def consulta_palabra(palabra, preview_chars=250):
    query = {'$text': {'$search': palabra}}
    res = list(mongo_col.find(query,
                              {'providencia': True, 'tipo': True,
                               'anio': True, 'texto': True, '_id': False}))

    if not res:
        return []

    for doc in res:
        texto = doc.get("texto", "")
        doc["texto_preview"] = texto[:preview_chars] + "..."
    mensage = f"Se encontraron {len(res)} resultados para con la palabra: '{palabra}'\n"
    return res, mensage


def mongo_obtener_preview(providencias):
    cursor = mongo_col.find(
        {"providencia": {"$in": list(providencias)}},
        {"_id": False}
    )
    docs = list(cursor)
    return {d["providencia"]: d for d in docs}


# ================================ #
#        funciones Neo4j           #
# ================================ #

def consultar_similitud_neo(providencia):
    query = """
        MATCH (p:Providencia {nombre: $nombre})-[r:SIMILAR_A]-(q:Providencia)
        RETURN p.nombre AS providencia,
               q.nombre AS similar_a,
               r.score AS similitud
        ORDER BY similitud DESC
    """

    with neo_driver.session() as session:
        result = session.run(query, nombre=providencia)
        records = result.data()

    df = pd.DataFrame(records)
    return df.reset_index(drop=True)


def grafo_similitud_nx(driver, providencia, threshold=0):

    query = """
    MATCH (p:Providencia {nombre: $nombre})-[r:SIMILAR_A]-(q:Providencia)
    WHERE r.score >= $threshold
    RETURN p.nombre AS origen,
           q.nombre AS destino,
           r.score AS similitud
    """

    # --- Ejecutar la consulta en Neo4j ---
    try:
        with driver.session() as session:
            data = session.run(
                query,
                nombre=providencia,
                threshold=threshold
            ).data()

    except Exception as e:
        print("Error ejecutando el grafo:", e)
        return None

    if not data:
        print("Sin resultados")
        return None

    # --- Construir red con PyVis ---
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")

    for row in data:
        origen = row["origen"]
        destino = row["destino"]
        score = row["similitud"]

        net.add_node(origen, label=origen, color="#FF66B3")
        net.add_node(destino, label=destino, color="#66B3FF")
        net.add_edge(origen, destino, label=f"{score:.3f}")

    net.force_atlas_2based()

    # retornar el grafo ya construido
    return net




# ================================ #
#        funciones Qdrant          #
# ================================ #

def preguntar_qdrant(texto_consulta: str,
                     limite: int = 5,
                     collection_name: str = "providencias"):

    try:
        # 1. Embedding del texto de consulta
        embedding = embedder.encode(
            texto_consulta,
            convert_to_numpy=True
        )

        # 2. Búsqueda en Qdrant (API correcta v1.16+)
        resultados = qdrant.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=limite
        )

        if not resultados.points:
            return []

        salida = []

        # 3. Recorrer resultados
        for r in resultados.points:

            payload = r.payload
            score = r.score

            prov = payload.get("providencia")
            tipo = payload.get("tipo")
            anio = payload.get("anio")

            # 4. Traer texto desde Mongo
            doc_mongo = mongo_obtener_completo(prov)

            if doc_mongo and "texto" in doc_mongo:
                preview = doc_mongo["texto"][:250] + "..."
            else:
                preview = "(Sin texto disponible)"

            # 5. Armar resultado limpio para Streamlit
            salida.append({
                "providencia": prov,
                "tipo": tipo,
                "anio": anio,
                "score": score,
                "preview_texto": preview
            })

        return salida

    except Exception as e:
        print("ERROR Qdrant:", e)
        return []


# ============ Generar PDF ============

def generar_pdf(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph(f"<b>Providencia {data['providencia']}</b>", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"<b>Tipo:</b> {data['tipo']}", styles["Normal"]),
        Paragraph(f"<b>Año:</b> {data['anio']}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(data["texto"], styles["Normal"])
    ]

    doc.build(content)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# ================================== #
#             STREAMLIT              #
# ================================== #

def main():
    st.set_page_config(page_title="Consulta Providencias",
                       layout="wide")

    st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1f2957;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .stRadio > label, .stRadio div label {
        color: white !important;
    }
    .header-img {
        position: absolute;
        top: 10px;
        right: 25px;
    }
</style>
<img src="colibri.png" class="header-img">

""", unsafe_allow_html=True)

    st.title("Explorador de Providencias")

    # ---------- estados globales ----------
    if "resultado_busqueda" not in st.session_state:
        st.session_state.resultado_busqueda = None
    if "full_doc" not in st.session_state:
        st.session_state.full_doc = None
    if "modo_detalle" not in st.session_state:
        st.session_state.modo_detalle = False
    if "ultima_opcion" not in st.session_state:
        st.session_state.ultima_opcion = None

    st.sidebar.header("Opciones")
    opcion = st.sidebar.radio(
        "¿Qué información desea consultar?:",
        [
            "Buscar por providencia",
            "Consultar por tipo de providencia",
            "Buscar por palabra clave",
            "Realizar búsquedas de providencias con similitud",
            "Grafo de similitud",
            "Búsqueda semántica"
        ])

    # >>> NUEVO: reseteo de estado al cambiar de opción <<<
    if st.session_state.ultima_opcion != opcion:
        st.session_state.modo_detalle = False
        st.session_state.full_doc = None
        st.session_state.resultado_busqueda = None
        st.session_state.ultima_opcion = opcion

    # ==========================================================
    #    "Buscar por providencia"
    # ==========================================================

    if opcion == "Buscar por providencia":

        # MODO DETALLE
        if st.session_state.modo_detalle and st.session_state.full_doc:
            doc = st.session_state.full_doc

            st.subheader("Detalle de la providencia")
            st.markdown(f"### {doc['tipo']} - {doc['providencia']} - {doc['anio']}")
            st.markdown("#### Texto completo:")
            st.write(doc["texto"])

            st.markdown("---")
            if st.button("⬅ Volver a la búsqueda"):
                st.session_state.modo_detalle = False
            return

        # MODO BÚSQUEDA
        st.subheader("* Consultar por nombre de una providencia")
        nombre = st.text_input("Nombre (ej: A053-24)")

        if st.button("Buscar") and nombre:
            with st.spinner("Consultando en MongoDB..."):
                docs = consulta_providecia(nombre)

            if not docs:
                st.session_state.resultado_busqueda = None
                st.session_state.full_doc = None
                st.warning(f"No hay resultados para {nombre}.")
            else:
                prev_doc = docs[0]
                st.session_state.resultado_busqueda = prev_doc
                st.session_state.full_doc = mongo_obtener_completo(nombre)
                st.session_state.modo_detalle = False

        if st.session_state.resultado_busqueda is not None:
            doc = st.session_state.resultado_busqueda
            full_doc = st.session_state.full_doc

            st.success(f"Resultado encontrado para {nombre}")

            st.markdown(f"""
            ###  {doc['tipo']} - {nombre} - {doc['anio']}
            **Vista previa:**  
            {doc['texto_preview']}
            """)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Ver detalle completo"):
                    st.session_state.modo_detalle = True

            with col2:
                pdf_bytes = generar_pdf(full_doc)
                st.download_button(
                    label=" Descargar PDF",
                    data=pdf_bytes,
                    file_name=f"{full_doc['providencia']}.pdf",
                    mime="application/pdf"
                )

    # ==========================================================
    #    CONSULTAR POR TIPO
    # ==========================================================

    elif opcion == "Consultar por tipo de providencia":
        st.subheader("* Consultar por tipo")
        tipo = st.selectbox("Tipo", ["Auto", "Tutela", "Constitucionalidad"])

        if st.button("Buscar por tipo"):
            with st.spinner("Consultando en MongoDB..."):
                docs, mensaje = consulta_tipo_sentencia(tipo)

            if docs:
                st.text(mensaje)

                # DataFrame sin texto_preview
                df = pd.DataFrame(docs).drop(columns=["texto_preview"])
                st.dataframe(df, use_container_width=True)

                # Botones debajo
                st.markdown("### Acciones por documento")

                for d in docs:
                    prov = d["providencia"]

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"Ver detalle {prov}", key=f"detalle_tipo_{prov}"):
                            st.session_state.full_doc = mongo_obtener_completo(prov)
                            st.session_state.modo_detalle = True

                    with col2:
                        full_doc = mongo_obtener_completo(prov)
                        pdf_bytes = generar_pdf(full_doc)
                        st.download_button(
                            label=f"PDF {prov}",
                            data=pdf_bytes,
                            file_name=f"{prov}.pdf",
                            mime="application/pdf",
                            key=f"pdf_tipo_{prov}"
                        )
            else:
                st.info(f"No hay documentos del tipo {tipo}")

    # ==========================================================
    #    BUSCAR POR PALABRA CLAVE
    # ==========================================================

    elif opcion == "Buscar por palabra clave":
        st.subheader("* Búsqueda por texto")
        palabra = st.text_input("Palabra a buscar:")

        if st.button("Buscar"):
            with st.spinner("Consultando en MongoDB..."):
                resultados, mensaje = consulta_palabra(palabra)

            if not resultados:
                st.text(mensaje)
                st.warning("No hay coincidencias.")
            else:
                df = pd.DataFrame(resultados).drop(columns=["texto_preview"])
                st.dataframe(df, use_container_width=True)

                st.markdown("### Acciones por documento")

                for d in resultados:
                    prov = d["providencia"]

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"Ver detalle {prov}", key=f"detalle_kw_{prov}"):
                            st.session_state.full_doc = mongo_obtener_completo(prov)
                            st.session_state.modo_detalle = True

                    with col2:
                        full_doc = mongo_obtener_completo(prov)
                        pdf_bytes = generar_pdf(full_doc)
                        st.download_button(
                            label=f"PDF {prov}",
                            data=pdf_bytes,
                            file_name=f"{prov}.pdf",
                            mime="application/pdf",
                            key=f"pdf_kw_{prov}"
                        )

    # ==========================================================
    #    CONSULTAR SIMILITUDES (tabla)
    # ==========================================================

    elif opcion == "Realizar búsquedas de providencias con similitud":
        st.subheader("* Consultar similitudes")
        providencia = st.text_input("Nombre (ej: A053-24)")

        if st.button("Consultar similitudes") and providencia:
            with st.spinner("Consultando en Neo4j..."):
                rows = consultar_similitud_neo(providencia)
            if not rows.empty:
                st.dataframe(rows)
            else:
                st.info("No se encontraron relaciones.")

    # ==========================================================
    #    GRAFO DE SIMILITUD
    # ==========================================================

    elif opcion == "Grafo de similitud":
        st.subheader("* Grafo de similitud")
        nombre_raw = st.text_input("Providencia base (ej: A053-24)")
        # normalizar (quita espacios y soft-hyphen)
        nombre = nombre_raw.strip().replace("\u00AD", "")
        threshold = st.slider("Umbral mínimo", 0.0, 1.0, 0.3)

        if st.button("Generar grafo") and nombre:
            with st.spinner("Construyendo grafo..."):
                net = grafo_similitud_nx(neo_driver, nombre, threshold)

                net.save_graph("graph.html")
                with open("graph.html", "r", encoding="utf-8") as f:
                    html = f.read()

                st.components.v1.html(html, height=650, scrolling=True)

    # ==========================================================
    #    BÚSQUEDA SEMÁNTICA
    # ==========================================================

    elif opcion == "Búsqueda semántica":
        st.subheader("* Búsqueda semántica")

        consulta = st.text_input("Consulta natural",
                                 value="derecho a la salud en tutela")
        limite = st.slider("Resultados", 1, 10, 5)

        if st.button("Buscar semánticamente") and consulta:
            with st.spinner("Consultando en Qdrant..."):
                resultados = preguntar_qdrant(consulta, limite)

            if not resultados:
                st.info("Qdrant no devolvió resultados.")
                return

            # Convertir resultados en DataFrame directo
            df = pd.DataFrame(resultados)

            # Mostrar tabla con vista previa incluida
            st.dataframe(df, use_container_width=True)

            # --- Acciones por cada resultado ---
            st.markdown("### Acciones por documento")

            for r in resultados:
                prov = r["providencia"]

                col1, col2 = st.columns(2)

                # Ver detalle
                with col1:
                    if st.button(f"Ver detalle {prov}", key=f"detalle_sem_{prov}"):
                        st.session_state.full_doc = mongo_obtener_completo(prov)
                        st.session_state.modo_detalle = True

                # Descargar PDF
                with col2:
                    full_doc = mongo_obtener_completo(prov)
                    pdf_bytes = generar_pdf(full_doc)
                    st.download_button(
                        label=f"PDF {prov}",
                        data=pdf_bytes,
                        file_name=f"{prov}.pdf",
                        mime="application/pdf",
                        key=f"pdf_sem_{prov}"
                    )


if __name__ == "__main__":
    main()



