# Explorador de Providencias Judiciales

Aplicación web interactiva construida con **Streamlit**, diseñada para consultar, explorar y analizar providencias judiciales almacenadas en múltiples fuentes de datos:

- **MongoDB** → Texto completo de las providencias  
- **Neo4j Aura** → Grafo de similitud entre providencias  
- **Qdrant** → Búsqueda semántica mediante embeddings  
- **Sentence Transformers** → Modelo de lenguaje para generar embeddings

---

##  Funcionalidades principales

###  Búsqueda directa por nombre
Permite consultar cualquier providencia por su identificador (por ejemplo *A053-24*).  
Muestra:
- Tipo de providencia  
- Año  
- Vista previa  
- PDF descargable  
- Texto completo (modo detalle)

---

### Búsqueda por tipo
Consulta por categorías como:  
- Auto  
- Tutela  
- Constitucionalidad  

Con opción de descargar PDF de cada documento.

---

### Búsqueda por palabra clave
Realiza un `$text search` sobre MongoDB.  
Muestra coincidencias con vista previa y opción de descargar.

---

### Búsqueda semántica (Qdrant)
Basada en embeddings generados por:
