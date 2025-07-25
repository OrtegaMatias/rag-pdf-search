# 📚 Sistema RAG con DPR y FAISS

Un sistema de Recuperación Aumentada por Generación (RAG) que utiliza Dense Passage Retrieval (DPR) de Facebook y FAISS para búsqueda semántica eficiente sobre documentos PDF. El ejemplo incluido trabaja con Don Quijote de la Mancha, pero el sistema es compatible con cualquier documento PDF.

## 🌟 Características

- **Procesamiento automático de PDFs**: Convierte PDFs a texto limpio, eliminando headers, footers y elementos no deseados
- **Chunking inteligente**: División del texto en fragmentos con solapamiento para mantener contexto
- **Embeddings con DPR**: Utiliza modelos pre-entrenados de Facebook para generar representaciones vectoriales densas
- **Búsqueda ultrarrápida**: Implementación con FAISS para búsquedas vectoriales eficientes
- **Interfaz web intuitiva**: Gradio UI para interacción fácil con el sistema
- **Caché inteligente**: Guarda embeddings e índices para evitar recálculos

## 🏗️ Arquitectura

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF Doc   │ --> │ Text Chunker │ --> │ DPR Encoder │
└─────────────┘     └──────────────┘     └─────────────┘
                                                  │
                                                  v
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Gradio    │ <-- │ FAISS Search │ <-- │ Embeddings  │
│     UI      │     │              │     │   Storage   │
└─────────────┘     └──────────────┘     └─────────────┘
```

## 📋 Requisitos

- Python 3.8+
- CUDA (opcional, para aceleración GPU)
- 4GB RAM mínimo (8GB recomendado)

## 🚀 Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/rag-pdf-search.git
cd rag-pdf-search
```

2. **Crear entorno virtual** (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Preparar tu documento PDF**
   - Renombra tu archivo PDF a `Mancha.pdf` y colócalo en el directorio raíz
   - El sistema procesará automáticamente cualquier PDF con este nombre
   - **Nota**: Aunque el ejemplo usa Don Quijote, puedes usar cualquier documento PDF (libros, papers, documentación, etc.)

## 💻 Uso

### Ejecución básica
```bash
python src/main.py
```

Esto iniciará la interfaz web de Gradio. Abre tu navegador en `http://localhost:7860` para interactuar con el sistema.

### Ejemplos de uso con diferentes documentos

El sistema funciona con cualquier PDF. Solo necesitas renombrarlo a `Mancha.pdf`:

```bash
# Para un paper académico
cp mi_paper_investigacion.pdf Mancha.pdf
python src/main.py

# Para documentación técnica
cp manual_python.pdf Mancha.pdf
python src/main.py

# Para un libro (ejemplo: Don Quijote)
cp don_quijote.pdf Mancha.pdf
python src/main.py
```

### Uso programático
```python
from src.retriever import search_relevant_contexts
from src.embedding import load_dpr_models

# Cargar modelos
ctx_tokenizer, ctx_encoder, q_tokenizer, q_encoder = load_dpr_models()

# Buscar contextos relevantes
D, I = search_relevant_contexts(
    "¿Cuál es el tema principal del documento?",
    q_tokenizer,
    q_encoder,
    index,
    k=5
)
```

## 📁 Estructura del Proyecto

```
rag-pdf-search/
│
├── src/
│   ├── main.py          # Punto de entrada principal y UI Gradio
│   ├── ingestion.py     # Procesamiento de PDFs y chunking
│   ├── embedding.py     # Generación de embeddings con DPR
│   ├── database.py      # Construcción del índice FAISS
│   └── retriever.py     # Lógica de búsqueda semántica
│
├── requirements.txt     # Dependencias del proyecto
├── Mancha.pdf          # Tu PDF de entrada (cualquier documento)
├── README.md           # Este archivo
│
└── [Archivos generados automáticamente]
    ├── Ficha.txt       # Texto extraído del PDF
    ├── paragraphs.txt  # Chunks de texto
    ├── embeddings.pt   # Embeddings guardados
    └── faiss.index     # Índice FAISS
```

## 🔧 Cómo Funciona (Paso a Paso)

### 1. **Procesamiento del PDF** (`ingestion.py`)
   - Extrae texto del PDF usando PyMuPDF
   - Elimina headers/footers repetidos
   - Limpia números de página y viñetas
   - Une palabras partidas por guiones
   - Normaliza espacios en blanco

### 2. **Chunking del Texto** (`ingestion.py`)
   - Divide el texto en fragmentos de 200 tokens
   - Mantiene 40 tokens de solapamiento entre chunks
   - Preserva el contexto entre fragmentos

### 3. **Generación de Embeddings** (`embedding.py`)
   - Carga modelos DPR pre-entrenados:
     - `facebook/dpr-ctx_encoder-single-nq-base` para contextos
     - `facebook/dpr-question_encoder-single-nq-base` para preguntas
   - Procesa chunks en batches de 32
   - Genera vectores de 768 dimensiones
   - Soporta GPU automáticamente si está disponible

### 4. **Construcción del Índice** (`database.py`)
   - Crea índice FAISS con distancia L2
   - Optimizado para búsquedas exactas
   - Persiste en disco para reutilización

### 5. **Búsqueda Semántica** (`retriever.py`)
   - Codifica la pregunta del usuario
   - Busca los k=5 contextos más similares
   - Retorna distancias y fragmentos relevantes

### 6. **Interfaz de Usuario** (`main.py`)
   - Gradio proporciona interfaz web
   - Muestra resultados rankeados con distancias
   - Incluye métricas de rendimiento

## ⚙️ Configuración

### Parámetros ajustables:

```python
# En main.py - preparar_sistema()
chunk_size=200        # Tamaño de los chunks en tokens
overlap=40           # Solapamiento entre chunks
batch_size=32        # Tamaño del batch para embeddings
max_length=256       # Longitud máxima de entrada
k=5                  # Número de resultados a retornar

# En ingestion.py - pdf_to_txt()
hf_lines=3           # Líneas a considerar como header/footer
cleanup_hyphens=True # Unir palabras partidas
normalize_whitespace=True # Normalizar espacios
```

## 🐛 Solución de Problemas

### Error: "KMP_DUPLICATE_LIB_OK"
El código ya incluye `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` para resolver conflictos de librerías en Windows.

### Memoria insuficiente
- Reduce `batch_size` en `encode_chunks()`
- Usa chunks más pequeños ajustando `chunk_size`
- Considera usar `faiss-gpu` si tienes GPU disponible

### El índice tarda mucho en crearse
- Los embeddings e índice se guardan automáticamente
- En ejecuciones posteriores se cargarán desde disco
- Para regenerar, elimina `embeddings.pt` y `faiss.index`

## 🚀 Mejoras Sugeridas

- [ ] Agregar soporte para múltiples documentos PDF simultáneos
- [ ] Permitir cargar PDFs desde la interfaz web sin renombrar
- [ ] Implementar re-ranking de resultados
- [ ] Añadir generación de respuestas con LLM
- [ ] Crear API REST para integración
- [ ] Implementar índices FAISS aproximados (IVF, HNSW) para datasets grandes
- [ ] Añadir logging y monitoreo
- [ ] Dockerizar la aplicación
- [ ] Añadir tests unitarios
- [ ] Implementar cache LRU para consultas frecuentes
- [ ] Soporte multiidioma
- [ ] Detección automática del tipo de documento (técnico, narrativo, académico)

## 📊 Rendimiento

- **Tiempo de indexación inicial**: ~30-60 segundos (dependiendo del hardware)
- **Tiempo de búsqueda**: <0.1 segundos para 5 resultados
- **Uso de memoria**: ~1-2GB con modelos cargados
- **Precisión**: Depende de la calidad del PDF y la similitud semántica

## 📄 Tipos de documentos soportados

El sistema funciona mejor con:
- **Libros y novelas** (ej: Don Quijote)
- **Papers académicos** y artículos científicos
- **Documentación técnica** y manuales
- **Informes** y documentos corporativos
- **Tesis** y trabajos de investigación

### Consideraciones:
- PDFs con mucho contenido gráfico pueden tener menor precisión
- Documentos escaneados requieren OCR previo
- Mejor rendimiento con textos en español e inglés
- El tamaño recomendado es hasta 1000 páginas

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea tu rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- Facebook Research por los modelos DPR
- El equipo de FAISS por la librería de búsqueda vectorial
- Gradio por la interfaz de usuario simple
- La comunidad de Hugging Face por transformers

## 📞 Contacto

Tu Nombre - [@tu_twitter](https://twitter.com/tu_twitter) - email@ejemplo.com

Link del Proyecto: [https://github.com/tu-usuario/rag-pdf-search](https://github.com/tu-usuario/rag-pdf-search)

---

⭐ Si este proyecto te resulta útil, considera darle una estrella!
