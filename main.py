from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
import io
import PyPDF2
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# Initialize embedding model and FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_dim = 384  # Dimension for 'all-MiniLM-L6-v2'
faiss_index = faiss.IndexFlatL2(vector_dim)
# List to store text chunks in the same order as embeddings added to FAISS
text_chunks = []

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_html(file_bytes: bytes) -> str:
    """Extract text from an HTML file."""
    soup = BeautifulSoup(file_bytes, "html.parser")
    return soup.get_text(separator="\n")

def split_text(text: str, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.
    chunk_size: maximum words per chunk.
    overlap: number of words to overlap between chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def add_chunks_to_index(chunks):
    """Embed text chunks and add them to the FAISS index."""
    global text_chunks, faiss_index
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    faiss_index.add(embeddings)
    text_chunks.extend(chunks)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload endpoint: Accepts a PDF or HTML file, extracts text, splits into chunks,
    computes embeddings, and indexes them using FAISS.
    """
    file_bytes = await file.read()
    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(file_bytes)
    elif file.content_type == "text/html":
        text = extract_text_from_html(file_bytes)
    else:
        return {"error": "Unsupported file type. Please upload a PDF or HTML file."}
    
    chunks = split_text(text)
    add_chunks_to_index(chunks)
    return {"status": "Document uploaded and indexed", "num_chunks": len(chunks)}

def keyword_search(query, top_k=5):
    """Simple keyword search: returns text chunks that contain the query."""
    results = []
    query_lower = query.lower()
    for chunk in text_chunks:
        if query_lower in chunk.lower():
            results.append(chunk)
        if len(results) >= top_k:
            break
    return results

def semantic_search(query, top_k=5):
    """Semantic search: uses FAISS index to retrieve similar text chunks based on embeddings."""
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
    return results

def generate_answer(query, context):
    """
    Stub function for Retrieval Augmented Generation.
    Replace this stub with a call to an LLM API for production usage.
    """
    answer = f"Answer based on context:\n\n{context}\n\nfor query: {query}"
    return answer

@app.get("/")
async def root():
    """Serve a simple HTML frontend to interact with the application."""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>RAG App</title>
        </head>
        <body>
            <h1>Retrieval Augmented Generation App</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf,.html"/>
                <button type="submit">Upload Document</button>
            </form>
            <br>
            <form id="queryForm">
                <input type="text" id="queryInput" placeholder="Enter your question" size="50"/>
                <select id="searchType">
                    <option value="semantic">Semantic Search</option>
                    <option value="keyword">Keyword Search</option>
                </select>
                <button type="submit">Ask</button>
            </form>
            <pre id="response"></pre>
            <script>
                // File upload handler
                document.getElementById("uploadForm").onsubmit = async (e) => {
                    e.preventDefault();
                    const formData = new FormData(e.target);
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    document.getElementById("response").textContent = JSON.stringify(result, null, 2);
                };
                // Query form handler with streaming response
                document.getElementById("queryForm").onsubmit = async (e) => {
                    e.preventDefault();
                    const query = document.getElementById("queryInput").value;
                    const searchType = document.getElementById("searchType").value;
                    const response = await fetch(`/ask?query=${encodeURIComponent(query)}&type=${searchType}`);
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let result = '';
                    while(true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        result += decoder.decode(value);
                        document.getElementById("response").textContent = result;
                    }
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/ask")
async def ask(query: str, type: str = "semantic"):
    """
    Ask endpoint: Processes the query by performing either semantic or keyword search,
    generates an answer with context, and streams the answer back to the client.
    """
    if type == "semantic":
        results = semantic_search(query)
    else:
        results = keyword_search(query)
    context = "\n".join(results)
    
    async def answer_generator():
        answer = generate_answer(query, context)
        # Streaming simulation: yield answer one character at a time.
        for char in answer:
            yield char

    return StreamingResponse(answer_generator(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
