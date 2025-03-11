# RAG AI Engineer Assignment

This repository contains a Retrieval Augmented Generation (RAG) application for the Shyftlabs AI Engineer Take Home Assignment.

## Overview

The application allows users to:
- **Upload Documents:** Supports PDF and HTML file uploads.
- **Index Text:** Extracts text from documents, splits it into chunks, and indexes using FAISS.
- **Search:** Supports both semantic (via FAISS) and keyword-based search.
- **Generate Answers:** Uses the indexed context to generate answers (stub implementation provided).
- **Frontend UI:** Provides a simple HTML interface for uploading and querying.

## Files

- **`main.py`**: Application code for the FastAPI server.
- **`requirements.txt`**: List of Python dependencies.
- **`README.md`**: Project overview and setup instructions.

## Instructions

1. **Clone the repository.**
2. **Ensure dependencies are installed via `requirements.txt`.**
3. **Deploy the application as per your deployment process (e.g., using a cloud service or containerized environment).**
4. **Access the application using the provided endpoints.**

## License

This project is licensed under the MIT License.

