# Essay Grader and Chatbot

## Description
This project is a web application that provides essay grading and chatbot functionalities. Users can upload essays for automated grading and interact with a language model through a chat interface.

## Features
*   **Essay Upload and Grading:** Upload PDF essays and receive automated grades and feedback.
*   **Interactive Chatbot:** Engage in conversations with a language model.
*   **Vector Database Integration:** Utilizes ChromaDB for efficient storage and retrieval of document embeddings.

## Setup
To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Chat_Scholar
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the application:**
    ```bash
    python app.py
    ```

    The application will typically run on `http://127.0.0.1:5000/`.

## Usage
*   **Home Page (`/`):** Provides navigation to other sections.
*   **Upload Essay (`/upload_essay`):** Upload your PDF essay here. After uploading, you will be redirected to the grading result page.
*   **Chat (`/chat`):** Interact with the chatbot.

## Folder Structure
*   `app.py`: The main Flask application file.
*   `requirements.txt`: Lists all Python dependencies.
*   `templates/`: Contains HTML templates for the web interface.
*   `essay_uploads/`: Stores uploaded essay files.
*   `chroma_db/`: Stores the ChromaDB vector database.
*   `venv/`: Python virtual environment.

## Technologies Used
*   Python
*   Flask (Web Framework)
*   ChromaDB (Vector Database)
*   Language Models (via API, e.g., Google Gemini, OpenAI GPT - *specify which one if applicable*)
