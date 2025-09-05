# /your_project_folder/app.py
import os
import shutil
from flask import Flask, request, redirect, url_for, render_template, flash, jsonify, session
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Define the folder to store uploaded PDF files
UPLOAD_FOLDER = 'uploads'
CHROMA_DB_FOLDER = 'chroma_db'
# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CHROMA_DB_FOLDER'] = CHROMA_DB_FOLDER
# It's a good practice to set a secret key for session management (e.g., for flashing messages)
app.secret_key = os.urandom(24)

def allowed_file(filename):
    """Checks if the file's extension is in the allowed set."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def show_upload_form():
    return render_template('index.html')

# Route for handling the file upload (POST)
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('show_upload_form'))
    
    file = request.files['file']

    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('show_upload_form'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # --- Clear old database ---
            if os.path.exists(app.config['CHROMA_DB_FOLDER']):
                shutil.rmtree(app.config['CHROMA_DB_FOLDER'])
            os.makedirs(app.config['CHROMA_DB_FOLDER'], exist_ok=True)

            # --- LangChain processing ---
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            vectordb = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                persist_directory=app.config['CHROMA_DB_FOLDER']
            )

            flash(f'File "{filename}" uploaded and embedded successfully!')
        except Exception as e:
            print(f"Error processing PDF and creating embeddings: {e}")
            flash(f'File "{filename}" was uploaded, but processing failed. Error: {e}')

        return redirect(url_for('show_upload_form'))
    else:
        flash('Invalid file type. Only PDF files are allowed.')
        return redirect(url_for('show_upload_form'))

@app.route('/chat', methods=['GET'])
def chat():
    # Check if a document has been processed and a DB exists.
    db_path = app.config['CHROMA_DB_FOLDER']
    if not os.path.exists(db_path) or not os.listdir(db_path):
        flash('Please upload a document first to start a chat.')
        return redirect(url_for('show_upload_form'))

    # Initialize chat history with a greeting if it doesn't exist
    if 'chat_history' not in session:
        session['chat_history'] = [
            {'role': 'bot', 'content': 'Hello! How can I help you with your document today?'}
        ]
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Check if a document has been processed and a DB exists.
    db_path = app.config['CHROMA_DB_FOLDER']
    if not os.path.exists(db_path) or not os.listdir(db_path):
        return jsonify({
            "error": "No document has been processed. Please upload a document first."
        }), 400

    # Ensure the user's query is added to history
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({'role': 'user', 'content': query})
    session.modified = True

    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load the existing vector store
        vectordb = Chroma(persist_directory=app.config['CHROMA_DB_FOLDER'], embedding_function=embeddings)
        
        # Initialize the LLM and the QA chain
        # Make sure you have Ollama running with a model, e.g., `ollama run llama2`
        llm = Ollama(model="phi3:mini") 
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Get the answer
        result = qa_chain.invoke(query)
        answer = result.get('result', 'Sorry, I could not find an answer.')

        # Add bot's answer to history
        session['chat_history'].append({'role': 'bot', 'content': answer})
        session.modified = True
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error during question answering: {e}")
        return jsonify({"error": "An error occurred while processing your question."}), 500

if __name__ == '__main__':
    app.run(debug=True)
