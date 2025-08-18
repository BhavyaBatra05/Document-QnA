import os
import time
import yaml
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import datetime
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import your backend code
from enhanced_doc_qa import (
    SmartDocumentProcessor, 
    InMemoryVectorStore,
    HallucinationResistantAnswerer,
    run_document_qa_system,
    QAState
)
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Initialize models
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load AI models with graceful fallback for missing API keys."""
    try:
        # Load LLM
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            llm = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash', 
                google_api_key=gemini_api_key
            )
            # st.success("✅ Google Gemini LLM loaded successfully!")
        else:
            llm = None
            st.warning("⚠️ Gemini API key not found in .env file. Running in demo mode without LLM.")
        
        # Load VLM models
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
            vlm_processor = AutoProcessor.from_pretrained(
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                token=huggingface_key
            )
            vlm_model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                token=huggingface_key
            )
            # st.success("✅ Vision Language Model loaded successfully!")
        except Exception as e:
            vlm_processor = None
            vlm_model = None
            st.warning(f"⚠️ Could not load VLM models: {e}")
            
        return llm, vlm_processor, vlm_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load models with error handling
try:
    llm, vlm_processor, vlm_model = load_models()
    
    # Expose models globally for agent functions
    globals()["llm"] = llm
    globals()["vlm_processor"] = vlm_processor
    globals()["vlm_model"] = vlm_model
except Exception as e:
    st.error(f"Error initializing models: {e}")
    llm, vlm_processor, vlm_model = None, None, None

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config("config.yaml")

PRIMARY_COLOR = CONFIG.get("company", {}).get("primary_color", "#2E86AB")
SECONDARY_COLOR = CONFIG.get("company", {}).get("secondary_color", "#A23B72")
SUCCESS_COLOR = "#F18F01"
BACKGROUND_COLOR = "#F8F9FA"
TEXT_COLOR = "#212529"
BORDER_COLOR = "#DEE2E6"
COMPANY_NAME = CONFIG.get("company", {}).get("name", "Document Q&A")
LOGO_URL = CONFIG.get("company", {}).get("logo_url", None)

SUPPORTED_FORMATS = CONFIG.get("system", {}).get("supported_formats", ["pdf", "docx", "txt"])
MAX_FILE_SIZE = CONFIG.get("system", {}).get("max_file_size", 50)
BATCH_SIZE = CONFIG.get("system", {}).get("batch_size", 5)
MAX_WORKERS = CONFIG.get("system", {}).get("max_workers", 3)

st.set_page_config(
    page_title=f"{COMPANY_NAME}",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
)

# Current user and timestamp info
CURRENT_USER = "BhavyaBatra05"
CURRENT_TIMESTAMP = "2025-08-13 17:04:17"

# Try to parse the timestamp or use current time as fallback
try:
    CURRENT_DATETIME = datetime.datetime.strptime(CURRENT_TIMESTAMP, "%Y-%m-%d %H:%M:%S")
except ValueError:
    CURRENT_DATETIME = datetime.datetime.now()

def get_current_timestamp():
    """Get formatted current timestamp."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_formatted_display_date():
    """Get a nicely formatted date for display."""
    return datetime.datetime.now().strftime("%d-%m-%Y")

def get_formatted_date():
    """Get a formatted date string that works across platforms."""
    now = datetime.datetime.now()
    try:
        # Try Unix-style formatting first
        return now.strftime("%-d-%-m-%y")
    except ValueError:
        # Fall back to Windows-compatible formatting
        day = str(now.day).lstrip('0')
        month = str(now.month).lstrip('0')
        year = now.strftime("%y")
        return f"{day}-{month}-{year}"
    
def display_logo():
    """Display the logo image."""
    # Adjusted to keep logo properly positioned within the frame
    logo_html = """
    <div style="display: flex; justify-content: flex-start; align-items: center; padding-left: 15px; padding-top: 10px;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT7zBX4VmSnYfD1Ismi7kk8MCDAGd-tPJrQwQ&s" alt="Logo" width="80" height="80">
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)

def start_new_chat():
    """Start a new chat session."""
    # Initialize chat_sessions if it doesn't exist
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = []
        
    # Get next session ID
    next_id = 1
    if st.session_state.chat_sessions:
        next_id = max([s.get("id", 0) for s in st.session_state.chat_sessions]) + 1
    
    # Create new session
    new_session = {
        "id": next_id,
        "messages": [{
            "role": "assistant", 
            "content": "Hello! I'm your document assistant. How can I help you today?"
        }],
        "last_message": "Hello! I'm your document assistant. How can I help you today?",
        "timestamp": get_current_timestamp(),
        "user": st.session_state.username
    }
    
    # Add to sessions and set as current
    st.session_state.chat_sessions.insert(0, new_session)
    st.session_state.chat_history = new_session["messages"].copy()

def load_chat_session(session_id):
    """Load a specific chat session by ID."""
    for session in st.session_state.chat_sessions:
        if session.get("id") == session_id:
            # Move this session to the top
            st.session_state.chat_sessions.remove(session)
            st.session_state.chat_sessions.insert(0, session)
            # Set as current chat
            st.session_state.chat_history = session["messages"].copy()
            return True
    return False

# =========================
# ----- STYLING ----------
# =========================

def set_custom_css():
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    .css-18e3th9 {{
        padding-top: 0rem;
        padding-bottom: 0rem;
        background-color: {BACKGROUND_COLOR} !important;
    }}
    .css-1d391kg {{
        padding-top: 0rem;
        padding-right: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
    }}
    div.block-container {{
        padding-top: 0rem;
        padding-bottom: 0rem;
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    button[data-testid="baseButton-secondary"] {{
        background-color: transparent;
        color: {PRIMARY_COLOR};
        border: none;
        border-radius: 50%;
        font-size: 24px;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
    }}
    .chat-container {{
        height: 400px;
        overflow-y: auto;
        border: 1px solid {BORDER_COLOR};
        padding: 1rem;
        background-color: white;
        border-radius: 5px;
    }}
    .header-style {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid {BORDER_COLOR};
        margin-bottom: 10px;
    }}
    .admin-toggle {{
        color: {PRIMARY_COLOR};
        cursor: pointer;
        font-weight: bold;
        text-decoration: none;
    }}
    .footer {{
        text-align: center;
        margin-top: 1rem;
        padding: 0.5rem;
        border-top: 1px solid {BORDER_COLOR};
        font-size: 0.8rem;
        color: #666;
    }}
    .history-item {{
        padding: 5px;
        margin-bottom: 5px;
        border-bottom: 1px solid {BORDER_COLOR};
    }}
    .user-message {{
        text-align: left;
        margin-bottom: 15px;
    }}
    .user-icon {{
        display: inline-block;
        background-color: #ff6b6b;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        margin-right: 8px;
        vertical-align: top;
    }}
    .bot-message {{
        text-align: center;
        margin-bottom: 15px;
    }}
    .bot-icon {{
        display: inline-block;
        background-color: #ffa94d;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        margin-right: 8px;
        vertical-align: top;
    }}
    .message-content {{
        display: inline-block;
        padding: 8px 12px;
        border-radius: 18px;
        background-color: #f1f3f5;
        max-width: 80%;
    }}
    .chat-input-container {{
        display: flex;
        align-items: center;
        margin-top: 15px;
        border: 1px solid #ddd;
        border-radius: 24px;
        padding: 5px;
    }}
    .chat-input {{
        flex-grow: 1;
        border: none;
        outline: none;
        padding: 8px 12px;
    }}
    .send-button {{
        background: none;
        border: none;
        color: {PRIMARY_COLOR};
        font-size: 20px;
        cursor: pointer;
        margin-right: 8px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    th, td {{
        text-align: left;
        padding: 8px;
        border: 1px solid {BORDER_COLOR};
    }}
    th {{
        background-color: #f2f2f2;
    }}
    tr:nth-child(even) {{
        background-color: #f9f9f9;
    }}
    </style>
    """, unsafe_allow_html=True)

# =========================
# ----- SESSION STATE -----
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "doc_states" not in st.session_state:
    st.session_state.doc_states = {}  # {filename: doc_state}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "username" not in st.session_state:
    st.session_state.username = CURRENT_USER
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant", 
        "content": "Hello! I'm your document assistant. How can I help you today?"
    }]
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = [{
        "id": 1,
        "messages": st.session_state.chat_history.copy(),
        "last_message": "Hello! I'm your document assistant. How can I help you today?",
        "timestamp": get_current_timestamp(),
        "user": st.session_state.username
    }]
# Update the session state initialization for admin_files_data
if "admin_files_data" not in st.session_state:
    # Initialize with empty data (no sample files)
    st.session_state.admin_files_data = {
        "Date": [],
        "File Name": [],
        "IsActive": [],
        "Ingested": []
    }
if "message_input" not in st.session_state:
    st.session_state.message_input = ""
if "current_doc_state" not in st.session_state:
    st.session_state.current_doc_state = None
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = {}

if "document_tracking" not in st.session_state:
    st.session_state.document_tracking = {
        "current_document_id": None,
        "current_document_name": None,
        "last_update": None
    }

# Function to handle document processing
def process_document(file_path):
    """Process a document using the backend system."""
    try:
        # Initialize document processor
        processor = SmartDocumentProcessor(
            llm=llm,
            vlm_processor=vlm_processor,
            vlm_model=vlm_model,
            batch_size=BATCH_SIZE,
            max_workers=MAX_WORKERS
        )
        
        # Extract text from document
        extraction_result = processor.extract_text_smart(file_path)
        
        if not extraction_result["success"]:
            return None, f"Failed to extract text: {extraction_result.get('error', 'Unknown error')}"
        
        # Create vector store
        vector_store = InMemoryVectorStore()
        vs_result = vector_store.create_vectorstore(extraction_result["text"])
        
        if not vs_result["success"]:
            return None, f"Failed to create vector store: {vs_result.get('error', 'Unknown error')}"
        
        # Create QA state
        qa_state = {
            "file_path": file_path,
            "text": extraction_result["text"],
            "chunks": vector_store.chunks if vector_store.chunks else [],
            "vectorstore": vector_store,
            "query": "",
            "answer": "",
            "retrieved_chunks": [],
            "extraction_method": extraction_result["extraction_method"],
            "word_count": extraction_result["word_count"],
            "chunk_count": vs_result["chunk_count"],
            "next_action": "continue"
        }
        
        return qa_state, None
    except Exception as e:
        return None, str(e)

# Function to handle message submission
def handle_message_submit():
    """Process user message and generate response using the backend QA system."""
    if st.session_state.message_input:
        message = st.session_state.message_input
        
        # Initialize chat_sessions if it doesn't exist
        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = []
        
        # Get or create current session
        if not st.session_state.chat_sessions:
            current_session = {
                "id": 1,
                "messages": [],
                "timestamp": get_current_timestamp(),
                "user": st.session_state.username
            }
            st.session_state.chat_sessions.insert(0, current_session)
        else:
            current_session = st.session_state.chat_sessions[0]
        
        # Add user message to chat history and current session
        st.session_state.chat_history.append({"role": "user", "content": message})
        current_session["messages"].append({"role": "user", "content": message})
        
        # Check if we have a processed document
        if not st.session_state.current_doc_state:
            response = "Please upload a document first before asking questions."
        else:
            try:
                # Set the query in the QA state
                st.session_state.current_doc_state["query"] = message
                
                # Use the backend to get an answer
                vector_store = st.session_state.current_doc_state.get("vectorstore")
                if vector_store:
                    # Get relevant chunks
                    chunks = vector_store.retrieve_chunks(message, k=5)
                    
                    # Generate answer
                    answerer = HallucinationResistantAnswerer(llm=llm)
                    answer_result = answerer.generate_answer(message, chunks)
                    
                    # Set response
                    response = answer_result["answer"]
                else:
                    response = "Document processing is incomplete. Please try uploading the document again."
            except Exception as e:
                response = f"Error processing your question: {str(e)}"
        
        # Add assistant response to chat history and current session
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        current_session["messages"].append({"role": "assistant", "content": response})
        
        # Update session metadata
        current_session["last_message"] = response
        current_session["timestamp"] = get_current_timestamp()
        
        # Clear the input
        st.session_state.message_input = ""

# =========================
# ----- USER INTERFACE ----
# =========================

def user_interface():
    """Render the user interface with increased top padding to lower the header."""
    # Significantly increase top padding to push everything further down
    st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
    header_container = st.container()
    
    with header_container:
        # Keep the same column ratio
        logo_col, title_col, info_col = st.columns([2, 3, 2])
        
        with logo_col:
            # Keep the same logo positioning
            st.markdown("""
            <div style="display: flex; align-items: center; height: 80px; padding-left: 20px; padding-top: 15px;">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT7zBX4VmSnYfD1Ismi7kk8MCDAGd-tPJrQwQ&s" alt="Logo" width="65" height="65" style="float: left;">
            </div>
            """, unsafe_allow_html=True)
        
        with title_col:
            # Keep title margin the same
            st.markdown("<h2 style='text-align: center; margin-top: 25px;'>Document Q&A System</h2>", unsafe_allow_html=True)
        
        with info_col:
            # Keep user info the same
            st.markdown(f"""
            <div style="text-align: right; margin-top: 20px; padding-right: 15px;">
                <div>Logged in: {st.session_state.username}</div>
                <div>Current time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Keep minimal spacing after header content
    st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
    
    # Create another row of columns for the buttons
    _, _, button_space, admin_col, logout_col = st.columns([1, 1, 1, 0.5, 0.5])
    
    with admin_col:
        if st.button("🔒 Admin", key="user_to_admin_switch", use_container_width=True):
            st.session_state.is_admin = True
            st.rerun()
    
    with logout_col:
        if st.button("🚪 Logout", key="user_logout_button", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    # Add a separator after the header
    st.markdown("<hr style='margin: 10px 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # The rest of your function remains unchanged
    # Main content section with chat on left and history on right
    chat_col, history_col = st.columns([3, 1])
    
    with chat_col:
        # Display current document info
        if "current_doc_state" in st.session_state and st.session_state.current_doc_state:
            file_path = st.session_state.current_doc_state.get("file_path", "Unknown")
            file_name = Path(file_path).name if file_path != "Unknown" else "Unknown"
            word_count = st.session_state.current_doc_state.get("word_count", 0)
            st.info(f"📄 Currently using: {file_name} ({word_count} words)")
        else:
            st.warning("Please upload a document in Admin section to start")
        
        # New chat button
        if st.button("🆕 New Chat"):
            start_new_chat()
            st.rerun()
        
        # Chat container
        chat_container = st.container(height=400, border=True)
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <span class="user-icon">👤</span>
                        <span class="message-content">{message["content"]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <span class="bot-icon">🤖</span>
                        <span class="message-content">{message["content"]}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Create a row with text input and send button
        input_col, button_col = st.columns([5, 1])
        
        with input_col:
            # Input area with standard callback 
            st.text_input("Type your message here...", 
                         key="message_input",
                         on_change=handle_message_submit,
                         label_visibility="collapsed")
        
        with button_col:
            # Add a send button
            if st.button("Send", key="send_button", use_container_width=True):
                # Only process if there's text in the input
                if "message_input" in st.session_state and st.session_state.message_input.strip():
                    # Call the same handler function that's used for the on_change event
                    handle_message_submit()
                    st.rerun()
    
    with history_col:
        st.markdown("### Chat History")
        st.caption("With date specific to user. Latest message on top.")
        
        # Show actual chat sessions in reverse order (newest first)
        max_sessions = min(10, len(st.session_state.chat_sessions))
        for i in range(max_sessions):
            session = st.session_state.chat_sessions[i]
            
            # Format timestamp
            timestamp = session.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    formatted_date = dt.strftime("%d-%m-%Y")
                except ValueError:
                    formatted_date = timestamp
            else:
                formatted_date = "Unknown date"
            
            # Get preview of last message
            last_message = session.get("last_message", "")
            preview = last_message[:30] + "..." if len(last_message) > 30 else last_message
            
            # Make each history item clickable to load that chat session
            if st.button(f"Chat {session.get('id', i+1)}", key=f"session_{i}", use_container_width=True):
                load_chat_session(session.get("id"))
                st.rerun()
            
            st.markdown(f"""
            <div class="history-item">
                <small>{preview}</small><br>
                <small style="color:#999;">{formatted_date}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("© 2025 Document Q&A System - All rights reserved")

# =========================
# ----- ADMIN INTERFACE ---
# =========================

def admin_interface():
    """Render the admin interface with improved header positioning."""
    # Significantly increase top padding to push everything further down
    st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
    header_container = st.container()
    
    with header_container:
        # Keep the same column ratio
        logo_col, title_col, info_col = st.columns([2, 3, 2])
        
        with logo_col:
            # Adjust logo positioning to be more to the left with padding
            st.markdown("""
            <div style="display: flex; align-items: center; height: 80px; padding-left: 20px; padding-top: 15px;">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT7zBX4VmSnYfD1Ismi7kk8MCDAGd-tPJrQwQ&s" alt="Logo" width="65" height="65" style="float: left;">
            </div>
            """, unsafe_allow_html=True)
        
        with title_col:
            # Keep title margin consistent with user interface
            st.markdown("<h2 style='text-align: center; margin-top: 25px;'>Admin Dashboard</h2>", unsafe_allow_html=True)
        
        with info_col:
            # Add padding to user info to match user interface
            st.markdown(f"""
            <div style="text-align: right; margin-top: 20px; padding-right: 15px;">
                <div>Logged in: {st.session_state.username}</div>
                <div>Current time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Keep minimal spacing after header content
    st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
    
    # Create another row of columns for the buttons
    _, _, button_space, user_col, logout_col = st.columns([1, 1, 1, 0.5, 0.5])
    
    with user_col:
        if st.button("👤 User", key="admin_to_user_switch", use_container_width=True):
            st.session_state.is_admin = False
            st.rerun()
    
    with logout_col:
        if st.button("🚪 Logout", key="admin_logout_button", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    # Add a separator after the header
    st.markdown("<hr style='margin: 10px 0; opacity: 0.3;'>", unsafe_allow_html=True)

    # The rest of your admin interface remains unchanged
    # Admin content
    st.markdown("### System Status")
    
    # Browse file button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        uploaded_file = st.file_uploader("Browse File", type=SUPPORTED_FORMATS)
        if uploaded_file:
            # First add the file to the table with processing status
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)
                
                # Update admin table - file is added but not yet processed
                current_date = get_formatted_date()
                st.session_state.admin_files_data["Date"].append(current_date)
                st.session_state.admin_files_data["File Name"].append(uploaded_file.name)
                st.session_state.admin_files_data["IsActive"].append("✗")
                st.session_state.admin_files_data["Ingested"].append("✗")
                
                # Display a processing message
                with st.spinner(f"Adding {uploaded_file.name} to the table..."):
                    time.sleep(0.5)  # Short delay to show the message
                st.rerun()
            
            # Now process the file if it hasn't been processed yet
            if "processed_documents" not in st.session_state:
                st.session_state.processed_documents = {}
                
            if uploaded_file.name not in st.session_state.processed_documents:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name
                
                with st.spinner(f"Processing {uploaded_file.name}... This may take a while"):
                    # Process the document
                    qa_state, error = process_document(file_path)
                    
                    if qa_state:
                        # Store the processed document
                        st.session_state.processed_documents[uploaded_file.name] = qa_state
                        
                        # Update the file's ingested status to ticked
                        file_index = st.session_state.admin_files_data["File Name"].index(uploaded_file.name)
                        st.session_state.admin_files_data["Ingested"][file_index] = "✓"
                        
                        # Set as current document
                        st.session_state.current_doc_state = qa_state
                        
                        # Update IsActive status - make this one active, others inactive
                        for i in range(len(st.session_state.admin_files_data["IsActive"])):
                            st.session_state.admin_files_data["IsActive"][i] = "✓" if i == file_index else "✗"
                        
                        st.success(f"Document processed successfully! Extracted {qa_state['word_count']} words and created {qa_state['chunk_count']} chunks.")
                        st.rerun()
                    else:
                        st.error(f"Failed to process document: {error}")
    
    # User files table
    st.markdown("### User Specific Uploaded Files")
    
    # Create DataFrame from the session state data
    df = pd.DataFrame(st.session_state.admin_files_data)
    
    # Add the column headers first
    if len(df) > 0:
        # Add columns with adjusted width
        header_cols = st.columns([1, 3, 0.7, 0.7])
        
        # Center all headers
        header_cols[0].markdown("<div style='text-align: center;'><strong>Date</strong></div>", unsafe_allow_html=True)
        header_cols[1].markdown("<div style='text-align: center;'><strong>File Name</strong></div>", unsafe_allow_html=True)
        header_cols[2].markdown("<div style='text-align: center;'><strong>IsActive</strong></div>", unsafe_allow_html=True)
        header_cols[3].markdown("<div style='text-align: center;'><strong>Ingested</strong></div>", unsafe_allow_html=True)
        
        # Add a separator below the headers
        st.markdown("<hr style='margin: 5px 0; opacity: 0.5;'>", unsafe_allow_html=True)
    
        # Add the rows
        for i in range(len(df)):
            file_name = df.iloc[i]["File Name"]
            is_ingested = df.iloc[i]["Ingested"] == "✓"
            is_active = df.iloc[i]["IsActive"] == "✓"
            
            # Create a row with columns
            cols = st.columns([1, 3, 0.7, 0.7])
            
            # Date column - left aligned
            cols[0].write(df.iloc[i]["Date"])
            
            # File name column with selection on click - center aligned
            if cols[1].button(f"{file_name}", key=f"file_{i}", use_container_width=True):
                if "processed_documents" in st.session_state and file_name in st.session_state.processed_documents:
                    # Set as current document
                    st.session_state.current_doc_state = st.session_state.processed_documents[file_name]
                    
                    # Update IsActive status - make this one active, others inactive
                    for j in range(len(st.session_state.admin_files_data["IsActive"])):
                        st.session_state.admin_files_data["IsActive"][j] = "✓" if j == i else "✗"
                    
                    st.success(f"Now using '{file_name}' for Q&A")
                    st.rerun()
                else:
                    st.warning(f"Document '{file_name}' not fully processed. Please re-upload.")
            
            # IsActive status - center align the checkmark/cross
            active_mark = "✓" if is_active else "✗"
            cols[2].markdown(f"<div style='text-align: center;'>{active_mark}</div>", unsafe_allow_html=True)
            
            # Ingested status with toggle on click - center align
            ingested_mark = "✓" if is_ingested else "✗"
            
            # Create a button with just the checkmark or cross and center it
            if cols[3].button(ingested_mark, key=f"ingest_{i}", use_container_width=True):
                if "processed_documents" in st.session_state and file_name in st.session_state.processed_documents:
                    # Toggle ingestion for this file and ensure others are not ingested
                    for j in range(len(st.session_state.admin_files_data["Ingested"])):
                        if j == i:
                            # Toggle this file
                            new_ingested_value = "✗" if is_ingested else "✓"
                            st.session_state.admin_files_data["Ingested"][j] = new_ingested_value
                        else:
                            # Un-ingest all others if this one is being ingested
                            if not is_ingested:  # Only if we're ingesting this file
                                st.session_state.admin_files_data["Ingested"][j] = "✗"
                    
                    # If ingesting this file, also make it active
                    if not is_ingested:
                        for j in range(len(st.session_state.admin_files_data["IsActive"])):
                            st.session_state.admin_files_data["IsActive"][j] = "✓" if j == i else "✗"
                        
                        # Set as current document
                        st.session_state.current_doc_state = st.session_state.processed_documents[file_name]
                    
                    st.rerun()
                else:
                    st.warning(f"Document '{file_name}' not fully processed. Cannot ingest.")
            
            # Add a separator between rows
            st.markdown("<hr style='margin: 5px 0; opacity: 0.3;'>", unsafe_allow_html=True)
    else:
        st.info("No files uploaded yet.")
    
    # Display current active file
    if "current_doc_state" in st.session_state and st.session_state.current_doc_state:
        file_path = st.session_state.current_doc_state.get("file_path", "Unknown")
        file_name = Path(file_path).name if file_path != "Unknown" else "Unknown"
        st.success(f"Current active file for Q&A: {file_name}")
    
    # Footer
    st.markdown("---")
    st.caption("© 2025 Document Q&A System - Admin Panel")
    
# =========================
# ----- LOGIN PAGE --------
# =========================

def login_page():
    """Render the login page."""
    # Current date and time display
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"Current Date and Time (UTC): {current_time}")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        display_logo()
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>Document Q&A System</h2>", unsafe_allow_html=True)
    
    with col3:
        st.write("")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        admin_check = st.checkbox("Login as Admin")
        
        if st.button("Login", type="primary", use_container_width=True):
            if username and password:
                # Update these variables with the login values
                st.session_state.username = username
                st.session_state.is_admin = admin_check
                st.session_state.logged_in = True  # <-- Set logged_in flag to True
                st.rerun()
            else:
                st.error("Please enter username and password")

# =========================
# ----- MAIN ROUTER -------
# =========================

def main():
    """Main application function."""
    # Apply custom CSS
    set_custom_css()
    
    # Check if user is logged in
    if not st.session_state.logged_in:
        login_page()
    else:
        # Show admin or user interface based on role
        if st.session_state.is_admin:
            admin_interface()
        else:
            user_interface()

if __name__ == "__main__":
    main()