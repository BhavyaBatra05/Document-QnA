from pathlib import Path
from typing import Dict, Any

# Demo documents configuration with relative paths
DEMO_DOCUMENTS = {
    # "java_intro": {
    #     "filename": "Java's Role in AIML.docx",
    #     "display_name": "Java in AI/ML - Introduction (DOCX)",
    #     "file_path": Path("./demo_files/Java's Role in AIML.docx"),
    #     "file_type": "docx",
    #     "has_visuals": True,
    #     "sample_queries": [
    #         "What is Java's role in AI/ML?",
    #         "What does the table say?",
    #         "List some Java's applications.",
    #     ]
    # },
    
    # "java_comprehensive": {
    #     "filename": "Java's Role in AIML.pdf",
    #     "display_name": "Java in AI/ML - Complete Guide (PDF)",
    #     "file_path": Path("./demo_files/Java's Role in AIML.pdf"),
    #     "file_type": "pdf",
    #     "has_visuals": True,
    #     "sample_queries": [
    #         "What are the top Java ML libraries?",
    #         "Give me the complete schedule week-wise.",
    #         "What does the visual say?",
    #         "List the machine learning frameworks for Java"
    #     ]
    # },
    
    "sales_report": {
        "filename": "sample w graph.pdf",
        "display_name": "Sales Performance Report 2024 (PDF)",
        "file_path": Path("./demo_files/sample w graph.pdf"),
        "file_type": "pdf", 
        "has_visuals": True,
        "sample_queries": [
            "What are the 2024 sales figures by region?",
            "Describe the bar chart in the document",
            "Which region has the highest sales?",
            "Compare North America vs Asia performance"
        ]
    }
}

def get_demo_file_path(doc_key: str) -> str:
    """Get file path for demo document."""
    doc_info = DEMO_DOCUMENTS.get(doc_key)
    if not doc_info:
        raise KeyError(f"Demo document key '{doc_key}' not found")
    
    file_path = doc_info["file_path"]
    if file_path.exists():
        return str(file_path)
    else:
        raise FileNotFoundError(f"Demo file not found: {file_path}")

def get_demo_document_info(doc_key: str) -> Dict[str, Any]:
    """Get demo document information."""
    return DEMO_DOCUMENTS.get(doc_key, {})

def list_available_demos() -> Dict[str, str]:
    """List all available demo documents."""
    return {key: info["display_name"] for key, info in DEMO_DOCUMENTS.items()}

def check_demo_files_exist() -> Dict[str, bool]:
    """Check which demo files exist."""
    return {
        key: info["file_path"].exists() 
        for key, info in DEMO_DOCUMENTS.items()
    }
