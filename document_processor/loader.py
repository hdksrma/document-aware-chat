# document_processor/loader.py

import os
import PyPDF2
import docx
import pandas as pd
from typing import Dict, List, Any

class DocumentLoader:
    """Loads and processes documents (PDF, Word, etc.)"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.doc': self._load_docx,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.csv': self._load_csv,
            '.txt': self._load_text
        }
    
    def load_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Load all supported documents from a directory"""
        documents = []
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in self.supported_formats:
                try:
                    print(f"Loading {filename}...")
                    doc_content = self.supported_formats[file_ext](file_path)
                    
                    if doc_content:
                        documents.append({
                            'filename': filename,
                            'path': file_path,
                            'extension': file_ext,
                            'content': doc_content
                        })
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        return documents
    
    def _load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Load PDF document and extract text by page"""
        pages = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    pages.append({
                        'page_num': page_num + 1,
                        'text': text
                    })
        
        return pages
    
    def _load_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Load Word document and extract text"""
        doc = docx.Document(file_path)
        paragraphs = []
        
        for para_num, para in enumerate(doc.paragraphs):
            if para.text.strip():
                paragraphs.append({
                    'paragraph_num': para_num + 1,
                    'text': para.text
                })
        
        return paragraphs
    
    def _load_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """Load Excel document and extract tables by sheet"""
        sheets = []
        
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if not df.empty:
                sheets.append({
                    'sheet_name': sheet_name,
                    'table': df.to_dict(orient='records'),
                    'text': df.to_string()
                })
        
        return sheets
    
    def _load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CSV document and extract table"""
        df = pd.read_csv(file_path)
        if not df.empty:
            return [{
                'table': df.to_dict(orient='records'),
                'text': df.to_string()
            }]
        return []
    
    def _load_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Load text document"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            
        return [{
            'text': text
        }]