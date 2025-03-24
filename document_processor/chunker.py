# document_processor/chunker.py

import re
from typing import Dict, List, Any

class DocumentChunker:
    """Splits documents into manageable chunks for embedding"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a list of documents and split them into chunks"""
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self._process_document(doc)
            all_chunks.extend(doc_chunks)
        
        return all_chunks
    
    def _process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single document and split it into chunks"""
        doc_chunks = []
        
        # Extract document metadata
        filename = document['filename']
        file_path = document['path']
        file_ext = document['extension']
        content = document['content']
        
        # Process content based on document type
        if file_ext == '.pdf':
            chunks = self._chunk_pdf(content, filename)
        elif file_ext in ['.docx', '.doc']:
            chunks = self._chunk_docx(content, filename)
        elif file_ext in ['.xlsx', '.xls', '.csv']:
            chunks = self._chunk_tabular(content, filename)
        elif file_ext == '.txt':
            chunks = self._chunk_text(content, filename)
        else:
            return []
        
        doc_chunks.extend(chunks)
        return doc_chunks
    
    def _chunk_text_with_overlap(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk of text
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Create chunk with metadata
            chunk = {
                'text': chunk_text,
                'metadata': metadata.copy()
            }
            
            # Add chunk offset information
            chunk['metadata']['chunk_start'] = start
            chunk['metadata']['chunk_end'] = min(end, len(text))
            
            chunks.append(chunk)
            
            # Move start position for next chunk, considering overlap
            start = end - self.chunk_overlap
            
            # If we can't advance further, break
            if start >= len(text) - self.chunk_overlap:
                break
        
        return chunks
    
    def _chunk_pdf(self, content: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """Chunk PDF content page by page"""
        chunks = []
        
        for page in content:
            page_num = page['page_num']
            page_text = page['text']
            
            metadata = {
                'filename': filename,
                'file_type': 'pdf',
                'page_num': page_num
            }
            
            page_chunks = self._chunk_text_with_overlap(page_text, metadata)
            chunks.extend(page_chunks)
        
        return chunks
    
    def _chunk_docx(self, content: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """Chunk Word document content paragraph by paragraph"""
        chunks = []
        current_chunk_text = ""
        current_paras = []
        
        for para in content:
            para_num = para['paragraph_num']
            para_text = para['text']
            
            # If adding this paragraph would exceed chunk size, create a new chunk
            if len(current_chunk_text) + len(para_text) > self.chunk_size:
                # Only create a chunk if we have accumulated text
                if current_chunk_text:
                    metadata = {
                        'filename': filename,
                        'file_type': 'docx',
                        'paragraph_range': f"{current_paras[0]}-{current_paras[-1]}"
                    }
                    
                    chunk = {
                        'text': current_chunk_text,
                        'metadata': metadata
                    }
                    
                    chunks.append(chunk)
                
                # Start a new chunk
                current_chunk_text = para_text
                current_paras = [para_num]
            else:
                # Add to current chunk
                if current_chunk_text:
                    current_chunk_text += " "
                current_chunk_text += para_text
                current_paras.append(para_num)
        
        # Add the last chunk if there's anything left
        if current_chunk_text:
            metadata = {
                'filename': filename,
                'file_type': 'docx',
                'paragraph_range': f"{current_paras[0]}-{current_paras[-1]}"
            }
            
            chunk = {
                'text': current_chunk_text,
                'metadata': metadata
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_tabular(self, content: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """Chunk tabular content (Excel, CSV)"""
        chunks = []
        
        for item in content:
            sheet_name = item.get('sheet_name', 'default')
            table_text = item['text']
            
            metadata = {
                'filename': filename,
                'file_type': 'tabular',
                'sheet_name': sheet_name
            }
            
            # Split the table text into chunks
            table_chunks = self._chunk_text_with_overlap(table_text, metadata)
            
            # Also store the original table structure separately
            if 'table' in item:
                table_data = item['table']
                for chunk in table_chunks:
                    chunk['metadata']['has_table'] = True
                    chunk['table_data'] = table_data
            
            chunks.extend(table_chunks)
        
        return chunks
    
    def _chunk_text(self, content: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """Chunk plain text content"""
        if not content or not content[0].get('text'):
            return []
        
        text = content[0]['text']
        
        metadata = {
            'filename': filename,
            'file_type': 'text'
        }
        
        chunks = self._chunk_text_with_overlap(text, metadata)
        return chunks