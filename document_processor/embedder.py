import openai
from typing import Dict, List, Any

class DocumentEmbedder:
    """Generates embeddings for document chunks using OpenAI's latest embedding model."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.Client(api_key=api_key)
        self.model = model
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 20) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of text chunks in batches."""
        embedded_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_texts = [chunk['text'] for chunk in batch]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                
                for j, chunk in enumerate(batch):
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding['embedding'] = batch_embeddings[j]
                    embedded_chunks.append(chunk_with_embedding)
                
            except openai.OpenAIError as e:
                print(f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
                embedded_chunks.extend(batch)
        
        return embedded_chunks
