import openai
from typing import Dict, List, Any

class QueryProcessor:
    """Processes user queries and generates embeddings using OpenAI's latest API."""
    
    def __init__(self, openai_api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.Client(api_key=openai_api_key)
        self.model = model
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query
        
        Args:
            query: User query text
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[query]
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except openai.OpenAIError as e:
            print(f"Error generating query embedding: {str(e)}")
            return [0] * 1536  # Fallback vector
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with query and embedding
        """
        embedding = self.generate_query_embedding(query)
        
        return {
            'query': query,
            'embedding': embedding
        }
