# chat_interface/vector_search.py
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Dict, List, Any

class VectorSearch:
    """Vector search from document embeddings in Qdrant vector database"""
    
    def __init__(self, api_key=None, db_url="http://localhost:6333", collection="document_chunks"):
        """
        Initialize Qdrant client
        
        Args:
            api_key: API key (not needed for local Qdrant)
            db_url: URL to Qdrant server
            collection: Collection name
        """
        # For local Qdrant instance
        if "localhost" in db_url:
            self.client = QdrantClient(url=db_url)
        # For cloud Qdrant instance
        else:
            self.client = QdrantClient(url=db_url, api_key=api_key)
            
        self.collection = collection
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Number of results to return
            
        Returns:
            List of document chunks with similarity scores
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection not in collection_names:
                print(f"Collection '{self.collection}' does not exist")
                return []
            
            # Search for similar vectors
            search_results = self.client.search(
                collection_name=self.collection,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Process results
            results = []
            for match in search_results:
                payload = match.payload
                
                # Extract text from payload
                text = payload.pop('text', '')
                
                # Parse table data if present
                if 'table_data' in payload and isinstance(payload['table_data'], str):
                    try:
                        payload['table_data'] = json.loads(payload['table_data'])
                    except:
                        print(f"Failed to parse table data: {payload['table_data'][:50]}...")
                
                # Create result object
                result = {
                    'text': text,
                    'metadata': payload,
                    'score': match.score
                }
                
                results.append(result)
            
            print(f"Found {len(results)} matches for query")
            return results
            
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            return []