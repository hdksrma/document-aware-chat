# document_processor/indexer.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Dict, List, Any
import uuid

class VectorIndexer:
    """Stores document chunk embeddings in Qdrant vector database"""
    
    def __init__(self, api_key=None, db_url="http://localhost:6333", collection="document_chunks"):
        """Initialize Qdrant client"""
        # For local Qdrant instance
        if "localhost" in db_url:
            self.client = QdrantClient(url=db_url)
        # For cloud Qdrant instance
        else:
            self.client = QdrantClient(url=db_url, api_key=api_key)
            
        self.collection = collection
    
    def create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection not in collection_names:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Collection '{self.collection}' created successfully")
            else:
                print(f"Collection '{self.collection}' already exists")
                
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
    
    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """Index chunks in the vector database"""
        # Create collection if it doesn't exist
        self.create_collection()
        
        # Filter out chunks without embeddings
        valid_chunks = [chunk for chunk in chunks if 'embedding' in chunk]
        print(f"Indexing {len(valid_chunks)} chunks with embeddings")
        
        # Process in batches
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i:i+batch_size]
            
            try:
                points = []
                
                for j, chunk in enumerate(batch):
                    # Generate a proper unsigned integer ID based on batch index and position
                    # This ensures IDs are proper unsigned integers as required by Qdrant
                    chunk_id = i * batch_size + j + 1  # Start from 1
                    
                    # Prepare metadata (convert any complex types to strings)
                    metadata = {
                        "text": chunk['text']
                    }
                    
                    # Add all metadata fields, ensuring they're serializable
                    for k, v in chunk['metadata'].items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            metadata[k] = v
                        else:
                            metadata[k] = str(v)
                    
                    # Add table data as JSON string if present
                    if 'table_data' in chunk:
                        import json
                        metadata['table_data'] = json.dumps(chunk['table_data'])
                    
                    # Create point
                    points.append(
                        models.PointStruct(
                            id=chunk_id,
                            vector=chunk['embedding'],
                            payload=metadata
                        )
                    )
                
                # Upload points
                self.client.upsert(
                    collection_name=self.collection,
                    points=points
                )
                
                print(f"Indexed batch {i//batch_size + 1} ({len(batch)} chunks)")
                
            except Exception as e:
                print(f"Error indexing batch {i//batch_size + 1}: {str(e)}")
                return False
        
        return True