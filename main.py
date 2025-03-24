# main.py

import os
import json
import argparse
from typing import Dict, List, Any
import traceback

# Import our modules
from document_processor import DocumentLoader, DocumentChunker, DocumentEmbedder, VectorIndexer
from summarization import DocumentSummarizer
from data_visualization import DataExtractor, DataVisualizer
from chat_interface import ChatInterface, CLIChat

class DocumentAwareChat:
    """Main application coordinating all components"""
    
    def __init__(self, config_path: str):
        """
        Initialize the document-aware chat system
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.output_dir = self.config.get('output_dir', './outputs')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{self.output_dir}/summaries", exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            return {}
    
    def ingest_documents(self, documents_dir: str) -> bool:
        """
        Ingest documents from a directory
        
        Args:
            documents_dir: Directory containing documents to ingest
            
        Returns:
            Success status
        """
        try:
            print("\n===== Document Ingestion =====")
            
            # Load documents
            print("Loading documents...")
            loader = DocumentLoader()
            documents = loader.load_documents(documents_dir)
            
            print(f"Loaded {len(documents)} documents")
            
            # Chunk documents
            print("Chunking documents...")
            chunker = DocumentChunker()
            chunks = chunker.chunk_documents(documents)
            
            print(f"Created {len(chunks)} chunks")
            
            # Generate embeddings
            print("Generating embeddings...")
            embedder = DocumentEmbedder(self.config.get('openai', {}).get('api_key', ''))
            embedded_chunks = embedder.embed_chunks(chunks)
            
            print(f"Generated embeddings for {len(embedded_chunks)} chunks")
            
            # Store in vector database
            print("Storing in vector database...")
            indexer = VectorIndexer(
                self.config.get('vector_db', {}).get('api_key', ''),
                self.config.get('vector_db', {}).get('url', ''),
                self.config.get('vector_db', {}).get('collection', 'document_chunks')
            )
            
            success = indexer.index_chunks(embedded_chunks)
            
            if success:
                print("Documents ingested successfully")
            else:
                print("Failed to ingest documents")
            
            # Save ingestion metadata
            ingestion_metadata = {
                'documents_dir': documents_dir,
                'document_count': len(documents),
                'chunk_count': len(chunks),
                'embedded_chunk_count': len(embedded_chunks),
                'documents': [doc['filename'] for doc in documents]
            }
            
            with open(f"{self.output_dir}/ingestion_metadata.json", 'w') as f:
                json.dump(ingestion_metadata, f, indent=2)
            
            return success
            
        except Exception as e:
            print(f"Error ingesting documents: {str(e)}")
            traceback.print_exc()
            return False
    
    def generate_summaries(self) -> Dict[str, Any]:
        """
        Generate document summaries
        
        Returns:
            Dictionary of summaries
        """
        try:
            print("\n===== Summary Generation =====")
            
            # Initialize summarizer
            summarizer = DocumentSummarizer(self.config)
            
            # Retrieve chunks from vector database
            print("Retrieving document chunks...")
            indexer = VectorIndexer(
                self.config.get('vector_db', {}).get('api_key', ''),
                self.config.get('vector_db', {}).get('url', ''),
                self.config.get('vector_db', {}).get('collection', 'document_chunks')
            )
            
            # For simplicity, we'll use a sample of chunks
            # In a real system, you might want to retrieve all chunks or use a different approach
            
            # Generate summaries
            print("Generating summaries...")
            
            # Create a chat interface to retrieve chunks (using it for retrieval only)
            chat = ChatInterface(os.path.abspath(self.config_path))
            
            # Get a sample query to retrieve chunks
            query_processor = chat.query_processor
            vector_search = chat.vector_search
            
            # Generate embedding for a general query
            general_query = "What are the main topics and key information in these documents?"
            query_embedding = query_processor.generate_query_embedding(general_query)
            
            # Get a large number of chunks to summarize
            chunks = vector_search.search(query_embedding, top_k=50)
            
            # Generate summaries with different providers and types
            providers = ["openai", "anthropic", "google"]
            summary_types = ["general", "executive", "detailed"]
            
            available_providers = []
            if self.config.get('openai', {}).get('api_key'):
                available_providers.append("openai")
            if self.config.get('anthropic', {}).get('api_key'):
                available_providers.append("anthropic")
            if self.config.get('google', {}).get('api_key'):
                available_providers.append("google")
            
            summaries = {}
            
            for provider in available_providers:
                provider_summaries = {}
                
                for summary_type in summary_types:
                    print(f"Generating {summary_type} summary with {provider}...")
                    summary = summarizer.summarize_with_provider(chunks, provider, summary_type)
                    
                    provider_summaries[summary_type] = summary
                    
                    # Save summary to file
                    with open(f"{self.output_dir}/summaries/{provider}_{summary_type}.txt", 'w') as f:
                        f.write(summary.get('summary', ''))
                    
                    # Save citations to file
                    with open(f"{self.output_dir}/summaries/{provider}_{summary_type}_citations.json", 'w') as f:
                        json.dump(summary.get('citations', []), f, indent=2)
                
                summaries[provider] = provider_summaries
            
            print("Summaries generated successfully")
            
            return summaries
            
        except Exception as e:
            print(f"Error generating summaries: {str(e)}")
            traceback.print_exc()
            return {}
    
    def extract_visualizations(self) -> List[Dict[str, Any]]:
        """
        Extract data and generate visualizations
        
        Returns:
            List of visualization results
        """
        try:
            print("\n===== Data Extraction and Visualization =====")
            
            # Initialize extractor and visualizer
            extractor = DataExtractor(self.config.get('openai', {}).get('api_key', ''))
            visualizer = DataVisualizer(f"{self.output_dir}/visualizations")
            
            # Create a chat interface to retrieve chunks
            chat = ChatInterface(os.path.abspath(self.config_path))
            
            # Get chunks for different types of data
            query_processor = chat.query_processor
            vector_search = chat.vector_search
            
            print("Retrieving document chunks for data extraction...")
            
            # Queries for different types of data
            data_queries = {
                "financial": "financial data revenue profit margin",
                "metrics": "metrics statistics KPIs performance indicators",
                "trends": "trends growth changes over time increases decreases",
                "comparisons": "comparisons between different entities metrics comparison"
            }
            
            all_chunks = []
            for query_type, query in data_queries.items():
                query_embedding = query_processor.generate_query_embedding(query)
                chunks = vector_search.search(query_embedding, top_k=10)
                all_chunks.extend(chunks)
            
            # Remove duplicates
            unique_chunks = []
            seen_texts = set()
            for chunk in all_chunks:
                text = chunk.get('text', '')
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_chunks.append(chunk)
            
            print(f"Retrieved {len(unique_chunks)} unique chunks for data extraction")
            
            # Extract numerical data
            print("Extracting numerical data...")
            numerical_extractions = extractor.extract_numerical_data(unique_chunks)
            
            # Extract data with LLM
            extraction_types = ["financials", "metrics", "trends", "comparisons"]
            llm_extractions = []
            
            for extraction_type in extraction_types:
                print(f"Extracting {extraction_type} data...")
                extractions = extractor.extract_with_llm(unique_chunks, extraction_type)
                llm_extractions.extend(extractions)
            
            all_extractions = numerical_extractions + llm_extractions
            
            # Generate visualizations
            print("Generating visualizations...")
            visualizations = visualizer.generate_visualizations(all_extractions)
            
            # Save extractions and visualizations metadata
            extraction_metadata = {
                'extraction_count': len(all_extractions),
                'numerical_extraction_count': len(numerical_extractions),
                'llm_extraction_count': len(llm_extractions),
                'visualization_count': len(visualizations)
            }
            
            with open(f"{self.output_dir}/extraction_metadata.json", 'w') as f:
                json.dump(extraction_metadata, f, indent=2)
            
            # Save visualization index
            visualization_index = []
            for viz in visualizations:
                index_entry = {
                    'title': viz.get('title', ''),
                    'type': viz.get('type', ''),
                    'description': viz.get('description', ''),
                    'filename': viz.get('filename', '')
                }
                visualization_index.append(index_entry)
            
            with open(f"{self.output_dir}/visualizations/index.json", 'w') as f:
                json.dump(visualization_index, f, indent=2)
            
            print(f"Generated {len(visualizations)} visualizations")
            
            return visualizations
            
        except Exception as e:
            print(f"Error extracting and visualizing data: {str(e)}")
            traceback.print_exc()
            return []
    
    def start_chat(self):
        """Start the chat interface"""
        try:
            print("\n===== Chat Interface =====")
            
            cli_chat = CLIChat(os.path.abspath(self.config_path))
            cli_chat.start()
            
        except Exception as e:
            print(f"Error starting chat interface: {str(e)}")
            traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Document-Aware Chat System')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--documents', type=str, help='Path to documents directory')
    parser.add_argument('--ingest', action='store_true', help='Ingest documents')
    parser.add_argument('--summarize', action='store_true', help='Generate summaries')
    parser.add_argument('--visualize', action='store_true', help='Extract data and generate visualizations')
    parser.add_argument('--chat', action='store_true', help='Start chat interface')
    
    args = parser.parse_args()
    
    # Create system
    system = DocumentAwareChat(args.config)
    
    # Execute requested operations
    if args.ingest:
        if not args.documents:
            print("Error: --documents path is required for ingestion")
            return
        
        success = system.ingest_documents(args.documents)
        if not success:
            print("Document ingestion failed")
            return
    
    if args.summarize:
        summaries = system.generate_summaries()
        
        if not summaries:
            print("Summary generation failed")
    
    if args.visualize:
        visualizations = system.extract_visualizations()
        
        if not visualizations:
            print("Data extraction and visualization failed")
    
    if args.chat:
        system.start_chat()


if __name__ == "__main__":
    main()