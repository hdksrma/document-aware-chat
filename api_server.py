# api_server.py

from flask import Flask, request, jsonify, send_from_directory
import os
import json
from werkzeug.utils import secure_filename
import traceback

# Import our modules
from document_processor import DocumentLoader, DocumentChunker, DocumentEmbedder, VectorIndexer
from chat_interface import ChatInterface
from summarization import DocumentSummarizer
from data_visualization import DataExtractor, DataVisualizer

app = Flask(__name__, static_folder='./frontend/build')

# Initialize with configuration
CONFIG_PATH = os.environ.get('CONFIG_PATH', 'config.json')
OUTPUT_DIR = './outputs'
UPLOAD_DIR = './uploads'

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/summaries", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load config
def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return {}

config = load_config()

# Initialize chat interface
chat_interface = ChatInterface(CONFIG_PATH)

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat interaction"""
    try:
        data = request.json
        message = data.get('message', '')
        provider = data.get('provider', 'openai')
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        # Process the message
        response = chat_interface.process_message(message, provider)
        
        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """API endpoint to get list of ingested documents"""
    try:
        # Check if ingestion metadata exists
        metadata_path = f"{OUTPUT_DIR}/ingestion_metadata.json"
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return jsonify({"documents": metadata.get('documents', [])})
        
        # If no metadata, check upload directory
        documents = []
        for filename in os.listdir(UPLOAD_DIR):
            if os.path.isfile(os.path.join(UPLOAD_DIR, filename)):
                documents.append(filename)
        
        return jsonify({"documents": documents})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/summaries', methods=['GET'])
def get_summaries():
    """API endpoint to get document summaries"""
    try:
        summaries = {}
        summaries_dir = f"{OUTPUT_DIR}/summaries"
        
        if not os.path.exists(summaries_dir):
            return jsonify({"summaries": {}})
        
        # Read all summary files
        for provider in ['openai', 'anthropic', 'google']:
            provider_summaries = {}
            
            for summary_type in ['general', 'executive', 'detailed']:
                summary_path = f"{summaries_dir}/{provider}_{summary_type}.txt"
                citation_path = f"{summaries_dir}/{provider}_{summary_type}_citations.json"
                
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary_text = f.read()
                    
                    citations = []
                    if os.path.exists(citation_path):
                        with open(citation_path, 'r') as f:
                            citations = json.load(f)
                    
                    provider_summaries[summary_type] = {
                        "summary": summary_text,
                        "citations": citations
                    }
            
            if provider_summaries:
                summaries[provider] = provider_summaries
        
        return jsonify({"summaries": summaries})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visualizations', methods=['GET'])
def get_visualizations():
    """API endpoint to get data visualizations"""
    try:
        visualizations = []
        viz_dir = f"{OUTPUT_DIR}/visualizations"
        
        if not os.path.exists(viz_dir):
            return jsonify({"visualizations": []})
        
        # Check for visualization index
        index_path = f"{viz_dir}/index.json"
        
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                visualization_index = json.load(f)
                return jsonify({"visualizations": visualization_index})
        
        # If no index, list visualization files
        for filename in os.listdir(viz_dir):
            if filename.endswith('.html'):
                visualizations.append({
                    "title": filename.split('.')[0].replace('_', ' ').title(),
                    "filename": filename,
                    "type": "unknown"
                })
        
        return jsonify({"visualizations": visualizations})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve visualization files"""
    return send_from_directory(f"{OUTPUT_DIR}/visualizations", filename)

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """API endpoint to upload documents"""
    try:
        if 'documents' not in request.files:
            return jsonify({"error": "No documents part"}), 400
        
        uploaded_files = request.files.getlist('documents')
        
        if not uploaded_files or uploaded_files[0].filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save uploaded files
        uploaded = []
        for file in uploaded_files:
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_DIR, filename)
                file.save(file_path)
                uploaded.append(filename)
        
        # Process the documents
        if uploaded:
            # Initialize components
            loader = DocumentLoader()
            chunker = DocumentChunker()
            embedder = DocumentEmbedder(config.get('openai', {}).get('api_key', ''))
            indexer = VectorIndexer(
                config.get('vector_db', {}).get('api_key', ''),
                config.get('vector_db', {}).get('url', ''),
                config.get('vector_db', {}).get('collection', 'document_chunks')
            )
            
            # Load and process documents
            documents = loader.load_documents(UPLOAD_DIR)
            chunks = chunker.chunk_documents(documents)
            embedded_chunks = embedder.embed_chunks(chunks)
            indexer.index_chunks(embedded_chunks)
            
            # Update ingestion metadata
            metadata_path = f"{OUTPUT_DIR}/ingestion_metadata.json"
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['documents'].extend(uploaded)
                metadata['document_count'] += len(uploaded)
                metadata['chunk_count'] += len(chunks)
                metadata['embedded_chunk_count'] += len(embedded_chunks)
            else:
                metadata = {
                    'documents_dir': UPLOAD_DIR,
                    'document_count': len(documents),
                    'chunk_count': len(chunks),
                    'embedded_chunk_count': len(embedded_chunks),
                    'documents': [doc['filename'] for doc in documents]
                }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return jsonify({"message": "Files uploaded successfully", "uploaded": uploaded})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/generate-summaries', methods=['POST'])
def generate_summaries():
    """API endpoint to generate document summaries"""
    try:
        print("Starting summary generation process...")
        
        # Initialize summarizer
        summarizer = DocumentSummarizer(config)
        
        # Create a chat interface to retrieve chunks
        print("Creating chat interface...")
        chat = ChatInterface(CONFIG_PATH)
        
        # Get a sample query to retrieve chunks
        print("Retrieving document chunks...")
        query_processor = chat.query_processor
        vector_search = chat.vector_search
        
        # Generate embedding for a general query
        general_query = "What are the main topics and key information in these documents?"
        print(f"Generating embedding for query: '{general_query}'")
        query_embedding = query_processor.generate_query_embedding(general_query)
        
        # Get a large number of chunks to summarize
        print("Searching for chunks...")
        chunks = vector_search.search(query_embedding, top_k=50)
        print(f"Retrieved {len(chunks)} chunks")
        
        if not chunks:
            print("No document chunks found!")
            return jsonify({"success": False, "error": "No document data found. Please upload documents first."}), 400
        
        # Generate summaries with different providers and types
        providers = ["openai", "anthropic", "google"]
        summary_types = ["general", "executive", "detailed"]
        
        available_providers = []
        if config.get('openai', {}).get('api_key'):
            available_providers.append("openai")
        if config.get('anthropic', {}).get('api_key'):
            available_providers.append("anthropic")
        if config.get('google', {}).get('api_key'):
            available_providers.append("google")
        
        print(f"Available providers: {available_providers}")
        
        if not available_providers:
            print("No LLM providers configured!")
            return jsonify({"success": False, "error": "No LLM providers configured. Please add API keys to config.json"}), 400
        
        summaries = {}
        
        for provider in available_providers:
            provider_summaries = {}
            
            for summary_type in summary_types:
                print(f"Generating {summary_type} summary with {provider}...")
                summary = summarizer.summarize_with_provider(chunks, provider, summary_type)
                
                if 'error' in summary:
                    print(f"Error generating summary: {summary['error']}")
                    continue
                
                print(f"Summary generated successfully ({len(summary.get('summary', ''))} chars)")
                provider_summaries[summary_type] = summary
                
                # Save summary to file
                summary_dir = f"{OUTPUT_DIR}/summaries"
                os.makedirs(summary_dir, exist_ok=True)
                
                with open(f"{summary_dir}/{provider}_{summary_type}.txt", 'w') as f:
                    f.write(summary.get('summary', ''))
                
                # Save citations to file
                with open(f"{summary_dir}/{provider}_{summary_type}_citations.json", 'w') as f:
                    json.dump(summary.get('citations', []), f, indent=2)
            
            if provider_summaries:
                summaries[provider] = provider_summaries
        
        print(f"Summaries generated for providers: {list(summaries.keys())}")
        
        # For debugging, print some info about the returned structure
        for provider, provider_data in summaries.items():
            print(f"Provider {provider}: {list(provider_data.keys())}")
            
        # Return at least empty structure if no summaries were generated
        if not summaries:
            print("Warning: No summaries were generated!")
            # Return empty structure that matches what frontend expects
            return jsonify({"success": True, "message": "No summaries could be generated.", "summaries": {"openai": {"general": {"summary": "No summary could be generated.", "citations": []}}}})
        
        print("Successfully returning summaries to frontend.")
        return jsonify({"success": True, "message": "Summaries generated successfully", "summaries": summaries})
        
    except Exception as e:
        print(f"Error generating summaries: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate-visualizations', methods=['POST'])
def generate_visualizations():
    """API endpoint to generate data visualizations with improved extraction and visualization"""
    try:
        print("\n===== Starting Enhanced Data Extraction and Visualization =====")
        
        # Initialize extractor and visualizer with enhanced implementations
        extractor = DataExtractor(config.get('openai', {}).get('api_key', ''))
        visualizer = DataVisualizer(f"{OUTPUT_DIR}/visualizations")
        
        # Create a chat interface to retrieve chunks
        chat = ChatInterface(CONFIG_PATH)
        
        # Get chunks for different types of data
        query_processor = chat.query_processor
        vector_search = chat.vector_search
        
        print("Retrieving document chunks for data extraction...")
        
        # Enhanced data queries to capture more patterns
        data_queries = {
            "financial": "financial data revenue profit margin budget costs expenses sales quarterly annual",
            "metrics": "metrics statistics KPIs performance indicators measurements benchmarks scores ratings",
            "trends": "trends growth changes over time increases decreases patterns evolution development progress",
            "comparisons": "comparisons differences between vs versus contrast compare competitive analysis",
            "time_series": "time series monthly quarterly yearly timeline chronological sequence historical data",
            "percentages": "percentages ratios proportions share allocation distribution percentage"
        }
        
        all_chunks = []
        for query_type, query in data_queries.items():
            print(f"Searching for {query_type} data...")
            query_embedding = query_processor.generate_query_embedding(query)
            chunks = vector_search.search(query_embedding, top_k=15)  # Increased from 10 to 15
            print(f"Found {len(chunks)} chunks for {query_type}")
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return jsonify({"success": False, "error": "No document data found. Please upload documents first."}), 400
        
        # Remove duplicates
        unique_chunks = []
        seen_texts = set()
        for chunk in all_chunks:
            text = chunk.get('text', '')
            if text not in seen_texts:
                seen_texts.add(text)
                unique_chunks.append(chunk)
        
        print(f"Processing {len(unique_chunks)} unique chunks for visualization")
        
        # Extract numerical data with enhanced pattern recognition
        print("Extracting numerical patterns from text...")
        numerical_extractions = extractor.extract_numerical_data(unique_chunks)
        print(f"Extracted {len(numerical_extractions)} numerical patterns")
        
        # Extract data with LLM using auto-detection
        print("Extracting structured data with LLM...")
        llm_extractions = extractor.extract_with_llm(unique_chunks, "auto")
        print(f"Extracted {len(llm_extractions)} structured data elements with auto-detection")
        
        # Also try specific extraction types
        extraction_types = ["financials", "metrics", "trends", "comparisons"]
        for extraction_type in extraction_types:
            print(f"Extracting {extraction_type} data...")
            extractions = extractor.extract_with_llm(unique_chunks, extraction_type)
            llm_extractions.extend(extractions)
        
        all_extractions = numerical_extractions + llm_extractions
        print(f"Total data extractions: {len(all_extractions)}")
        
        if not all_extractions:
            return jsonify({"success": False, "error": "No structured data found in documents to visualize."}), 400
        
        # Generate visualizations with enhanced capabilities
        print("Generating visualizations...")
        visualizations = visualizer.generate_visualizations(all_extractions)
        
        # Save visualization index with more metadata
        visualization_index = []
        for viz in visualizations:
            index_entry = {
                'title': viz.get('title', ''),
                'type': viz.get('type', ''),
                'description': viz.get('description', ''),
                'filename': viz.get('filename', ''),
                'has_stats': 'stats' in viz
            }
            
            # Add additional metadata for frontend
            if 'metrics' in viz:
                index_entry['metrics'] = viz['metrics']
            
            if 'data' in viz:
                # Include sample data but limit size
                sample_data = viz['data']
                if isinstance(sample_data, dict) and len(str(sample_data)) > 1000:
                    # Truncate large data objects
                    keys = list(sample_data.keys())[:5]
                    index_entry['sample_data'] = {k: sample_data[k] for k in keys}
                elif isinstance(sample_data, list) and len(sample_data) > 5:
                    index_entry['sample_data'] = sample_data[:5]
                else:
                    index_entry['sample_data'] = sample_data
            
            visualization_index.append(index_entry)
        
        with open(f"{OUTPUT_DIR}/visualizations/index.json", 'w') as f:
            json.dump(visualization_index, f, indent=2)
        
        print(f"Successfully generated {len(visualizations)} visualizations")
        
        # Group visualizations by type for better organization
        viz_by_type = {}
        for viz in visualization_index:
            viz_type = viz.get('type', 'other')
            if viz_type not in viz_by_type:
                viz_by_type[viz_type] = []
            viz_by_type[viz_type].append(viz)
        
        return jsonify({
            "success": True, 
            "message": f"Generated {len(visualizations)} visualizations",
            "visualizations": visualization_index,
            "visualization_types": list(viz_by_type.keys()),
            "visualizations_by_type": viz_by_type
        })
        
    except Exception as e:
        print(f"Error in visualization generation: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/api/generate-smart-visualizations', methods=['POST'])
def generate_smart_visualizations():
    """API endpoint to generate intelligent data visualizations"""
    try:
        print("Starting smart visualization generation...")
        
        # Import the SmartVisualizer
        from data_visualization.smart_visualizer import SmartVisualizer
        
        # Initialize visualizer
        visualizer = SmartVisualizer(config, f"{OUTPUT_DIR}/visualizations")
        
        # Create a chat interface to retrieve chunks
        print("Creating chat interface...")
        chat = ChatInterface(CONFIG_PATH)
        
        # Get a sample query to retrieve chunks with potential data for visualization
        print("Retrieving document chunks...")
        query_processor = chat.query_processor
        vector_search = chat.vector_search
        
        # Generate embedding for a query targeting data
        data_query = "financial data metrics statistics numbers percentages trends comparisons figures data tables"
        print(f"Generating embedding for data query...")
        query_embedding = query_processor.generate_query_embedding(data_query)
        
        # Get chunks with potential data for visualization
        print("Searching for data-rich chunks...")
        chunks = vector_search.search(query_embedding, top_k=30)
        print(f"Retrieved {len(chunks)} data-rich chunks")
        
        if not chunks:
            print("No document chunks found!")
            return jsonify({"success": False, "error": "No document data found. Please upload documents first."}), 400
        
        # Generate intelligent visualizations
        print("Generating smart visualizations...")
        visualizations = visualizer.generate_smart_visualizations(chunks)
        print(f"Generated {len(visualizations)} smart visualizations")
        
        if not visualizations:
            print("No visualizations could be generated!")
            return jsonify({
                "success": False, 
                "error": "No suitable data found for visualization. Try uploading documents with numerical data."
            }), 400
        
        # Prepare visualization data for frontend
        visualization_data = []
        for viz in visualizations:
            visualization_data.append({
                'title': viz.get('title', ''),
                'type': viz.get('type', ''),
                'description': viz.get('description', ''),
                'filename': viz.get('filename', '')
            })
        
        return jsonify({
            "success": True,
            "message": f"Generated {len(visualizations)} intelligent visualizations",
            "visualizations": visualization_data
        })
        
    except Exception as e:
        print(f"Error generating smart visualizations: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)