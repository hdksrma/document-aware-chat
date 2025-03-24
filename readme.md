# Document-Aware Chat System with Source Citation

A comprehensive AI-powered system for ingesting heterogeneous documents, generating cross-document summaries, extracting structured data with intelligent visualizations, and enabling contextual Q&A with source attribution.

## Key Features

- **Multi-format Document Processing**: Support for PDF, Word, Excel, CSV, and text files
- **Contextual Q&A with Source Attribution**: Ask questions about your documents and get answers with specific citations
- **Cross-Document Summarization**: Generate summaries across multiple documents with different levels of detail
- **Intelligent Data Visualization**: Extract and visualize structured data from unstructured text using AI
- **Multiple LLM Support**: Integration with OpenAI, Anthropic Claude, and Google Gemini models
- **Vector Search**: Semantic search using embeddings stored in Qdrant vector database
- **Web Interfaces**: An interactive web interface.

## System Architecture

```
┌─────────────────┐         ┌─────────────────┐        ┌──────────────────┐
│                 │         │                 │        │                  │
│    Documents    │────────▶│    Processing   │───────▶│  Vector Storage  │
│  PDF/Word/Excel │         │ Pipeline (RAG)  │        │      Qdrant      │
│                 │         │                 │        │                  │
└─────────────────┘         └─────────────────┘        └──────────────────┘
                                                                │
                                                                ▼
┌─────────────────┐         ┌─────────────────┐        ┌──────────────────┐
│                 │         │                 │        │                  │
│  React Frontend │◀────────│  Flask Backend  │◀───────│   LLM Services   │
│    UI/UX        │         │  API Endpoints  │        │OpenAI/Claude/etc │
│                 │         │                 │        │                  │
└─────────────────┘         └─────────────────┘        └──────────────────┘
```

## Prerequisites

- Python 3.8 <= Python Version < Python 3.11 
- Node.js 14+ (for React frontend)
- API keys for:
  - OpenAI (required for embeddings and LLM)
  - Qdrant vector database (local installation or cloud)
  - Anthropic (optional for Claude)
  - Google AI (optional for Gemini)

## Installation

### 1. Set Up Environment

```bash
# Clone repository (if using git)
git clone https://github.com/hdksrma/document-aware-chat.git
cd document-aware-chat

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `config.json` file in the root directory:

```json
{
  "openai": {
    "api_key": "your-openai-api-key",
    "model": "gpt-4-turbo"
  },
  "anthropic": {
    "api_key": "your-anthropic-api-key",
    "model": "claude-3-sonnet-20240229"
  },
  "google": {
    "api_key": "your-google-ai-api-key",
    "model": "gemini-1.5-pro"
  },
  "vector_db": {
    "api_key": "",  // Not needed for local Qdrant; needed only for cloud instance
    "url": "http://localhost:6333",
    "collection": "document_chunks"
  },
  "output_dir": "./outputs"
}
```

### 3. Set Up Qdrant Vector Database Locally

#### First, download the latest Qdrant image from Dockerhub:
```bash
docker pull qdrant/qdrant
```
#### Then, run the service:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 4. Set Up the Frontend

```bash
cd frontend
npm install
npm run build  # Builds the production version
cd ..
```

### 5. Create Required Directories

```bash
mkdir -p uploads outputs/visualizations outputs/summaries
```

## Usage

### Starting the Web Interface

```bash
# Start the API server
python api_server.py
```

Access the web interface at http://localhost:5000


## Using the Web Interface

### 1. Document Upload
- Click the upload button in the sidebar
- Select one or more documents (PDF, Word, Excel, etc.)
- Wait for the upload and processing to complete

### 2. Chat with Your Documents
- Type questions about your documents in the chat input
- Receive answers with source citations highlighting where the information came from
- Ask follow-up questions to explore topics in more depth

### 3. Generate Summaries
- Navigate to the "Summaries" tab
- Click "Generate Summaries" to create summaries of your documents
- Choose between different summary types:
  - **General**: Balanced overview of document content
  - **Executive**: Concise summary focused on key points
  - **Detailed**: Comprehensive summary with specific details

### 4. Create Visualizations
- Navigate to the "Visualizations" tab
- Click "Smart Visualizations" to generate AI-driven charts
- The system automatically:
  - Extracts structured data from your documents
  - Creates appropriate visualizations based on data type
  - Provides explanations of what each visualization represents

## File Structure

```
document-aware-chat/
├── document_processor/          # Document processing components
│   ├── __init__.py
│   ├── loader.py                # Document loading
│   ├── chunker.py               # Text chunking
│   ├── embedder.py              # Embedding generation
│   └── indexer.py               # Vector database indexing
│
├── summarization/               # Summarization components
│   ├── __init__.py
│   └── summarizer.py            # Document summarization
│
├── data_visualization/          # Data visualization components
│   ├── __init__.py
│   ├── extractor.py             # Data extraction
│   ├── visualizer.py            # Basic visualization generation
│   └── smart_visualizer.py      # LLM-driven smart visualizations
│
├── chat_interface/              # Chat components
│   ├── __init__.py
│   ├── query_processor.py       # Query processing
│   ├── vector_search.py         # Vector search
│   ├── response_generator.py    # Response generation
│   └── cli_chat.py              # Command-line interface
│
├── frontend/                    # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── App.js               # Main React application
│   │   └── App.css              # CSS styling
│   └── package.json
│
├── main.py                      # Main application entrypoint
├── api_server.py                # Flask API server for web interface
├── config.json                  # Configuration file
├── requirements.txt             # Python dependencies
│
├── uploads/                     # Directory for uploaded documents
└── outputs/                     # Output directories
    ├── summaries/               # Generated summaries
    └── visualizations/          # Generated visualizations
```

## Troubleshooting

### Common Issues


1. **Vector Database Connection**
   - Error: `Failed to connect to vector database`
   - Fix: Make sure Qdrant is running or verify cloud credentials

2. **Document Processing Errors**
   - Error: `Error loading document`
   - Fix: Check file isn't password protected or corrupted

4. **Missing Dependencies**
   - Error: `ModuleNotFoundError`
   - Fix: Install the required package with `pip install [package-name]`


## Customization Options

### Vector Database
The system uses Qdrant by default but can be modified to support:
- Pinecone
- Weaviate
- Milvus
- ChromaDB

### LLM Providers
Configure one or more of:
- OpenAI (GPT-4, etc.)
- Anthropic (Claude)
- Google (Gemini)

### Document Types
The system supports:
- PDF documents
- Word documents (DOCX)
- Excel spreadsheets (XLSX, XLS)
- CSV files
- Plain text files

## License

[MIT License](LICENSE)