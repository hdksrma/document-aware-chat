# chat_interface/cli_chat.py

import json
import os
from typing import Dict, List, Any

from .query_processor import QueryProcessor
from .vector_search import VectorSearch
from .response_generator import ResponseGenerator

class ChatInterface:
    """Main chat interface combining all components"""
    
    def __init__(self, config_path: str):
        """
        Initialize chat interface with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self._initialize_components()
        self.conversation_history = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            return {}
    
    def _initialize_components(self):
        """Initialize all required components"""
        # Query processor
        openai_api_key = self.config.get('openai', {}).get('api_key', '')
        self.query_processor = QueryProcessor(openai_api_key)
        
        # Vector search
        vector_db_api_key = self.config.get('vector_db', {}).get('api_key', '')
        vector_db_url = self.config.get('vector_db', {}).get('url', '')
        vector_db_collection = self.config.get('vector_db', {}).get('collection', 'document_chunks')
        self.vector_search = VectorSearch(vector_db_api_key, vector_db_url, vector_db_collection)
        
        # Response generator
        self.response_generator = ResponseGenerator(self.config)
    
    def process_message(self, 
                       message: str, 
                       provider: str = "openai", 
                       top_k: int = 5) -> Dict[str, Any]:
        """
        Process a user message and generate a response
        
        Args:
            message: User query
            provider: LLM provider to use
            top_k: Number of context items to retrieve
            
        Returns:
            Response with citations
        """
        # Process the query
        processed_query = self.query_processor.process_query(message)
        
        # Search for relevant context
        search_results = self.vector_search.search(processed_query['embedding'], top_k)
        
        # Generate response
        response = self.response_generator.generate_response(
            message, 
            search_results, 
            provider
        )
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': response.get('response', ''),
            'citations': response.get('citations', [])
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []


class CLIChat:
    """Command Line Interface for the chat system"""
    
    def __init__(self, config_path: str):
        """
        Initialize CLI chat interface
        
        Args:
            config_path: Path to configuration file
        """
        self.chat_interface = ChatInterface(config_path)
    
    def start(self):
        """Start the CLI chat interface"""
        print("Document-Aware Chat System")
        print("Type 'exit' to end the conversation")
        print("Type 'clear' to clear the conversation history")
        print("----------------------------------------")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                self.chat_interface.clear_conversation_history()
                print("Conversation history cleared")
                continue
            
            print("\nProcessing...")
            
            response = self.chat_interface.process_message(user_input)
            
            if 'error' in response:
                print(f"Error: {response['error']}")
                continue
            
            print("\nAssistant:")
            print(response['response'])
            
            # Print citation information
            if response.get('citations'):
                print("\nSources:")
                for i, citation in enumerate(response['citations']):
                    print(f"  {i+1}. {', '.join([f'{k}: {v}' for k, v in citation.items() if k != 'score'])}")