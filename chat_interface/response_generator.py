import openai
import anthropic
import json
import re
from typing import Dict, List, Any

class ResponseGenerator:
    """Generates responses with source attribution using LLMs"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with API keys for different providers
        
        Args:
            config: Dictionary containing API keys and configurations
        """
        self.config = config
        self.initialize_providers()
    
    def initialize_providers(self):
        """Initialize API clients for different LLM providers"""
        # Initialize OpenAI
        if 'openai' in self.config and self.config['openai'].get('api_key'):
            self.openai_client = openai.Client(api_key=self.config['openai']['api_key'])
            self.openai_available = True
        else:
            self.openai_available = False
            
        # Initialize Anthropic
        if 'anthropic' in self.config and self.config['anthropic'].get('api_key'):
            self.anthropic_client = anthropic.Anthropic(api_key=self.config['anthropic']['api_key'])
            self.anthropic_available = True
        else:
            self.anthropic_available = False
    
    def generate_response(self, query: str, context: List[Dict[str, Any]], provider: str = "openai") -> Dict[str, Any]:
        """
        Generate a response with source attribution
        
        Args:
            query: User query
            context: List of document chunks from vector search
            provider: LLM provider to use
        
        Returns:
            Response with source attribution
        """
        formatted_context = self._format_context(context)
        prompt = self._generate_prompt(query, formatted_context)
        
        if provider == "openai" and self.openai_available:
            return self._generate_with_openai(prompt, query, context)
        elif provider == "anthropic" and self.anthropic_available:
            return self._generate_with_anthropic(prompt, query, context)
        else:
            if self.openai_available:
                return self._generate_with_openai(prompt, query, context)
            elif self.anthropic_available:
                return self._generate_with_anthropic(prompt, query, context)
            else:
                return {"error": "No LLM providers available for response generation"}
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context chunks for the prompt"""
        formatted_chunks = []
        
        for i, chunk in enumerate(context):
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            score = chunk.get('score', 0)
            
            source_info = []
            if 'filename' in metadata:
                source_info.append(f"File: {metadata['filename']}")
            
            if 'file_type' in metadata:
                if metadata['file_type'] == 'pdf' and 'page_num' in metadata:
                    source_info.append(f"Page: {metadata['page_num']}")
                elif metadata['file_type'] == 'docx' and 'paragraph_range' in metadata:
                    source_info.append(f"Paragraphs: {metadata['paragraph_range']}")
                elif metadata['file_type'] == 'tabular' and 'sheet_name' in metadata:
                    source_info.append(f"Sheet: {metadata['sheet_name']}")
            
            source_str = ", ".join(source_info)
            formatted_chunk = f"CONTEXT ITEM {i+1} [SOURCE: {source_str}] [RELEVANCE: {score:.2f}]:\n{text}\n"
            
            if 'table_data' in metadata and metadata['table_data']:
                formatted_chunk += "\nTABULAR DATA:\n"
                try:
                    if isinstance(metadata['table_data'], str):
                        table_data = json.loads(metadata['table_data'])
                    else:
                        table_data = metadata['table_data']
                    formatted_chunk += json.dumps(table_data, indent=2)
                except:
                    formatted_chunk += str(metadata['table_data'])
            
            formatted_chunks.append(formatted_chunk)
        
        return "\n\n".join(formatted_chunks)
    
    def _generate_prompt(self, query: str, formatted_context: str) -> str:
        """Generate a prompt for the LLM"""
        return f"""
        You are a document-aware assistant that provides accurate answers based on the retrieved document context and cites sources.
        Always refer to the provided context to answer questions, and do not make up information.
        
        CONTEXT:
        {formatted_context}
        
        USER QUERY:
        {query}
        
        ANSWER (with source citations):
        """
    
    def _generate_with_openai(self, prompt: str, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using OpenAI GPT models"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['openai'].get('model', 'gpt-4-turbo'),
                messages=[{"role": "system", "content": "You are a document-aware assistant with source citation abilities."},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            response_text = response.choices[0].message.content
            citations = self._extract_citations(response_text, context)
            return {"query": query, "response": response_text, "citations": citations, "provider": "openai"}
        except Exception as e:
            return {"error": f"OpenAI response generation failed: {str(e)}"}
    
    def _generate_with_anthropic(self, prompt: str, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using Anthropic Claude"""
        try:
            response = self.anthropic_client.messages.create(
                model=self.config['anthropic'].get('model', 'claude-3-sonnet-20240229'),
                max_tokens=1500,
                temperature=0.3,
                system="You are a document-aware assistant with source citation abilities.",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text
            citations = self._extract_citations(response_text, context)
            return {"query": query, "response": response_text, "citations": citations, "provider": "anthropic"}
        except Exception as e:
            return {"error": f"Anthropic response generation failed: {str(e)}"}
    
    def _extract_citations(self, response_text: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source citations from response text"""
        citation_pattern = r'\[Source: ([^\]]+)\]'
        matches = re.findall(citation_pattern, response_text)
        citations = []
        for match in matches:
            citations.append({"source": match})
        return citations
