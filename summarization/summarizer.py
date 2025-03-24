# summarization/summarizer.py

import json
import re
import anthropic
import google.generativeai as genai
from typing import Dict, List, Any, Union, Tuple
from openai import OpenAI

class DocumentSummarizer:
    """Summarizes documents with source attribution using various LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with API keys for different providers
        
        Args:
            config: Dictionary containing API keys and configurations for LLM providers
        """
        self.config = config
        self.initialize_providers()
        
    def initialize_providers(self):
        """Initialize API clients for different LLM providers"""
        # Initialize OpenAI
        if 'openai' in self.config and self.config['openai'].get('api_key'):
            self.openai_client = OpenAI(api_key=self.config['openai']['api_key'])
            self.openai_available = True
        else:
            self.openai_available = False
            
        # Initialize Anthropic
        if 'anthropic' in self.config and self.config['anthropic'].get('api_key'):
            self.anthropic_client = anthropic.Anthropic(api_key=self.config['anthropic']['api_key'])
            self.anthropic_available = True
        else:
            self.anthropic_available = False
            
        # Initialize Google Gemini
        if 'google' in self.config and self.config['google'].get('api_key'):
            genai.configure(api_key=self.config['google']['api_key'])
            self.google_available = True
        else:
            self.google_available = False
    
    def summarize_with_provider(self, 
                               chunks: List[Dict[str, Any]], 
                               provider: str = "openai",
                               summary_type: str = "general",
                               max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a summary using a specified provider
        
        Args:
            chunks: List of document chunks to summarize
            provider: The LLM provider to use ("openai", "anthropic", or "google")
            summary_type: Type of summary to generate ("general", "executive", or "detailed")
            max_tokens: Maximum tokens for the summary
            
        Returns:
            Dictionary containing the summary and source citations
        """
        # Select provider and call appropriate method
        if provider == "openai" and self.openai_available:
            return self._summarize_with_openai(chunks, summary_type, max_tokens)
        elif provider == "anthropic" and self.anthropic_available:
            return self._summarize_with_anthropic(chunks, summary_type, max_tokens)
        elif provider == "google" and self.google_available:
            return self._summarize_with_google(chunks, summary_type, max_tokens)
        else:
            # Fall back to available provider
            if self.openai_available:
                return self._summarize_with_openai(chunks, summary_type, max_tokens)
            elif self.anthropic_available:
                return self._summarize_with_anthropic(chunks, summary_type, max_tokens)
            elif self.google_available:
                return self._summarize_with_google(chunks, summary_type, max_tokens)
            else:
                return {"error": "No LLM providers available for summarization"}
    
    def _prepare_chunks_for_summarization(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare document chunks for summarization by formatting with source information"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.get('metadata', {})
            source_info = self._format_source_info(metadata)
            
            formatted_chunk = f"DOCUMENT SEGMENT {i+1}:\n"
            formatted_chunk += f"SOURCE: {source_info}\n"
            formatted_chunk += f"CONTENT: {chunk.get('text', '')}\n"
            
            formatted_chunks.append(formatted_chunk)
        
        return "\n\n".join(formatted_chunks)
    
    def _format_source_info(self, metadata: Dict[str, Any]) -> str:
        """Format source information from metadata"""
        source_parts = []
        
        if 'filename' in metadata:
            source_parts.append(f"File: {metadata['filename']}")
        
        if 'file_type' in metadata:
            if metadata['file_type'] == 'pdf' and 'page_num' in metadata:
                source_parts.append(f"Page: {metadata['page_num']}")
            elif metadata['file_type'] == 'docx' and 'paragraph_range' in metadata:
                source_parts.append(f"Paragraphs: {metadata['paragraph_range']}")
            elif metadata['file_type'] == 'tabular' and 'sheet_name' in metadata:
                source_parts.append(f"Sheet: {metadata['sheet_name']}")
        
        return ", ".join(source_parts) if source_parts else "Unknown source"
    
    def _get_summary_prompt(self, formatted_chunks: str, summary_type: str) -> str:
        """Generate a prompt for summarization based on summary type"""
        base_prompt = f"""You are a professional document summarizer tasked with distilling key information from multiple documents while maintaining traceability to the original sources.

Below are segments from various documents. Each segment includes SOURCE information that indicates where it came from.

{formatted_chunks}

"""
        
        if summary_type == "executive":
            base_prompt += """Please create a concise executive summary (no more than 2-3 paragraphs) that captures the most critical insights across all documents. Focus on high-level takeaways, key decisions, and strategic implications.

For each main point or insight in your summary, include a citation referencing the source document information in parentheses. For example: "The quarterly revenue increased by 15% (File: Q2_Report.pdf, Page: 7)."

FORMAT:
- Executive Summary: [Your 2-3 paragraph summary with inline citations]
- Key Points: [Bullet list of 3-5 critical takeaways with citations]
"""
        elif summary_type == "detailed":
            base_prompt += """Please create a comprehensive detailed summary that thoroughly covers the key information from all documents. Include specific data points, supporting evidence, and nuanced insights.

Organize the summary by main themes or topics rather than by document. For each statement, claim, or data point, include a citation referencing the source document information in parentheses. For example: "The manufacturing process requires six quality control checkpoints (File: Production_Manual.docx, Paragraphs: 45-48)."

FORMAT:
- Overview: [1 paragraph general introduction]
- Main Findings: [Detailed exploration of key themes with inline citations]
- Supporting Data: [Important statistics, figures or facts with citations]
- Implications/Conclusions: [What these documents collectively suggest or conclude]
"""
        else:  # general summary
            base_prompt += """Please create a balanced summary that captures the key information across all documents. Highlight important facts, findings, arguments, and conclusions.

For each main point in your summary, include a citation referencing the source document information in parentheses. For example: "The project timeline has been extended to Q3 (File: Project_Update.pdf, Page: 4)."

FORMAT:
- Summary: [Your 1-2 page summary with inline citations]
- Key Findings: [Bullet list of important information with citations]
- Questions for Further Investigation: [List any apparent gaps or questions raised by the documents]
"""
        
        return base_prompt
    
    def _summarize_with_openai(self, 
                              chunks: List[Dict[str, Any]], 
                              summary_type: str = "general",
                              max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate a summary using OpenAI models"""
        # Prepare formatted chunks
        formatted_chunks = self._prepare_chunks_for_summarization(chunks)
        
        # Generate prompt based on summary type
        prompt = self._get_summary_prompt(formatted_chunks, summary_type)
        
        try:
            # Use GPT model for summarization
            model = self.config['openai'].get('model', 'gpt-4-turbo')
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional document summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content
            
            # Parse citations for source tracking
            citations = self._extract_citations(summary)
            
            return {
                "summary": summary,
                "citations": citations,
                "provider": "openai",
                "model": model
            }
            
        except Exception as e:
            print(f"OpenAI summarization failed: {str(e)}")
            return {"error": f"OpenAI summarization failed: {str(e)}"}
    
    def _summarize_with_anthropic(self, 
                                 chunks: List[Dict[str, Any]], 
                                 summary_type: str = "general",
                                 max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate a summary using Anthropic Claude"""
        # Prepare formatted chunks
        formatted_chunks = self._prepare_chunks_for_summarization(chunks)
        
        # Generate prompt based on summary type
        prompt = self._get_summary_prompt(formatted_chunks, summary_type)
        
        try:
            # Use Claude model for summarization
            model = self.config['anthropic'].get('model', 'claude-3-sonnet-20240229')
            
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.3,
                system="You are a professional document summarizer.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response.content[0].text
            
            # Parse citations for source tracking
            citations = self._extract_citations(summary)
            
            return {
                "summary": summary,
                "citations": citations,
                "provider": "anthropic",
                "model": model
            }
            
        except Exception as e:
            print(f"Anthropic summarization failed: {str(e)}")
            return {"error": f"Anthropic summarization failed: {str(e)}"}
    
    def _summarize_with_google(self, 
                              chunks: List[Dict[str, Any]], 
                              summary_type: str = "general",
                              max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate a summary using Google Gemini"""
        # Prepare formatted chunks
        formatted_chunks = self._prepare_chunks_for_summarization(chunks)
        
        # Generate prompt based on summary type
        prompt = self._get_summary_prompt(formatted_chunks, summary_type)
        
        try:
            # Use Gemini model for summarization
            model_name = self.config['google'].get('model', 'gemini-1.5-pro')
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(
                [prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=max_tokens
                )
            )
            
            summary = response.text
            
            # Parse citations for source tracking
            citations = self._extract_citations(summary)
            
            return {
                "summary": summary,
                "citations": citations,
                "provider": "google",
                "model": model_name
            }
            
        except Exception as e:
            print(f"Google Gemini summarization failed: {str(e)}")
            return {"error": f"Google Gemini summarization failed: {str(e)}"}
    
    def _extract_citations(self, summary: str) -> List[Dict[str, Any]]:
        """
        Extract citation information from a summary
        
        Example citation formats:
        - (File: Report.pdf, Page: 7)
        - (File: Manual.docx, Paragraphs: 45-48)
        """
        citations = []
        
        # Regular expression to find citations in parentheses
        citation_pattern = r'\(File: ([^,]+), (Page|Paragraphs|Sheet): ([^\)]+)\)'
        
        matches = re.findall(citation_pattern, summary)
        
        for match in matches:
            filename, location_type, location_value = match
            
            citation = {
                "filename": filename.strip(),
                "location_type": location_type.strip(),
                "location_value": location_value.strip()
            }
            
            citations.append(citation)
        
        return citations
    
    def summarize_collection(self, 
                            chunks: List[Dict[str, Any]], 
                            providers: List[str] = ["openai", "anthropic", "google"],
                            summary_types: List[str] = ["general", "executive", "detailed"]) -> Dict[str, Any]:
        """
        Generate multiple summaries across different providers and types
        
        Args:
            chunks: List of document chunks to summarize
            providers: List of LLM providers to use
            summary_types: List of summary types to generate
            
        Returns:
            Dictionary of summaries by provider and type
        """
        results = {}
        
        for provider in providers:
            if provider == "openai" and not self.openai_available:
                continue
            if provider == "anthropic" and not self.anthropic_available:
                continue
            if provider == "google" and not self.google_available:
                continue
                
            provider_results = {}
            
            for summary_type in summary_types:
                summary = self.summarize_with_provider(chunks, provider, summary_type)
                provider_results[summary_type] = summary
            
            if provider_results:
                results[provider] = provider_results
        
        return results