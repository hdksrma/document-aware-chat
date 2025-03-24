import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Import components
import { Message, SummaryPanel, VisualizationPanel, DocumentList } from './components';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [summaries, setSummaries] = useState({});
  const [visualizations, setVisualizations] = useState([]);
  const [activePanel, setActivePanel] = useState('chat');
  const [fileUpload, setFileUpload] = useState(null);
  
  const messagesEndRef = useRef(null);
  
  // Scroll to the bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // Load initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        console.log("Fetching initial data...");
        
        // Fetch document list
        const documentsResponse = await axios.get('/api/documents');
        console.log("Documents response:", documentsResponse.data);
        setDocuments(documentsResponse.data.documents || []);
        
        // Fetch summaries
        const summariesResponse = await axios.get('/api/summaries');
        console.log("Summaries response:", summariesResponse.data);
        setSummaries(summariesResponse.data.summaries || {});
        
        // Fetch visualizations
        const visualizationsResponse = await axios.get('/api/visualizations');
        console.log("Visualizations response:", visualizationsResponse.data);
        setVisualizations(visualizationsResponse.data.visualizations || []);
      } catch (error) {
        console.error('Error fetching initial data:', error);
      }
    };
    
    fetchInitialData();
  }, []);
  
  // Send a message to the chat API
  const sendMessage = async () => {
    if (!input.trim()) return;
    
    // Add user message to the chat
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    
    // Clear input and set loading state
    setInput('');
    setLoading(true);
    
    try {
      // Send message to API
      const response = await axios.post('/api/chat', {
        message: input,
        provider: 'openai' // Could be configurable
      });
      
      // Add AI response to chat
      const aiMessage = {
        role: 'assistant',
        content: response.data.response,
        citations: response.data.citations || []
      };
      
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Add error message
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request.'
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle document upload
  const handleFileUpload = async (e) => {
    e.preventDefault();
    
    if (!fileUpload) return;
    
    const formData = new FormData();
    for (let i = 0; i < fileUpload.length; i++) {
      formData.append('documents', fileUpload[i]);
    }
    
    try {
      setLoading(true);
      
      // Upload documents
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      // Update document list
      setDocuments(prev => [...prev, ...response.data.uploaded]);
      
      // Clear file upload
      setFileUpload(null);
      
      // Show confirmation
      const systemMessage = {
        role: 'system',
        content: `Uploaded ${response.data.uploaded.length} documents successfully.`
      };
      
      setMessages(prev => [...prev, systemMessage]);
    } catch (error) {
      console.error('Error uploading documents:', error);
      
      // Show error
      const errorMessage = {
        role: 'system',
        content: 'Error uploading documents. Please try again.'
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Document-Aware Chat System</h1>
        <div className="tab-navigation">
          <button 
            className={activePanel === 'chat' ? 'active' : ''} 
            onClick={() => setActivePanel('chat')}
          >
            Chat
          </button>
          <button 
            className={activePanel === 'summaries' ? 'active' : ''} 
            onClick={() => setActivePanel('summaries')}
          >
            Summaries
          </button>
          <button 
            className={activePanel === 'visualizations' ? 'active' : ''} 
            onClick={() => setActivePanel('visualizations')}
          >
            Visualizations
          </button>
        </div>
      </header>
      
      <div className="main-content">
        <div className="sidebar">
          <DocumentList documents={documents} />
          
          <div className="upload-section">
            <h3>Upload Documents</h3>
            <form onSubmit={handleFileUpload}>
              <input 
                type="file" 
                multiple
                onChange={(e) => setFileUpload(e.target.files)}
              />
              <button type="submit" disabled={!fileUpload || loading}>
                Upload
              </button>
            </form>
          </div>
        </div>
        
        <div className="content-area">
          {activePanel === 'chat' && (
            <div className="chat-container">
              <div className="messages">
                {messages.length === 0 && (
                  <div className="welcome-message">
                    <h2>Welcome to Document-Aware Chat!</h2>
                    <p>Ask questions about your documents and get answers with source citations.</p>
                  </div>
                )}
                
                {messages.map((message, idx) => (
                  <Message 
                    key={idx} 
                    message={message} 
                    isUser={message.role === 'user'} 
                  />
                ))}
                
                {loading && (
                  <div className="message ai-message">
                    <div className="message-content">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
              
              <div className="chat-input">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a question about your documents..."
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  disabled={loading}
                />
                <button onClick={sendMessage} disabled={loading || !input.trim()}>
                  Send
                </button>
              </div>
            </div>
          )}
          
          {activePanel === 'summaries' && (
            <SummaryPanel summaries={summaries} setSummaries={setSummaries} />
          )}
          
          {activePanel === 'visualizations' && (
            <VisualizationPanel visualizations={visualizations} setVisualizations={setVisualizations} />
          )}
        </div>
      </div>
    </div>
  );
};

export default App;