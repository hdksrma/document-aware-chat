import React from 'react';

const Message = ({ message, isUser }) => {
  // Extract citations from AI messages
  const renderCitations = (text) => {
    if (!text) return '';
    
    // Find citation patterns like [Source: File X, Page Y]
    const parts = [];
    let lastIndex = 0;
    
    const citationRegex = /\[Source: ([^\]]+)\]/g;
    let match;
    
    while ((match = citationRegex.exec(text)) !== null) {
      // Add text before the citation
      if (match.index > lastIndex) {
        parts.push(
          <span key={`text-${lastIndex}`}>
            {text.substring(lastIndex, match.index)}
          </span>
        );
      }
      
      // Add the citation with highlighting
      parts.push(
        <span 
          key={`citation-${match.index}`} 
          className="citation-highlight"
          title={match[1]}
        >
          {match[0]}
        </span>
      );
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add any remaining text
    if (lastIndex < text.length) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {text.substring(lastIndex)}
        </span>
      );
    }
    
    return parts.length > 0 ? parts : text;
  };

  return (
    <div className={`message ${isUser ? 'user-message' : 'ai-message'}`}>
      <div className="message-content">
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <div className="ai-content">
            <p>{renderCitations(message.content)}</p>
            
            {message.citations && message.citations.length > 0 && (
              <div className="citations-list">
                <h4>Sources:</h4>
                <ul>
                  {message.citations.map((citation, idx) => (
                    <li key={idx}>
                      {Object.entries(citation)
                        .filter(([key]) => key !== 'score')
                        .map(([key, value]) => `${key}: ${value}`)
                        .join(', ')}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Message;