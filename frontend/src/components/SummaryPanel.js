import React, { useState } from 'react';
import axios from 'axios';

const SummaryPanel = ({ summaries, setSummaries }) => {
  const [activeProvider, setActiveProvider] = useState(Object.keys(summaries)[0] || '');
  const [activeSummaryType, setActiveSummaryType] = useState('general');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const generateSummaries = async () => {
    setLoading(true);
    setError('');
    
    try {
      console.log("Requesting summary generation...");
      const response = await axios.post('/api/generate-summaries');
      console.log("Summary generation response:", response.data);
      
      if (response.data.success) {
        setSummaries(response.data.summaries);
        
        // Set active provider to first available one
        const providers = Object.keys(response.data.summaries);
        if (providers.length > 0) {
          setActiveProvider(providers[0]);
        }
      } else {
        setError(response.data.error || 'Failed to generate summaries');
      }
    } catch (error) {
      console.error('Error generating summaries:', error);
      setError('Error generating summaries: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };
  
  const hasSummaries = summaries && Object.keys(summaries).length > 0;
  
  // Set default provider if one exists in summaries but activeProvider is not set
  if (hasSummaries && !activeProvider) {
    setActiveProvider(Object.keys(summaries)[0]);
  }
  
  // For debugging
  console.log("Summaries:", summaries);
  console.log("Active provider:", activeProvider);
  console.log("Active summary type:", activeSummaryType);
  
  return (
    <div className="panel summary-panel">
      <div className="panel-header">
        <h3>Document Summary</h3>
        <div className="panel-controls">
          {hasSummaries && (
            <>
              <select 
                value={activeProvider} 
                onChange={(e) => setActiveProvider(e.target.value)}
              >
                {Object.keys(summaries).map(provider => (
                  <option key={provider} value={provider}>
                    {provider.charAt(0).toUpperCase() + provider.slice(1)}
                  </option>
                ))}
              </select>
              
              <select 
                value={activeSummaryType} 
                onChange={(e) => setActiveSummaryType(e.target.value)}
              >
                <option value="general">General</option>
                <option value="executive">Executive</option>
                <option value="detailed">Detailed</option>
              </select>
            </>
          )}
          
          <button 
            onClick={generateSummaries} 
            disabled={loading}
            className="generate-button"
          >
            {loading ? 'Generating...' : 'Generate Summaries'}
          </button>
        </div>
      </div>
      
      <div className="panel-content">
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        
        {loading ? (
          <div className="loading-indicator">
            <p>Generating summaries, this may take a few minutes...</p>
            <div className="spinner"></div>
          </div>
        ) : hasSummaries ? (
          <>
            {activeProvider && summaries[activeProvider] && 
             summaries[activeProvider][activeSummaryType] ? (
              <>
                <div className="summary-content">
                  {summaries[activeProvider][activeSummaryType].summary}
                </div>
                
                {summaries[activeProvider][activeSummaryType].citations && 
                 summaries[activeProvider][activeSummaryType].citations.length > 0 && (
                  <div className="citations-list">
                    <h4>Sources:</h4>
                    <ul>
                      {summaries[activeProvider][activeSummaryType].citations.map((citation, idx) => (
                        <li key={idx}>
                          {Object.entries(citation)
                            .map(([key, value]) => `${key}: ${value}`)
                            .join(', ')}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <p>No summary available for this provider and type. Try a different selection or generate new summaries.</p>
            )}
          </>
        ) : (
          <p>No summaries available. Click "Generate Summaries" to create summaries of your documents.</p>
        )}
      </div>
    </div>
  );
};

export default SummaryPanel;