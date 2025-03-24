import React, { useState } from 'react';
import axios from 'axios';

const VisualizationPanel = ({ visualizations, setVisualizations }) => {
  const [activeVisualization, setActiveVisualization] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const generateVisualizations = async (useSmartViz = true) => {
    setLoading(true);
    setError('');
    
    try {
      // Use the smart visualization endpoint if specified, otherwise use the regular one
      const endpoint = useSmartViz ? '/api/generate-smart-visualizations' : '/api/generate-visualizations';
      console.log(`Requesting visualization generation using ${endpoint}...`);
      
      const response = await axios.post(endpoint);
      console.log("Visualization generation response:", response.data);
      
      if (response.data.success) {
        setVisualizations(response.data.visualizations);
        
        // Reset active visualization if there are results
        if (response.data.visualizations.length > 0) {
          setActiveVisualization(0);
        }
      } else {
        setError(response.data.error || 'Failed to generate visualizations');
      }
    } catch (error) {
      console.error('Error generating visualizations:', error);
      setError('Error generating visualizations: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };
  
  const hasVisualizations = visualizations && visualizations.length > 0;
  
  return (
    <div className="panel visualization-panel">
      <div className="panel-header">
        <h3>Data Visualizations</h3>
        <div className="panel-controls">
          {hasVisualizations && (
            <select 
              value={activeVisualization} 
              onChange={(e) => setActiveVisualization(parseInt(e.target.value))}
            >
              {visualizations.map((viz, idx) => (
                <option key={idx} value={idx}>
                  {viz.title}
                </option>
              ))}
            </select>
          )}
          
          <div className="button-group">
            <button 
              onClick={() => generateVisualizations(true)} 
              disabled={loading}
              className="generate-button smart-button"
            >
              {loading ? 'Generating...' : 'Smart Visualizations'}
            </button>
          </div>
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
            <p>Generating visualizations, this may take a few minutes...</p>
            <div className="spinner"></div>
          </div>
        ) : hasVisualizations ? (
          <>
            {visualizations[activeVisualization] ? (
              <>
                <h4>{visualizations[activeVisualization].title}</h4>
                <p>{visualizations[activeVisualization].description}</p>
                
                <div className="visualization-frame">
                  <iframe 
                    src={`/visualizations/${visualizations[activeVisualization].filename}`}
                    title={visualizations[activeVisualization].title}
                    width="100%"
                    height="500px"
                    frameBorder="0"
                  />
                </div>
                
                {visualizations[activeVisualization].stats && (
                  <div className="visualization-stats">
                    <h5>Statistics:</h5>
                    <ul>
                      {Object.entries(visualizations[activeVisualization].stats).map(([key, value]) => (
                        <li key={key}>
                          <strong>{key}:</strong> {value}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <p>No visualization selected</p>
            )}
          </>
        ) : (
          <div className="viz-instructions">
            <h3>No visualizations available</h3>
            <p>The system can automatically analyze your documents and create intelligent visualizations from any numerical data found.</p>
            <p>Click "Smart Visualizations" to generate charts that best represent the data in your documents.</p>
            <div className="viz-examples">
              <div className="viz-example">
                <h4>Bar Charts</h4>
                <p>For comparing values across categories</p>
              </div>
              <div className="viz-example">
                <h4>Line Charts</h4>
                <p>For showing trends over time</p>
              </div>
              <div className="viz-example">
                <h4>Pie Charts</h4>
                <p>For showing proportions of a whole</p>
              </div>
            </div>
            <p className="note">Note: For best results, upload documents that contain numerical data, statistics, or tables.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default VisualizationPanel;