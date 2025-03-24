import React from 'react';

const DocumentList = ({ documents }) => {
  if (!documents || documents.length === 0) {
    return (
      <div className="panel document-list-panel">
        <div className="panel-header">
          <h3>Loaded Documents</h3>
        </div>
        <div className="panel-content">
          <p>No documents available. Upload documents to get started.</p>
        </div>
      </div>
    );
  }
  
  const getDocumentIcon = (filename) => {
    const extension = filename.split('.').pop().toLowerCase();
    
    switch (extension) {
      case 'pdf':
        return '📄';
      case 'docx':
      case 'doc':
        return '📝';
      case 'xlsx':
      case 'xls':
      case 'csv':
        return '📊';
      case 'txt':
        return '📃';
      default:
        return '📁';
    }
  };
  
  return (
    <div className="panel document-list-panel">
      <div className="panel-header">
        <h3>Loaded Documents</h3>
      </div>
      
      <div className="panel-content">
        <ul className="document-list">
          {documents.map((doc, idx) => (
            <li key={idx} className="document-item">
              <div className="document-icon">
                {getDocumentIcon(doc)}
              </div>
              <div className="document-name">{doc}</div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default DocumentList;