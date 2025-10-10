import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
// Force rebuild: 2025-10-10 triplet generation fix deployed

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);