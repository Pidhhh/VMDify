import React, { useState, useEffect, useRef } from 'react';

const BackendConnection = ({ onJobComplete, onFrame }) => {
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [messages, setMessages] = useState([]);
  const [backendInfo, setBackendInfo] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check backend health
  const checkBackendHealth = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/probe');
      const data = await response.json();
      setBackendInfo(data);
      return true;
    } catch (error) {
      console.error('Backend not available:', error);
      setBackendInfo({ status: 'error', message: 'Backend not running' });
      return false;
    }
  };

  // Connect to WebSocket
  const connectWebSocket = () => {
    try {
      const ws = new WebSocket('ws://127.0.0.1:8000/ws');
      
      ws.onopen = () => {
        setConnectionStatus('Connected');
        setMessages(prev => [...prev, { 
          type: 'system', 
          text: 'Connected to backend', 
          timestamp: new Date().toLocaleTimeString() 
        }]);
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data?.type === 'job_complete' && onJobComplete) {
          onJobComplete(data);
        }
        if (data?.type === 'frame' && onFrame) {
          onFrame(data);
        }
        setMessages(prev => [...prev, { 
          type: 'received', 
          text: JSON.stringify(data, null, 2),
          timestamp: new Date().toLocaleTimeString()
        }]);
      };
      
      ws.onclose = () => {
        setConnectionStatus('Disconnected');
        setMessages(prev => [...prev, { 
          type: 'system', 
          text: 'Connection closed',
          timestamp: new Date().toLocaleTimeString()
        }]);
      };
      
      ws.onerror = (error) => {
        setConnectionStatus('Error');
        setMessages(prev => [...prev, { 
          type: 'error', 
          text: `WebSocket error: ${error}`,
          timestamp: new Date().toLocaleTimeString()
        }]);
      };
      
      wsRef.current = ws;
    } catch (error) {
      setConnectionStatus('Error');
      console.error('WebSocket connection failed:', error);
    }
  };

  // Send test message
  const sendTestMessage = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const message = { type: 'test', message: 'Hello from frontend!' };
      wsRef.current.send(JSON.stringify(message));
      setMessages(prev => [...prev, { 
        type: 'sent', 
        text: JSON.stringify(message, null, 2),
        timestamp: new Date().toLocaleTimeString()
      }]);
    }
  };

  const clearMessages = () => {
    setMessages([]);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Connected': return 'text-green-400';
      case 'Disconnected': return 'text-gray-400';
      case 'Error': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'Connected':
        return (
          <svg className="w-4 h-4 text-green-400" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
        );
      case 'Error':
        return (
          <svg className="w-4 h-4 text-red-400" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm5 13.59L15.59 17 12 13.41 8.41 17 7 15.59 10.59 12 7 8.41 8.41 7 12 10.59 15.59 7 17 8.41 13.41 12 17 15.59z"/>
          </svg>
        );
      default:
        return (
          <svg className="w-4 h-4 text-gray-400" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
        );
    }
  };

  useEffect(() => {
    checkBackendHealth();
  // Auto-connect to the backend WebSocket for live streaming
  connectWebSocket();
  }, []);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <svg className="w-5 h-5 text-primary" fill="currentColor" viewBox="0 0 24 24">
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
          </svg>
          <h3 className="text-primary font-semibold">Backend Connection</h3>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-gray-400 hover:text-gray-300 transition-colors"
        >
          <svg 
            className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Status Card */}
      <div className="bg-dark-light rounded-lg p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {getStatusIcon(connectionStatus)}
            <span className={`font-medium ${getStatusColor(connectionStatus)}`}>
              {connectionStatus}
            </span>
          </div>
          <div className={`w-2 h-2 rounded-full ${connectionStatus === 'Connected' ? 'bg-green-400 pulse' : 'bg-gray-500'}`}></div>
        </div>

        {backendInfo && (
          <div className="text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Status:</span>
              <span className="text-gray-300">{backendInfo.status || 'Unknown'}</span>
            </div>
            {backendInfo.message && (
              <div className="flex justify-between">
                <span className="text-gray-400">Message:</span>
                <span className="text-gray-300 truncate">{backendInfo.message}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-1 gap-2">
        <button
          onClick={connectWebSocket}
          className="w-full btn"
          style={{
            backgroundColor: 'rgba(97, 218, 251, 0.15)',
            border: '1px solid rgba(97, 218, 251, 0.3)',
            color: 'var(--primary)'
          }}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
          <span>Connect WebSocket</span>
        </button>
        
        <div className="flex space-x-2">
          <button
            onClick={sendTestMessage}
            disabled={connectionStatus !== 'Connected'}
            className={`flex-1 btn ${
              connectionStatus === 'Connected'
                ? 'btn-secondary'
                : 'btn-disabled'
            }`}
          >
            Test Message
          </button>
          <button
            onClick={checkBackendHealth}
            className="flex-1 btn btn-secondary"
          >
            Health Check
          </button>
        </div>
      </div>

      {/* Messages Panel */}
      {isExpanded && (
        <div className="bg-dark-light rounded-lg p-3 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-gray-300">System Messages</h4>
            <button
              onClick={clearMessages}
              className="text-xs text-gray-400 hover:text-gray-300 transition-colors"
            >
              Clear
            </button>
          </div>
          
          <div className="max-h-48 overflow-y-auto space-y-2">
            {messages.length === 0 ? (
              <p className="text-gray-500 text-xs text-center py-4">No messages yet</p>
            ) : (
              messages.map((msg, index) => (
                <div
                  key={index}
                  className={`p-2 rounded text-xs border-l-2 ${
                    msg.type === 'error'
                      ? 'bg-red-900/20 border-red-500 text-red-200'
                      : msg.type === 'sent'
                      ? 'bg-blue-900/20 border-blue-500 text-blue-200'
                      : msg.type === 'received'
                      ? 'bg-green-900/20 border-green-500 text-green-200'
                      : 'bg-gray-800 border-gray-600 text-gray-300'
                  }`}
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-medium capitalize">{msg.type}</span>
                    <span className="text-gray-500">{msg.timestamp}</span>
                  </div>
                  <pre className="whitespace-pre-wrap text-xs">{msg.text}</pre>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      )}
    </div>
  );
};

export default BackendConnection;
