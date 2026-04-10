import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingFrames, setProcessingFrames] = useState([]);
  const [streamActive, setStreamActive] = useState(false);
  const [liveFrame, setLiveFrame] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const ws = useRef(null);
  const streamInterval = useRef(null);

  useEffect(() => {
    fetchData();
    return () => stopStream();
  }, []);

  const fetchData = async () => {
    try {
      const response = await fetch('http://localhost:8000/results');
      const data = await response.json();
      setSessions(data.sessions || []);
      if (data.sessions?.length > 0 && !activeSession) {
        setActiveSession(data.sessions[0]);
      }
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  useEffect(() => {
    if (activeSession?.frames?.length > 0) {
      setSelectedFrame(activeSession.frames[0]);
    }
  }, [activeSession]);

  const deleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await fetch(`http://localhost:8000/session/${sessionId}`, { method: 'DELETE' });
      await fetchData();
      if (activeSession?.id === sessionId) {
        setActiveSession(null);
        setSelectedFrame(null);
      }
    } catch (error) {
      console.error("Error deleting session:", error);
    }
  };

  const clearAllData = async () => {
    if (!window.confirm("Delete all video data?")) return;
    try {
      await fetch('http://localhost:8000/clear', { method: 'DELETE' });
      setSessions([]);
      setActiveSession(null);
      setSelectedFrame(null);
    } catch (error) {
      console.error("Error clearing data:", error);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsProcessing(true);
    setProcessingFrames([]);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/process', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      if (data.frames && data.frames.length > 0) {
        setProcessingFrames(data.frames);
        setSelectedFrame(data.frames[0]);
      }
      
      await fetchData();
    } catch (error) {
      console.error("Error processing video:", error);
    } finally {
      setIsProcessing(false);
      setProcessingFrames([]);
    }
  };

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch(e => console.error("Video play error:", e));
        };
      }

      ws.current = new WebSocket('ws://localhost:8000/ws');
      
      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setStreamActive(true);
        
        if (canvasRef.current) {
          streamInterval.current = setInterval(() => {
            if (videoRef.current && ws.current?.readyState === WebSocket.OPEN) {
              const context = canvasRef.current.getContext('2d');
              context.drawImage(videoRef.current, 0, 0, 640, 480);
              canvasRef.current.toBlob(blob => {
                if (ws.current?.readyState === WebSocket.OPEN) {
                  ws.current.send(blob);
                }
              }, 'image/jpeg', 0.7);
            }
          }, 1000);
        }
      };

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setLiveFrame(data);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.current.onclose = () => {
        setStreamActive(false);
        if (streamInterval.current) {
          clearInterval(streamInterval.current);
          streamInterval.current = null;
        }
      };

    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Could not access camera. Please check permissions.");
    }
  };

  const stopStream = () => {
    setStreamActive(false);
    setLiveFrame(null);
    
    if (streamInterval.current) {
      clearInterval(streamInterval.current);
      streamInterval.current = null;
    }

    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const displayFrames = isProcessing ? processingFrames : (activeSession?.frames || []);

  return (
    <div className="app">
      <header className="header">
        <h1>Video Understanding Hub</h1>
        <div className="header-controls">
          <label className="btn-upload">
            Upload
            <input type="file" accept="video/*,image/*" onChange={handleFileUpload} />
          </label>
          {!streamActive ? (
            <button className="btn btn-stream" onClick={startStream}>Go Live</button>
          ) : (
            <button className="btn btn-stop" onClick={stopStream}>Stop</button>
          )}
        </div>
      </header>

      {isProcessing && (
        <div className="processing">
          Processing... {processingFrames.length} frames found
        </div>
      )}

      <main className="main">
        <section className="sessions-panel">
          <div className="panel-header">
            <h2>Videos</h2>
            {sessions.length > 0 && (
              <button className="btn-clear" onClick={clearAllData}>Clear All</button>
            )}
          </div>
          <div className="sessions-list">
            {sessions.length === 0 && !streamActive && <p className="empty">No videos yet</p>}
            {sessions.map((session) => (
              <div 
                key={session.id}
                className={`session-item ${activeSession?.id === session.id ? 'active' : ''}`}
                onClick={() => { setActiveSession(session); }}
              >
                <span className="session-name">{session.name}</span>
                <div className="session-actions">
                  <span className="session-count">{session.frames?.length || 0}</span>
                  <button 
                    className="btn-delete" 
                    onClick={(e) => deleteSession(session.id, e)}
                  >×</button>
                </div>
              </div>
            ))}
            {streamActive && (
              <div className="session-item active">
                <span className="session-name">Live Stream</span>
                <span className="session-count live-indicator">LIVE</span>
              </div>
            )}
          </div>
        </section>

        <section className="frames-list">
          <h2>Frames {displayFrames.length > 0 ? `(${displayFrames.length})` : ''}</h2>
          <div className="frames-container">
            {streamActive && liveFrame && (
              <div className="frame-item active">
                <div className="frame-thumb">
                  <img src={liveFrame.image_data} alt="Live" />
                </div>
                <div className="frame-meta">
                  <span className="frame-time">{liveFrame.timestamp}</span>
                  <span className="frame-objects new">
                    {liveFrame.unique_objects?.join(', ') || 'Processing...'}
                  </span>
                </div>
              </div>
            )}
            {displayFrames.length === 0 && !streamActive && (
              <p className="empty">No frames</p>
            )}
            {displayFrames.map((item, index) => (
              <div 
                key={index} 
                className={`frame-item ${selectedFrame === item ? 'active' : ''}`}
                onClick={() => setSelectedFrame(item)}
              >
                <div className="frame-thumb">
                  {item.image_data ? (
                    <img src={item.image_data} alt={`Frame ${index + 1}`} />
                  ) : (
                    <div className="no-thumb">-</div>
                  )}
                </div>
                <div className="frame-meta">
                  <span className="frame-time">{String(item.timestamp).split('.')[0]}</span>
                  <span className="frame-objects">
                    {item.unique_objects?.length > 0 
                      ? item.unique_objects.slice(0, 3).join(', ')
                      : item.detected_objects?.map(o => o.label).slice(0, 2).join(', ') || 'None'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="frame-viewer">
          {streamActive ? (
            <div className="live-container">
              <div className="live-cam">
                <video ref={videoRef} autoPlay playsInline muted />
              </div>
              <div className="live-result">
                {liveFrame ? (
                  <img src={liveFrame.image_data} alt="Processed" />
                ) : (
                  <div className="no-image">Waiting for first frame...</div>
                )}
                <div className="live-label">Processed Result</div>
              </div>
              <canvas ref={canvasRef} width={640} height={480} style={{ display: 'none' }} />
            </div>
          ) : selectedFrame ? (
            <div className="selected-frame">
              {selectedFrame.image_data ? (
                <img src={selectedFrame.image_data} alt="Selected frame" />
              ) : (
                <div className="no-image">No image</div>
              )}
            </div>
          ) : (
            <div className="no-image">Select a frame to view</div>
          )}
        </section>

        <section className="frame-details">
          <h2>Details</h2>
          {selectedFrame ? (
            <div className="details-content">
              <div className="detail-row">
                <span className="label">Timestamp</span>
                <span className="value">{String(selectedFrame.timestamp).split('.')[0]}</span>
              </div>
              
              <div className="detail-row">
                <span className="label">All Objects</span>
                <div className="objects-grid">
                  {selectedFrame.detected_objects?.length > 0 ? (
                    selectedFrame.detected_objects.map((obj, i) => (
                      <span 
                        key={i} 
                        className={`object-tag ${obj.is_new ? 'new' : 'old'}`}
                      >
                        {obj.label}
                      </span>
                    ))
                  ) : (
                    <span className="none">None detected</span>
                  )}
                </div>
              </div>
              
              <div className="detail-row">
                <span className="label">Unique Objects</span>
                <div className="unique-objects">
                  {selectedFrame.unique_objects?.length > 0 ? (
                    selectedFrame.unique_objects.map((label, i) => (
                      <span key={i} className="unique-tag">{label}</span>
                    ))
                  ) : (
                    <span className="none">No new unique</span>
                  )}
                </div>
              </div>
              
              <div className="detail-row">
                <span className="label">Extracted Text</span>
                <p className="ocr-text">
                  {selectedFrame.extracted_text || 'No text detected'}
                </p>
              </div>
            </div>
          ) : (
            <p className="empty">No frame selected</p>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;