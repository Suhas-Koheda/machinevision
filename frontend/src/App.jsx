import React, { useEffect, useRef, useState } from 'react'

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const recognitionRef = useRef(null)
  
  const [processedFrame, setProcessedFrame] = useState(null)
  const [detections, setDetections] = useState([])
  const [extractedText, setExtractedText] = useState('')
  const [transcription, setTranscription] = useState('')
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [cameraStatus, setCameraStatus] = useState('OFF')
  
  const [sessions, setSessions] = useState([])
  const [selectedSessionId, setSelectedSessionId] = useState(null)
  const [sessionFrames, setSessionFrames] = useState([])
  const [viewingFrame, setViewingFrame] = useState(null)
  const [speechError, setSpeechError] = useState(null)

  useEffect(() => {
    fetchSessions()
    connectWS()
    return () => {
      wsRef.current?.close()
      recognitionRef.current?.stop()
    }
  }, [])

  const connectWS = () => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`)
    ws.onopen = () => setIsConnected(true)
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'recording_saved') return alert(`Saved: ${data.url}`)
      setProcessedFrame(data.image)
      setDetections(data.objects)
      if (data.text) setExtractedText(prev => prev + ' ' + data.text)
    }
    ws.onerror = (e) => console.error('WS Error:', e)
    ws.onclose = () => { setIsConnected(false); setTimeout(connectWS, 2000) }
    wsRef.current = ws
  }

  const initTranscription = () => {
    const SpeechSDK = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechSDK) { setSpeechError(window.isSecureContext ? 'Not supported' : 'Security Block'); return null; }
    const recognition = new SpeechSDK()
    recognition.continuous = true; recognition.interimResults = true; recognition.lang = 'en-US'
    recognition.onresult = (event) => {
      let finalTranscript = ''
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript
      }
      if (finalTranscript) setTranscription(prev => (prev + ' ' + finalTranscript).slice(-1000))
    }
    recognition.onerror = (e) => setSpeechError(e.error)
    recognition.onend = () => { if (isTranscribing) try { recognition.start() } catch(e) {} }
    recognitionRef.current = recognition
    return recognition
  }

  const toggleTranscription = () => {
    if (!recognitionRef.current && !initTranscription()) return
    if (!isTranscribing) {
      try { recognitionRef.current.start(); setIsTranscribing(true); setSpeechError(null); } catch (e) {}
    } else {
      recognitionRef.current.stop(); setIsTranscribing(false);
    }
  }

  const fetchSessions = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/sessions')
      if (res.ok) setSessions(await res.json())
    } catch(e) {}
  }

  const loadSessionFrames = async (sid) => {
    setSelectedSessionId(sid)
    const res = await fetch(`http://localhost:8000/api/sessions/${sid}/frames`)
    setSessionFrames(await res.json())
  }

  const startStream = async () => {
    setCameraStatus('INITIALIZING...')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480, frameRate: { ideal: 15 } }, 
        audio: true 
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play()
          setCameraStatus('LIVE')
          setIsStreaming(true)
        }
      }
      setTimeout(toggleTranscription, 500)
    } catch (err) { 
      setCameraStatus('ERROR')
      alert('Camera access denied or device not found.') 
    }
  }

  const stopStream = () => {
    videoRef.current?.srcObject?.getTracks().forEach(t => t.stop())
    setIsStreaming(false)
    setCameraStatus('OFF')
    setProcessedFrame(null)
    if (isTranscribing) { recognitionRef.current?.stop(); setIsTranscribing(false); }
    fetchSessions()
  }

  useEffect(() => {
    if (!isStreaming) return
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === 1 && videoRef.current && isStreaming) {
        const ctx = canvasRef.current.getContext('2d')
        ctx.drawImage(videoRef.current, 0, 0, 640, 480)
        canvasRef.current.toBlob(blob => {
          if (blob && wsRef.current) wsRef.current.send(blob)
        }, 'image/jpeg', 0.5)
      }
    }, 500)
    return () => clearInterval(interval)
  }, [isStreaming])

  return (
    <div className="container">
      <div className="main-content">
        <h1 className="title">Vision Intelligence Control</h1>
        
        <div className="controls-bar">
          {!isStreaming ? 
            <button className="btn-primary" onClick={startStream}>▶ START FEED</button> :
            <button className="btn-danger" onClick={stopStream}>⏹ STOP FEED</button>
          }
          <button className={`btn-outline ${isRecording ? 'pulse' : ''}`} onClick={() => {
            wsRef.current?.send(JSON.stringify({ type: isRecording ? 'stop_recording' : 'start_recording' }))
            setIsRecording(!isRecording)
          }} disabled={!isStreaming}>
            {isRecording ? '🔴 STOP' : '⏺ RECORD'}
          </button>
          
          <button className={`btn-outline ${isTranscribing ? 'active' : ''}`} onClick={toggleTranscription}>
             🎤 {isTranscribing ? 'MIC ON' : 'MIC OFF'}
          </button>

          <label className="btn-outline" style={{ cursor: 'pointer' }}>
            📁 UPLOAD
            <input type="file" hidden accept="video/*" onChange={async (e) => {
              const fd = new FormData(); fd.append('file', e.target.files[0])
              await fetch('http://localhost:8000/api/upload-video', { method: 'POST', body: fd })
              fetchSessions()
            }} />
          </label>
        </div>

        <div className="streams-grid">
          <div className="card">
            <div className="card-header">
              <h3>RAW INPUT</h3>
              <span style={{fontSize: '0.6rem', color: cameraStatus === 'LIVE' ? '#22c55e' : '#ff4b4b'}}>{cameraStatus}</span>
            </div>
            <div className="video-container">
              {viewingFrame ? (
                <img src={`http://localhost:8000${viewingFrame.image_path}`} alt="" />
              ) : (
                <video ref={videoRef} autoPlay playsInline muted style={{width: '100%', height: '100%', objectFit: 'contain'}} />
              )}
            </div>
          </div>
          <div className="card">
            <div className="card-header"><h3>AI PROCESSING</h3></div>
            <div className="video-container">
              {viewingFrame ? (
                <div style={{ padding: '1rem' }}>
                  <h3>Results:</h3>
                  <div className="detection-list">
                    {viewingFrame.objects.map((o,i) => <span key={i} className="detection-badge">{o.label}</span>)}
                  </div>
                  <h3>Text:</h3>
                  <p className="text-output">{viewingFrame.text}</p>
                </div>
              ) : (
                processedFrame ? <img src={processedFrame} alt="" /> : <div style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444'}}>Awaiting frames...</div>
              )}
            </div>
          </div>
        </div>

        <div className="card results-panel">
          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem'}}>
            <div>
              <h2>Integrated Object Feed</h2>
              <div className="detection-list" style={{marginBottom: '0.5rem'}}>
                {detections.slice(-10).map((d,i) => <span key={i} className="detection-badge">{d.label}</span>)}
              </div>
              <div className="text-output">{extractedText || 'Pending analysis...'}</div>
            </div>
            <div>
              <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                <h2>Voice Command Log</h2>
                {speechError && <span style={{fontSize: '0.5rem', color: '#ff4b4b'}}>{speechError}</span>}
              </div>
              <div className="text-output" style={{color: '#e879f9', borderLeftColor: '#c026d3'}}>
                {transcription || (isTranscribing ? 'Listening...' : 'Enable MIC or START FEED')}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="sidebar">
        <div className="card history-panel">
          <div className="card-header"><h3>SESSIONS</h3></div>
          <div style={{ overflowY: 'auto' }}>
            {sessions.map(s => (
              <div key={s.id} className={`history-item ${selectedSessionId === s.id ? 'active' : ''}`} onClick={() => loadSessionFrames(s.id)}>
                <div style={{display: 'flex', justifyContent: 'space-between'}}>
                  <strong>{s.id.split('_')[0].toUpperCase()}</strong>
                  <button onClick={async (e) => { e.stopPropagation(); await fetch(`http://localhost:8000/api/sessions/${s.id}`, { method: 'DELETE' }); fetchSessions(); }} style={{background: 'none', color: '#ff4b4b', fontSize: '0.5rem'}}>DEL</button>
                </div>
                <span>{s.start.split('T')[1].slice(0,5)} | {s.count} frames</span>
              </div>
            ))}
          </div>
        </div>

        {selectedSessionId && (
          <div className="card history-panel" style={{marginTop: '1rem'}}>
            <div className="card-header"><h3>FRAME LOG</h3></div>
            <div className="frame-grid">
              {sessionFrames.map(f => (
                <div key={f.id} className="frame-thumb" onClick={() => setViewingFrame(f)}>
                  <img src={`http://localhost:8000${f.image_path}`} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
      <div className="status-indicator">
        <div className={`dot ${isConnected ? 'connected' : ''}`}></div>
        <span>{isConnected ? 'BACKEND ONLINE' : 'DISCONNECTED'}</span>
      </div>
    </div>
  )
}

export default App
