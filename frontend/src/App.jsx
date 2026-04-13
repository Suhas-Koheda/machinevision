import React, { useEffect, useRef, useState } from 'react'

const Panel = ({ title, children, actions }) => (
  <div className="card">
    <div className="card-header">
      <h3>{title}</h3>
      {actions}
    </div>
    <div className="p-3" style={{ fontSize: '0.85rem' }}>{children}</div>
  </div>
)

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const recognitionRef = useRef(null)
  
  // High-frequency state for video feeds
  const [processedImg, setProcessedImg] = useState(null)
  const [diagnosticMaps, setDiagnosticMaps] = useState({ noise: null, flow: null })
  
  // Persistent Intelligence State (prevents flickering)
  const [intel, setIntel] = useState({
    objects: [],
    semantic: [],
    graph: [],
    text: '',
    events: [],
    scores: { motion: 0, noise: 0 }
  })

  const [transcription, setTranscription] = useState('')
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  
  const [sessions, setSessions] = useState([])
  const [selectedSid, setSelectedSid] = useState(null)
  const [sessionFrames, setSessionFrames] = useState([])
  const [viewingFrame, setViewingFrame] = useState(null)

  useEffect(() => {
    fetchSessions()
    connectWS()
    initTranscription()
    return () => { wsRef.current?.close(); recognitionRef.current?.stop(); }
  }, [])

  const connectWS = () => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`)
    ws.onopen = () => setIsConnected(true)
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'recording_saved') return alert(`Archive Exported: ${msg.url}`)
      
      // Update fast-moving visuals immediately
      if (msg.processed_img) setProcessedImg(msg.processed_img)
      if (msg.noise_map || msg.flow_map) {
        setDiagnosticMaps({ noise: msg.noise_map, flow: msg.flow_map })
      }

      // Update Intelligence ONLY if there is actual new data (Prevents the "flicker")
      setIntel(prev => ({
        objects: msg.objects?.length ? msg.objects : prev.objects,
        semantic: msg.semantic?.length ? msg.semantic : prev.semantic,
        graph: msg.graph?.length ? msg.graph : prev.graph,
        text: msg.text ? msg.text : prev.text,
        events: msg.events?.length ? [...prev.events, ...msg.events].slice(-50) : prev.events,
        scores: { 
            motion: msg.motion_score ?? prev.scores.motion, 
            noise: msg.noise_score ?? prev.scores.noise 
        }
      }))
    }
    ws.onclose = () => { setIsConnected(false); setTimeout(connectWS, 2000) }
    wsRef.current = ws
  }

  const initTranscription = () => {
    const SpeechSDK = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechSDK) {
        console.warn("Speech recognition not supported");
        return;
    }
    const rec = new SpeechSDK()
    rec.continuous = true; rec.interimResults = true; rec.lang = 'en-US'
    rec.onresult = (event) => {
      let final = ''
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) final += event.results[i][0].transcript
      }
      if (final) {
        setTranscription(prev => (prev + ' ' + final).slice(-1000))
        // Auto-scroll logic could go here
      }
    }
    rec.onerror = (e) => console.error("Speech Rec Error:", e.error)
    rec.onend = () => { if (isTranscribing) try { rec.start() } catch(e) {} }
    recognitionRef.current = rec
  }

  const toggleTranscription = () => {
    if (isTranscribing) { 
        recognitionRef.current?.stop(); 
        setIsTranscribing(false); 
    } else { 
        try { 
            recognitionRef.current?.start(); 
            setIsTranscribing(true); 
        } catch(e) { 
            alert("Mic access required for transcription."); 
            console.error(e);
        } 
    }
  }

  const fetchSessions = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/sessions')
      if (res.ok) setSessions(await res.json())
    } catch(e) {}
  }

  const loadSession = async (sid) => {
    setSelectedSid(sid); setViewingFrame(null);
    const res = await fetch(`http://localhost:8000/api/sessions/${sid}/frames`)
    if (res.ok) setSessionFrames(await res.json())
  }

  const deleteSession = async (sid) => {
    if (!confirm('Permanently delete session log?')) return
    await fetch(`http://localhost:8000/api/sessions/${sid}`, { method: 'DELETE' })
    fetchSessions()
    if (selectedSid === sid) setSelectedSid(null)
  }

  const startFeed = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
      setIsStreaming(true)
      // Automatically attempt to start transcription if not already active
      if (!isTranscribing) toggleTranscription()
    } catch(e) { alert("Signal Acquisition Failed. Check camera/mic permissions."); }
  }

  const stopFeed = () => {
    videoRef.current?.srcObject?.getTracks().forEach(t => t.stop())
    setIsStreaming(false); setProcessedImg(null);
    if (isTranscribing) toggleTranscription()
    fetchSessions()
  }

  useEffect(() => {
    if (!isStreaming) return
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === 1 && videoRef.current && isStreaming) {
        const ctx = canvasRef.current.getContext('2d')
        ctx.drawImage(videoRef.current, 0, 0, 640, 480)
        canvasRef.current.toBlob(b => wsRef.current?.send(b), 'image/jpeg', 0.5)
      }
    }, 500) // Lower frequency for stability
    return () => clearInterval(interval)
  }, [isStreaming])

  return (
    <div className="app-layout">
      <header className="main-header">
        <div className="brand">BLACKOUT // INTELLIGENCE ARCHIVE</div>
        <div className="controls">
          {!isStreaming ? 
            <button className="btn-primary" onClick={startFeed}>▶ LINK SIGNAL</button> :
            <button onClick={stopFeed}>⏹ DISCONNECT</button>
          }
          <button className={isRecording ? 'pulse' : ''} onClick={() => {
            wsRef.current?.send(JSON.stringify({ type: isRecording ? 'stop_recording' : 'start_recording' }))
            setIsRecording(!isRecording)
          }}>
            {isRecording ? 'RECORDING...' : '⏺ RECORD'}
          </button>
          <button onClick={toggleTranscription}>
             🎤 {isTranscribing ? 'VOICE ON' : 'VOICE OFF'}
          </button>
          <label className="card" style={{padding: '8px 15px', cursor: 'pointer', border: '1px solid #333', fontSize: '0.7rem', fontWeight: 900}}>
            UPLOAD LOCAL <input type="file" hidden onChange={async (e) => {
               const fd = new FormData(); fd.append('file', e.target.files[0])
               await fetch('http://localhost:8000/api/upload-video', { method: 'POST', body: fd })
               fetchSessions()
            }} />
          </label>
        </div>
      </header>

      <main className="dashboard-grid">
        <div className="main-column">
          <section className="visual-section">
            <div className="card visual-card">
              <div className="card-header"><h3>LOCAL SIGNAL</h3></div>
              <div className="video-container">
                {viewingFrame ? <img src={`http://localhost:8000${viewingFrame.image_path}`} /> : <video ref={videoRef} autoPlay playsInline muted />}
              </div>
            </div>
            <div className="card visual-card">
              <div className="card-header"><h3>PERCEPTION BRIDGE</h3></div>
              <div className="video-container">
                {processedImg ? <img src={processedImg} /> : <div className="placeholder">AWAITING LOCK...</div>}
              </div>
            </div>
          </section>

          <section className="analytics-grid">
            <Panel title="OBJECT REGISTRY">
              {intel.objects.map((o,i) => <div key={i} className="list-item"><span className="badge">ID:{o.track_id}</span> {o.label}</div>)}
              {!intel.objects.length && <div style={{color: '#333'}}>No entities detected.</div>}
            </Panel>
            <Panel title="SEMANTIC FLOW">
              {intel.semantic.map((s,i) => <div key={i} className="list-item"><span className="badge">ID:{s.track_id}</span> {s.semantic}</div>)}
              {!intel.semantic.length && <div style={{color: '#333'}}>Analyzing scene...</div>}
            </Panel>
            <Panel title="RELATION MAPPING">
              {intel.graph.map((g,i) => <div key={i} className="list-item">{g.subject} ➔ {g.predicate} ➔ {g.object}</div>)}
              {!intel.graph.length && <div style={{color: '#333'}}>Building graph...</div>}
            </Panel>
            <Panel title="DIAGNOSTIC TELEMETRY">
               <div className="diagnostic-container">
                <div className="diagnostic-map">
                  {diagnosticMaps.noise && <img src={diagnosticMaps.noise} />}
                  <span className="map-label">NOISE: {intel.scores.noise}</span>
                </div>
                <div className="diagnostic-map">
                  {diagnosticMaps.flow && <img src={diagnosticMaps.flow} />}
                  <span className="map-label">MOTION: {intel.scores.motion}</span>
                </div>
              </div>
            </Panel>
          </section>

          <section className="analytics-grid">
            <Panel title="OCR LOG">
              <div className="text-box">{intel.text || "NO TEXT DETECTED."}</div>
            </Panel>
            <Panel title="VOICE TRANSCRIPTION">
              <div className="text-box" style={{color: '#aaa', fontWeight: 500}}>{transcription || (isTranscribing ? "LISTENING..." : "VOICE CONTROL MUTE.")}</div>
            </Panel>
            <Panel title="EVENT HISTORY" style={{gridColumn: 'span 2'}}>
              <div className="event-box">
                {intel.events.slice(-10).map((e,i) => <div key={i} className="event-item"><strong>{e.type}</strong> // {e.msg}</div>)}
                {!intel.events.length && <div style={{color: '#333'}}>Awaiting signal events...</div>}
              </div>
            </Panel>
          </section>
        </div>

        <aside className="sidebar">
          <div className="card" style={{height: '100%'}}>
            <div className="card-header"><h3>SESSION LIBRARY</h3></div>
            <div className="sidebar-content">
              {sessions.map(s => (
                <div key={s.id} className={`session-item ${selectedSid === s.id ? 'active' : ''}`} onClick={() => loadSession(s.id)}>
                   <strong style={{color: '#fff'}}>{s.id.toUpperCase()}</strong>
                   <span style={{display: 'block', fontSize: '0.6rem', color: '#666'}}>{s.count} ITEMS // {s.start.split('T')[1].slice(0,8)}</span>
                   <button onClick={(e) => { e.stopPropagation(); deleteSession(s.id); }} style={{position: 'absolute', top: 10, right: 10, padding: '2px 8px', fontSize: '0.5rem', borderColor: '#444'}}>DEL</button>
                </div>
              ))}
              {selectedSid && (
                <div className="frame-grid-mini">
                   {sessionFrames.map(f => (
                     <div key={f.id} className="frame-thumb">
                        <img src={`http://localhost:8000${f.image_path}`} onClick={() => setViewingFrame(f)} />
                     </div>
                   ))}
                </div>
              )}
            </div>
          </div>
        </aside>
      </main>

      <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
    </div>
  )
}

export default App
