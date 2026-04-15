import React, { useEffect, useRef, useState } from 'react'

const SummaryModal = ({ summary, onClose }) => {
  if (!summary) return null
  const allText = summary.ocrLines.join(' ')
  const uniqueWords = [...new Set(allText.split(/\s+/).filter(Boolean))]
  const fullParagraph = summary.ocrLines.join(' ')

  return (
    <div className="summary-overlay" onClick={onClose}>
      <div className="summary-modal" onClick={e => e.stopPropagation()}>
        <div className="summary-modal-header">
          <div>
            <span className="summary-mode-tag">{summary.mode === 'live' ? '● LIVE SESSION' : '📁 UPLOAD'}</span>
            <h2>SESSION SUMMARY</h2>
            {summary.label && <p className="summary-label">{summary.label}</p>}
          </div>
          <button className="summary-close" onClick={onClose}>✕</button>
        </div>

        <div className="summary-stats">
          <div className="summary-stat">
            <span>{summary.ocrLines.length}</span>
            <label>OCR READS</label>
          </div>
          <div className="summary-stat">
            <span>{uniqueWords.length}</span>
            <label>UNIQUE WORDS</label>
          </div>
          <div className="summary-stat">
            <span>{allText.length}</span>
            <label>CHARACTERS</label>
          </div>
        </div>

        <div className="summary-section">
          <div className="summary-section-title">EXTRACTED TEXT // FULL LOG</div>
          <div className="summary-text-box">
            {summary.ocrLines.length
              ? summary.ocrLines.map((line, i) => (
                  <div key={i} className="summary-line">
                    <span className="summary-line-num">{String(i + 1).padStart(2, '0')}</span>
                    <span>{line}</span>
                  </div>
                ))
              : <span style={{ color: '#444' }}>No text was detected during this session.</span>
            }
          </div>
        </div>

        {fullParagraph && (
          <div className="summary-section">
            <div className="summary-section-title">COMPILED PARAGRAPH</div>
            <div className="summary-paragraph">{fullParagraph}</div>
          </div>
        )}

        <div className="summary-actions">
          <button onClick={() => navigator.clipboard.writeText(summary.ocrLines.join('\n'))} className="btn-primary">
            📋 COPY LOG
          </button>
          <button onClick={() => navigator.clipboard.writeText(fullParagraph)}>
            📄 COPY PARAGRAPH
          </button>
          <button onClick={onClose}>CLOSE</button>
        </div>
      </div>
    </div>
  )
}

const SearchModal = ({ isOpen, onClose, onSelect }) => {
  const [file, setFile] = useState(null)
  const [results, setResults] = useState([])
  const [isSearching, setIsSearching] = useState(false)

  if (!isOpen) return null

  const handleSearch = async (e) => {
    const uploadedFile = e.target.files[0]
    if (!uploadedFile) return
    setFile(URL.createObjectURL(uploadedFile))
    setIsSearching(true)

    const fd = new FormData()
    fd.append('file', uploadedFile)

    try {
      const res = await fetch('http://localhost:8000/api/search-image', { method: 'POST', body: fd })
      if (res.ok) setResults(await res.json())
    } catch (e) {
      alert("Search Failed")
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <div className="summary-overlay" onClick={onClose}>
      <div className="summary-modal" style={{ width: 'min(900px, 95vw)' }} onClick={e => e.stopPropagation()}>
        <div className="summary-modal-header">
          <div>
            <span className="summary-mode-tag">🧠 NEURAL SEARCH</span>
            <h2>VIRTUAL ARCHIVE LOOKUP</h2>
          </div>
          <button className="summary-close" onClick={onClose}>✕</button>
        </div>

        <div className="search-input-area">
          <label className="search-upload-box">
            {file ? <img src={file} className="search-preview" /> : <span>DRAG OR CLICK TO UPLOAD QUERY IMAGE</span>}
            <input type="file" hidden onChange={handleSearch} />
          </label>
          {isSearching && <div className="searching-indicator">SCANNING NEURAL ARCHIVE...</div>}
        </div>

        <div className="summary-section">
          <div className="summary-section-title">MATCHING INTELLIGENCE FRAGMENTS</div>
          <div className="search-results-grid">
            {results.length ? results.map((r, i) => (
              <div key={i} className="search-result-card" onClick={() => { onSelect(r); onClose(); }}>
                <img src={`http://localhost:8000${r.image_path}`} />
                <div className="search-result-info">
                  <span className="search-conf">MATCH: {(r.similarity * 100).toFixed(1)}%</span>
                  <span className="search-meta">{r.session_id} // f:{r.frame_index}</span>
                </div>
              </div>
            )) : <div className="search-empty">Upload an image to start neural comparison.</div>}
          </div>
        </div>

        <div className="summary-actions">
          <button onClick={onClose}>CLOSE SEARCH</button>
        </div>
      </div>
    </div>
  )
}

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

  // Accumulated OCR log (client-side, never clears mid-session)
  const ocrLogRef = useRef([])
  const [ocrLog, setOcrLog] = useState([])

  const [isConnected, setIsConnected] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [summary, setSummary] = useState(null)

  const [sessions, setSessions] = useState([])
  const [selectedSid, setSelectedSid] = useState(null)
  const [sessionFrames, setSessionFrames] = useState([])
  const [viewingFrame, setViewingFrame] = useState(null)
  const [isSearchOpen, setIsSearchOpen] = useState(false)

  useEffect(() => {
    fetchSessions()
    connectWS()
    return () => { wsRef.current?.close() }
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

      // Accumulate OCR text client-side with smart deduplication
      if (msg.text && msg.text.trim()) {
        const entry = msg.text.trim()
        const existing = ocrLogRef.current
        const last = existing[existing.length - 1]

        let shouldAdd = true
        if (last) {
          const l1 = last.toLowerCase().trim()
          const l2 = entry.toLowerCase().trim()
          
          const w1 = l1.split(/\s+/).filter(w => w.length > 1)
          const w2 = l2.split(/\s+/).filter(w => w.length > 1)
          const s1 = new Set(w1)
          const s2 = new Set(w2)
          
          const intersect = w2.filter(w => s1.has(w)).length
          const unionSize = new Set([...w1, ...w2]).size
          const Jaccard = intersect / unionSize

          // 1. Strict containment (handles logo only vs logo + text)
          if (l1 === l2 || l1.includes(l2)) {
            shouldAdd = false
          } 
          else if (l2.includes(l1)) {
            // New text is more complete (extension) - replace last entry
            const updated = [...existing.slice(0, -1), entry]
            ocrLogRef.current = updated
            setOcrLog([...updated])
            shouldAdd = false
          }
          // 2. High Jaccard similarity (handles slight OCR glitches)
          else if (Jaccard > 0.7) {
            // Keep the longer one as it's likely more accurate
            if (l2.length > l1.length) {
              const updated = [...existing.slice(0, -1), entry]
              ocrLogRef.current = updated
              setOcrLog([...updated])
            }
            shouldAdd = false
          }
          // 3. Otherwise, if they share a common word (logo) but have enough distinct words, 
          // we treat it as a new headline.
        }

        if (shouldAdd) {
          const updated = [...existing, entry].slice(-100)
          ocrLogRef.current = updated
          setOcrLog([...updated])
        }
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

  const fetchSessions = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/sessions')
      if (res.ok) setSessions(await res.json())
    } catch (e) { }
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
      // Video only — voice removed
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
      // Reset OCR log for new session
      ocrLogRef.current = []
      setOcrLog([])
      setIntel({ objects: [], semantic: [], graph: [], text: '', events: [], scores: { motion: 0, noise: 0 } })
      setIsStreaming(true)
    } catch (e) { alert("Signal Acquisition Failed. Check camera permissions."); }
  }

  const stopFeed = () => {
    videoRef.current?.srcObject?.getTracks().forEach(t => t.stop())
    // Show summary before clearing state
    if (ocrLogRef.current.length > 0) {
      setSummary({ mode: 'live', ocrLines: [...ocrLogRef.current], label: null })
    }
    setIsStreaming(false); setProcessedImg(null);
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
    }, 500)
    return () => clearInterval(interval)
  }, [isStreaming])

  // Sync dashboard with historic frame data when viewing archives
  useEffect(() => {
    if (viewingFrame) {
      setIntel({
        objects: viewingFrame.objects || [],
        semantic: viewingFrame.semantic || [],
        graph: viewingFrame.graph || [],
        text: viewingFrame.text || '',
        events: viewingFrame.events || [],
        scores: {
          motion: viewingFrame.motion_score || 0,
          noise: viewingFrame.noise_score || 0
        }
      })
    }
  }, [viewingFrame])

  return (
    <div className="app-layout">
      <header className="main-header">
        <div className="brand">VisionFlow // INTELLIGENCE ARCHIVE</div>
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
          <label className="card" style={{ padding: '8px 15px', cursor: 'pointer', border: '1px solid #333', fontSize: '0.7rem', fontWeight: 900 }}>
            UPLOAD LOCAL <input type="file" hidden onChange={async (e) => {
              const fileName = e.target.files[0]?.name
              // Reset OCR log before upload so we capture only this file's text
              ocrLogRef.current = []
              setOcrLog([])
              const fd = new FormData(); fd.append('file', e.target.files[0])
              await fetch('http://localhost:8000/api/upload-video', { method: 'POST', body: fd })
              // Small delay to let final WS messages arrive before reading log
              setTimeout(() => {
                setSummary({ mode: 'upload', ocrLines: [...ocrLogRef.current], label: fileName })
                fetchSessions()
              }, 800)
            }} />
          </label>
          <span style={{ fontSize: '0.65rem', color: isConnected ? '#00ff88' : '#ff4444', marginLeft: 8, fontWeight: 900 }}>
            {isConnected ? '● ONLINE' : '○ OFFLINE'}
          </span>
          <button className="btn-primary" style={{ marginLeft: 15 }} onClick={() => setIsSearchOpen(true)}>
            🔍 SEARCH ARCHIVE
          </button>
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
              {intel.objects.map((o, i) => <div key={i} className="list-item"><span className="badge">ID:{o.track_id}</span> {o.label}</div>)}
              {!intel.objects.length && <div style={{ color: '#333' }}>No entities detected.</div>}
            </Panel>
            <Panel title="SEMANTIC FLOW">
              {intel.semantic.map((s, i) => <div key={i} className="list-item"><span className="badge">ID:{s.track_id}</span> {s.semantic}</div>)}
              {!intel.semantic.length && <div style={{ color: '#333' }}>Analyzing scene...</div>}
            </Panel>
            <Panel title="RELATION MAPPING">
              {intel.graph.map((g, i) => <div key={i} className="list-item">{g.subject} ➔ {g.predicate} ➔ {g.object}</div>)}
              {!intel.graph.length && <div style={{ color: '#333' }}>Building graph...</div>}
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
            <Panel title={viewingFrame ? "OCR LOG // ARCHIVE FRAME TEXT" : "OCR LOG // LIVE TEXT EXTRACTION"}>
              <div
                className="text-box"
                style={{
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  lineHeight: 1.8,
                  height: '200px',
                  overflowY: 'auto',
                  display: 'flex',
                  flexDirection: 'column-reverse'
                }}
              >
                <div>
                  {viewingFrame ? (
                    intel.text ? <div style={{ color: '#00ff88' }}>{intel.text}</div> : <span style={{ color: '#444' }}>NO TEXT DETECTED IN THIS FRAME.</span>
                  ) : (
                    ocrLog.length
                      ? ocrLog.map((t, i) => <div key={i} style={{ borderBottom: '1px solid #1a1a1a', paddingBottom: 2, color: '#00ff88' }}>{t}</div>)
                      : <span style={{ color: '#444' }}>NO TEXT DETECTED IN FRAME.</span>
                  )}
                </div>
              </div>
            </Panel>
            <Panel title="EVENT HISTORY">
              <div className="event-box">
                {intel.events.slice(-10).map((e, i) => <div key={i} className="event-item"><strong>{e.type}</strong> // {e.msg}</div>)}
                {!intel.events.length && <div style={{ color: '#333' }}>Awaiting signal events...</div>}
              </div>
            </Panel>
          </section>
        </div>

        <aside className="sidebar">
          <div className="card" style={{ height: '100%' }}>
            <div className="card-header"><h3>SESSION LIBRARY</h3></div>
            <div className="sidebar-content">
              {sessions.map(s => (
                <div key={s.id} className={`session-item ${selectedSid === s.id ? 'active' : ''}`} onClick={() => loadSession(s.id)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <strong style={{ color: '#fff' }}>{s.id.toUpperCase()}</strong>
                      <span style={{ display: 'block', fontSize: '0.6rem', color: '#666' }}>{s.count} ITEMS // {s.start.split('T')[1].slice(0, 8)}</span>
                    </div>
                    <button onClick={(e) => { e.stopPropagation(); deleteSession(s.id); }} style={{ padding: '2px 8px', fontSize: '0.5rem', borderColor: '#444' }}>DEL</button>
                  </div>
                  {selectedSid === s.id && (
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        const ocrLines = sessionFrames.map(f => f.text).filter(Boolean);
                        setSummary({ mode: 'archive', ocrLines, label: s.id });
                      }}
                      className="btn-primary"
                      style={{ width: '100%', marginTop: 10, fontSize: '0.6rem', padding: '4px' }}
                    >
                      VIEW SESSION SUMMARY
                    </button>
                  )}
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
      <SummaryModal summary={summary} onClose={() => setSummary(null)} />
      <SearchModal isOpen={isSearchOpen} onClose={() => setIsSearchOpen(false)} onSelect={setViewingFrame} />
    </div>
  )
}

export default App
