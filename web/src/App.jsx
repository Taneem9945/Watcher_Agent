import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Pause, Play, RotateCcw, StepForward, Wifi, WifiOff } from 'lucide-react'

const emptyState = {
  row_index: 0,
  events_seen: 0,
  windows_emitted: 0,
  buffer_pending: 0,
  finished: false,
  max_windows: null,
  total_rows: 0,
  total_windows: 0,
}

function getLlmPacket(record) {
  return record?.llm_packet ?? null
}

function getWindowMetadata(record) {
  const packet = getLlmPacket(record)
  return packet?.window_metadata ?? record?.window ?? null
}

function getSourceMeaning(record) {
  const packet = getLlmPacket(record)
  return packet?.source_meaning ?? null
}

function getStableIdentifiers(record) {
  const packet = getLlmPacket(record)
  return packet?.stable_identifiers ?? null
}

function getMissingData(record) {
  const packet = getLlmPacket(record)
  return packet?.missing_data ?? null
}

function getReadableEvidence(record) {
  const packet = getLlmPacket(record)
  return packet?.readable_event_evidence ?? []
}

function valueList(values) {
  if (!Array.isArray(values) || values.length === 0) return 'n/a'
  return values.slice(0, 6).join(', ')
}

function countMapEntries(map) {
  if (!map || typeof map !== 'object') return []
  return Object.entries(map).sort((a, b) => Number(b[1]) - Number(a[1]))
}

function displayAssessment(assessment) {
  if (!assessment) return null
  const { llm_input_packet, raw_response_text, ...rest } = assessment
  return rest
}

function App() {
  const [status, setStatus] = useState(emptyState)
  const [connected, setConnected] = useState(false)
  const [running, setRunning] = useState(false)
  const [delay, setDelay] = useState(1)
  const [currentRecord, setCurrentRecord] = useState(null)
  const [history, setHistory] = useState([])
  const [rawMessage, setRawMessage] = useState('')
  const [streamError, setStreamError] = useState('')
  const wsRef = useRef(null)

  const wsUrl = useMemo(() => {
    if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}/ws`
  }, [])

  useEffect(() => {
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
    }

    ws.onclose = () => {
      setConnected(false)
      setRunning(false)
    }

    ws.onerror = () => {
      setConnected(false)
    }

    ws.onmessage = (event) => {
      setRawMessage(event.data)
      let payload
      try {
        payload = JSON.parse(event.data)
      } catch {
        return
      }
      if (payload.type === 'state') {
        setStatus(payload.payload)
        if (payload.payload.finished) {
          setRunning(false)
        }
      } else if (payload.type === 'window') {
        setCurrentRecord(payload.payload)
        setHistory((prev) => {
          const incomingId = payload.payload.window_id
          const withoutDuplicate = prev.filter((item) => item.window_id !== incomingId)
          return [payload.payload, ...withoutDuplicate].slice(0, 24)
        })
        setStatus((prev) => ({
          ...prev,
          row_index: payload.payload.window_id + 1,
          events_seen: payload.payload.events_seen,
          windows_emitted: payload.payload.window_id + 1,
          buffer_pending: payload.payload.pending_buffer_rows,
        }))
      } else if (payload.type === 'done') {
        setStatus(payload.payload)
        setRunning(false)
      } else if (payload.type === 'error') {
        setStreamError(String(payload.payload ?? 'Unknown stream error'))
        setRunning(false)
      }
    }

    return () => ws.close()
  }, [wsUrl])

  function send(message) {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }

  function handleStart() {
    setStreamError('')
    setRunning(true)
    send({ type: 'resume' })
  }

  function handlePause() {
    setRunning(false)
    send({ type: 'pause' })
  }

  function handleStep() {
    setStreamError('')
    send({ type: 'step' })
  }

  function handleReset() {
    setRunning(false)
    setHistory([])
    setCurrentRecord(null)
    setStreamError('')
    send({ type: 'reset' })
  }

  function handleDelayChange(event) {
    const next = Number(event.target.value)
    setDelay(next)
    send({ type: 'set_delay', value: next })
  }

  const currentAssessment = currentRecord?.llm_assessment ?? null
  const currentPacket = getLlmPacket(currentRecord)
  const currentAssessmentDisplay = displayAssessment(currentAssessment)
  const currentWindow = getWindowMetadata(currentRecord)
  const currentSourceMeaning = getSourceMeaning(currentRecord)
  const currentStableIdentifiers = getStableIdentifiers(currentRecord)
  const currentMissingData = getMissingData(currentRecord)
  const currentEvidence = getReadableEvidence(currentRecord)
  const currentWindowId = currentWindow?.window_id ?? currentRecord?.window_id ?? currentAssessment?.window_id ?? null
  const assessedCount = history.filter((item) => item.llm_assessment).length

  return (
    <div className="shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Watcher Agent</div>
          <h1>Mixed Evidence Watcher</h1>
        </div>
        <div className={`connection ${connected ? 'up' : 'down'}`}>
          {connected ? <Wifi size={16} /> : <WifiOff size={16} />}
          <span>{connected ? 'Connected' : 'Offline'}</span>
        </div>
      </header>

      <section className="controls">
        <button className="control primary" onClick={handleStart}>
          <Play size={16} /> Play Replay
        </button>
        <button className="control" onClick={handlePause}>
          <Pause size={16} /> Pause
        </button>
        <button className="control" onClick={handleStep}>
          <StepForward size={16} /> Step
        </button>
        <button className="control" onClick={handleReset}>
          <RotateCcw size={16} /> Reset
        </button>
        <label className="delay">
          <span>Delay</span>
          <input
            type="range"
            min="0.05"
            max="3"
            step="0.25"
            value={delay}
            onChange={handleDelayChange}
          />
          <strong>{delay.toFixed(2)}s</strong>
        </label>
      </section>

      <main className="grid">
        <section className="panel status-panel">
          <div className="panel-title">Stream State</div>
          <div className="stat-grid">
            <div>
              <span>Windows Processed</span>
              <strong>{status.row_index}/{status.total_windows ?? status.total_rows}</strong>
            </div>
            <div>
              <span>Events Seen</span>
              <strong>{status.events_seen}</strong>
            </div>
            <div>
              <span>60s Windows</span>
              <strong>{status.windows_emitted}</strong>
            </div>
            <div>
              <span>LLM Assessments</span>
              <strong>{assessedCount}</strong>
            </div>
          </div>
          <div className="meta-row">
            <span className={running ? 'pill live' : 'pill idle'}>{running ? 'replaying' : 'paused'}</span>
            <span className="pill">{status.finished ? 'done' : 'window replay'}</span>
            <span className="pill">pending: {status.buffer_pending}</span>
          </div>
        </section>
      </main>

      <section className="bottom-grid">
        <section className="panel history-panel">
          <div className="panel-title">Window Timeline</div>
          {streamError ? <div className="empty inline">Stream error: {streamError}</div> : null}
          <div className="history">
            {history.length ? history.map((item) => {
              const packet = getLlmPacket(item)
              const metadata = getWindowMetadata(item)
              const sourceTypes = Object.keys(packet?.source_meaning?.source_counts ?? metadata?.source_counts ?? {})
              return (
                <button key={item.window_id} className="history-item" onClick={() => setCurrentRecord(item)}>
                  <div>
                    <strong>Window {metadata?.window_id ?? item.window_id}</strong>
                    <span>
                      {metadata?.start_timestamp && metadata?.end_timestamp
                        ? `${metadata.start_timestamp} - ${metadata.end_timestamp}`
                        : `${item.source_row_start ?? 'n/a'}-${item.source_row_end ?? 'n/a'}`}
                    </span>
                    <span>{metadata?.event_count ?? metadata?.num_flows ?? 'n/a'} events - {sourceTypes.join(', ') || 'sources n/a'}</span>
                  </div>
                  <div className="history-verdicts">
                    <span className={`badge badge-${item.llm_assessment?.severity ?? 'neutral'}`}>
                      LLM: {item.llm_assessment?.assessment ?? 'waiting'}
                    </span>
                    <span className="badge badge-neutral">live Ollama</span>
                  </div>
                </button>
              )
            }) : <div className="empty inline">No windows yet.</div>}
          </div>
        </section>

        <section className="panel selected-panel">
          <div className="panel-title">Selected Window Assessment</div>
          {currentRecord ? (
            <div className="assessment-layout single">
              <section className="panel compare-panel inset">
                <div className="compare-head">
                  <div>
                    <div className="focus-label">LLM assessment</div>
                    <h2>{currentAssessment?.assessment ?? 'waiting'}</h2>
                    <div className="compare-meta">
                      Window {currentWindowId ?? 'n/a'} - {currentWindow?.event_count ?? currentWindow?.num_flows ?? 'n/a'} events
                      {' '} - new Ollama assessment
                    </div>
                  </div>
                  <div className={`risk risk-${currentAssessment?.severity ?? 'neutral'}`}>{currentAssessment?.severity ?? 'n/a'}</div>
                </div>
                {currentAssessment ? (
                  <div className="llm">
                    <div className="llm-head">
                      <strong>{currentAssessment.assessment}</strong>
                      <span>{currentAssessment.severity}</span>
                    </div>
                    <p>{currentAssessment.summary}</p>
                    <div className="chips">
                      {(currentAssessment.recommended_actions || []).slice(0, 3).map((item, index) => (
                        <span key={index} className="chip">{item}</span>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="empty">The LLM assessment will appear here after a window is assessed.</div>
                )}
              </section>

            </div>
          ) : (
            <div className="empty">Select a window to inspect its assessment.</div>
          )}
        </section>

        <section className="panel grounding-panel">
          <div className="panel-title">Grounding Facts</div>
          {currentRecord ? (
            <div className="grounding-grid">
              <section className="fact-card">
                <div className="focus-label">Stable identifiers</div>
                <dl>
                  <dt>Source IPs</dt><dd>{valueList(currentStableIdentifiers?.src_ips)}</dd>
                  <dt>Destination IPs</dt><dd>{valueList(currentStableIdentifiers?.dst_ips)}</dd>
                  <dt>Hosts</dt><dd>{valueList(currentStableIdentifiers?.hosts)}</dd>
                  <dt>Users</dt><dd>{valueList(currentStableIdentifiers?.usernames)}</dd>
                  <dt>UIDs</dt><dd>{valueList(currentStableIdentifiers?.uids)}</dd>
                </dl>
              </section>
              <section className="fact-card">
                <div className="focus-label">Source meaning</div>
                <div className="count-list">
                  {countMapEntries(currentSourceMeaning?.source_counts).map(([name, count]) => (
                    <span key={name} className="count-pill">{name}: {count}</span>
                  ))}
                  {!countMapEntries(currentSourceMeaning?.source_counts).length ? <span className="muted">n/a</span> : null}
                </div>
              </section>
              <section className="fact-card">
                <div className="focus-label">Missing data</div>
                <div className="count-list">
                  {countMapEntries(currentMissingData?.missing_field_counts).map(([name, count]) => (
                    <span key={name} className="count-pill warn">{name}: {count}</span>
                  ))}
                  {!countMapEntries(currentMissingData?.missing_field_counts).length ? <span className="muted">No missing fields reported</span> : null}
                </div>
              </section>
            </div>
          ) : (
            <div className="empty">Select a window to inspect stable identifiers, sources, and missing data.</div>
          )}
        </section>

        <section className="panel raw-panel">
          <div className="panel-title">Evidence And Debug</div>
          {currentRecord ? (
            <div className="raw-split three">
              <section className="raw-block">
                <div className="focus-label">Readable evidence</div>
                <pre>{JSON.stringify(currentEvidence, null, 2)}</pre>
              </section>
              <section className="raw-block">
                <div className="focus-label">LLM assessment</div>
                <pre>{JSON.stringify(currentAssessmentDisplay ?? {}, null, 2)}</pre>
              </section>
              <section className="raw-block">
                <div className="focus-label">LLM input packet</div>
                <pre>{JSON.stringify(currentPacket ?? {}, null, 2)}</pre>
              </section>
            </div>
          ) : (
            <pre>{rawMessage || '{}'}</pre>
          )}
        </section>
      </section>
    </div>
  )
}

export default App
