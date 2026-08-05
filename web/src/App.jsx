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
}

function formatNumber(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a'
  if (Math.abs(value) < 0.001) return value.toExponential(2)
  return value.toFixed(4)
}

function App() {
  const [status, setStatus] = useState(emptyState)
  const [connected, setConnected] = useState(false)
  const [running, setRunning] = useState(false)
  const [delay, setDelay] = useState(0.1)
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
        setHistory((prev) => [payload.payload, ...prev].slice(0, 24))
        setStatus((prev) => ({
          ...prev,
          row_index: payload.payload.events_seen,
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

  const currentSignal = currentRecord?.mamba_signal_packet?.signal_rows?.[0] ?? null
  const currentMemory = currentRecord?.stream_memory_after ?? null
  const currentAssessment = currentRecord?.llm_assessment ?? null
  const currentSummary = currentRecord?.context_packet?.window_summary ?? null
  const currentWindowId = currentRecord?.window_id ?? currentAssessment?.window_id ?? currentSignal?.window_id ?? null

  return (
    <div className="shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Watcher Agent</div>
          <h1>Live Replay Simulator</h1>
        </div>
        <div className={`connection ${connected ? 'up' : 'down'}`}>
          {connected ? <Wifi size={16} /> : <WifiOff size={16} />}
          <span>{connected ? 'Connected' : 'Offline'}</span>
        </div>
      </header>

      <section className="controls">
        <button className="control primary" onClick={handleStart}>
          <Play size={16} /> Start
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
            step="0.05"
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
              <span>Rows</span>
              <strong>{status.row_index}/{status.total_rows}</strong>
            </div>
            <div>
              <span>Events</span>
              <strong>{status.events_seen}</strong>
            </div>
            <div>
              <span>Windows</span>
              <strong>{status.windows_emitted}</strong>
            </div>
            <div>
              <span>Buffer</span>
              <strong>{status.buffer_pending}</strong>
            </div>
          </div>
          <div className="meta-row">
            <span className={running ? 'pill live' : 'pill idle'}>{running ? 'running' : 'paused'}</span>
            <span className="pill">{status.finished ? 'done' : 'streaming'}</span>
          </div>
        </section>
      </main>

      <section className="bottom-grid">
        <section className="panel history-panel">
          <div className="panel-title">Window Timeline</div>
          {streamError ? <div className="empty inline">Stream error: {streamError}</div> : null}
          <div className="history">
            {history.length ? history.map((item) => {
              const signal = item.mamba_signal_packet.signal_rows[0]
              return (
                <button key={item.window_id} className="history-item" onClick={() => setCurrentRecord(item)}>
                  <div>
                    <strong>Window {item.window_id}</strong>
                    <span>{item.source_row_start}-{item.source_row_end}</span>
                  </div>
                  <div className="history-verdicts">
                    <span className={`badge badge-${signal.risk_level}`}>Mamba: {signal.prediction_label}</span>
                    <span className={`badge badge-${item.llm_assessment?.severity ?? 'neutral'}`}>
                      LLM: {item.llm_assessment?.assessment ?? 'waiting'}
                    </span>
                  </div>
                </button>
              )
            }) : <div className="empty inline">No windows yet.</div>}
          </div>
        </section>

        <section className="panel selected-panel">
          <div className="panel-title">Selected Window Comparison</div>
          {currentSignal ? (
            <div className="comparison-grid compact">
              <section className="panel compare-panel inset">
                <div className="compare-head">
                  <div>
                    <div className="focus-label">Mamba signal</div>
                    <h2>{currentSignal.prediction_label}</h2>
                    <div className="compare-meta">
                      Window {currentWindowId ?? 'n/a'} - {currentSummary?.num_flows ?? 'n/a'} flows - {currentSummary?.num_features ?? 'n/a'} features
                    </div>
                  </div>
                  <div className={`risk risk-${currentSignal.risk_level}`}>{currentSignal.risk_level}</div>
                </div>
                <div className="meta-row compact">
                  <span className={`badge badge-${currentSignal.risk_level}`}>Mamba: {currentSignal.prediction_label}</span>
                </div>
                <div className="focus-grid">
                  <div><span>Attack Prob.</span><strong>{formatNumber(currentSignal.attack_probability)}</strong></div>
                  <div><span>Confidence</span><strong>{formatNumber(currentSignal.confidence)}</strong></div>
                  <div><span>Trend</span><strong>{currentMemory?.attack_probability_trend ?? 'n/a'}</strong></div>
                  <div><span>Streak</span><strong>{currentMemory?.prediction_streak ?? 'n/a'}</strong></div>
                </div>
                <div className="delta">
                  <span>Memory delta</span>
                  <pre>{JSON.stringify(currentMemory?.last_delta ?? {}, null, 2)}</pre>
                </div>
              </section>

              <section className="panel compare-panel inset">
                <div className="compare-head">
                  <div>
                    <div className="focus-label">LLM interpretation</div>
                    <h2>{currentAssessment?.assessment ?? 'waiting'}</h2>
                    <div className="compare-meta">
                      Window {currentAssessment?.window_id ?? currentWindowId ?? 'n/a'} - encoder reasoning from evidence packet
                    </div>
                  </div>
                  <div className="risk risk-neutral">{currentAssessment?.severity ?? 'n/a'}</div>
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
                  <div className="empty">The LLM assessment will appear here after the first window.</div>
                )}
              </section>
            </div>
          ) : (
            <div className="empty">Select a window to inspect its comparison.</div>
          )}
        </section>

        <section className="panel raw-panel">
          <div className="panel-title">Raw Data</div>
          {currentRecord ? (
            <div className="raw-split">
              <section className="raw-block">
                <div className="focus-label">Mamba packet</div>
                <pre>{JSON.stringify(currentRecord.mamba_signal_packet ?? {}, null, 2)}</pre>
              </section>
              <section className="raw-block">
                <div className="focus-label">LLM input packet</div>
                <pre>{JSON.stringify(currentRecord.llm_input_packet ?? currentRecord.llm_assessment?.llm_input_packet ?? {}, null, 2)}</pre>
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
