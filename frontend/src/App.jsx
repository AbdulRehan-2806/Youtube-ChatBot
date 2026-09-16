import { useState, useRef, useEffect } from 'react'

const API_BASE = 'http://127.0.0.1:8000'

function App() {
  const [videoUrl, setVideoUrl] = useState('')
  const [videoId, setVideoId] = useState(null)
  const [summary, setSummary] = useState('')
  const [processing, setProcessing] = useState(false)
  const [error, setError] = useState('')

  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)

  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleProcess() {
    if (!videoUrl.trim()) return

    setProcessing(true)
    setError('')
    setSummary('')
    setMessages([])
    setVideoId(null)

    try {
      const res = await fetch(`${API_BASE}/api/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: videoUrl }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Something went wrong processing that video.')
      }

      const data = await res.json()
      setVideoId(data.video_id)
      setSummary(data.summary)
    } catch (err) {
      setError(err.message)
    } finally {
      setProcessing(false)
    }
  }

  async function handleAsk() {
    if (!question.trim() || !videoId) return

    const userMessage = question
    setQuestion('')
    setMessages((prev) => [...prev, { role: 'user', text: userMessage }])
    setAsking(true)

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId, question: userMessage }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Something went wrong.')
      }

      const data = await res.json()
      setMessages((prev) => [...prev, { role: 'assistant', text: data.answer }])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: `Error: ${err.message}` },
      ])
    } finally {
      setAsking(false)
    }
  }

  const isLoading = processing || asking

  return (
    <div className="min-h-screen bg-canvas text-text">
      {isLoading && (
        <div className="scrubber">
          <div className="scrubber-fill" />
        </div>
      )}

      {/* Chrome bar */}
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-2xl items-center gap-2 px-4 py-4">
          <div className="flex gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-accent/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-accent-2/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-text-muted/40" />
          </div>
          <span className="font-display text-lg font-bold tracking-tight">
            Recap
          </span>
        </div>
      </header>

      <div className="mx-auto max-w-2xl px-4 py-12">
        {/* Hero */}
        <h1 className="font-display text-4xl font-bold leading-tight tracking-tight">
          Paste a link. Get the gist.
        </h1>
        <p className="mt-3 max-w-md text-text-muted">
          Recap reads the transcript so you don't have to — then sticks around
          to answer whatever you ask about it.
        </p>

        {/* URL input */}
        <div className="mt-8 flex gap-2">
          <input
            type="text"
            value={videoUrl}
            onChange={(e) => setVideoUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleProcess()}
            placeholder="https://www.youtube.com/watch?v=..."
            className="flex-1 rounded-full border border-border bg-surface px-5 py-3 text-sm outline-none placeholder:text-text-muted focus:border-accent"
          />
          <button
            onClick={handleProcess}
            disabled={processing}
            className="whitespace-nowrap rounded-full bg-accent px-5 py-3 text-sm font-semibold text-canvas transition-opacity disabled:opacity-50"
          >
            {processing ? 'Summarizing…' : 'Get summary'}
          </button>
        </div>

        {error && (
          <div className="mt-4 border-l-2 border-danger py-1 pl-3 text-sm text-danger">
            {error}
          </div>
        )}

        {/* Summary */}
        {summary && videoId && (
          <section className="mt-10">
            <img
              src={`https://img.youtube.com/vi/${videoId}/hqdefault.jpg`}
              alt="Video thumbnail"
              className="w-full rounded-xl border border-border object-cover"
            />
            <h2 className="mt-5 font-display text-xl font-semibold">
              Summary
            </h2>
            <div className="mt-2 whitespace-pre-wrap border-l-2 border-accent/60 pl-4 leading-relaxed text-text/90">
              {summary}
            </div>
          </section>
        )}

        {/* Chat */}
        {videoId && (
          <section className="mt-12">
            <h2 className="font-display text-xl font-semibold">
              Ask about this video
            </h2>

            <div className="mt-4 space-y-3">
              {messages.map((msg, i) =>
                msg.role === 'user' ? (
                  <div
                    key={i}
                    className="ml-auto max-w-[80%] rounded-2xl rounded-br-sm bg-accent/15 px-4 py-2.5 text-sm text-text"
                  >
                    {msg.text}
                  </div>
                ) : (
                  <div
                    key={i}
                    className="mr-auto max-w-[85%] whitespace-pre-wrap rounded-r-lg border-l-2 border-accent-2 bg-surface px-4 py-2.5 text-sm leading-relaxed text-text/90"
                  >
                    {msg.text}
                  </div>
                )
              )}

              {asking && (
                <div className="mr-auto rounded-r-lg border-l-2 border-accent-2 bg-surface px-4 py-2.5 text-sm text-text-muted">
                  Thinking…
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className="mt-4 flex gap-2">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
                placeholder="Ask a question about the video…"
                className="flex-1 rounded-full border border-border bg-surface px-5 py-3 text-sm outline-none placeholder:text-text-muted focus:border-accent"
              />
              <button
                onClick={handleAsk}
                disabled={asking}
                className="rounded-full bg-surface-2 px-5 py-3 text-sm font-semibold text-text transition-opacity disabled:opacity-50"
              >
                Send
              </button>
            </div>
          </section>
        )}
      </div>
    </div>
  )
}

export default App