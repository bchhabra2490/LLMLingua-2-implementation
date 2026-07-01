import { useEffect, useState } from "react";

const DEFAULT_PROMPT =
  "You are a helpful assistant. Please summarize the following document while preserving all key facts, names, dates, and numerical values. The document discusses climate policy, renewable energy adoption, and economic impacts across multiple regions over the last decade.";

export default function App() {
  const [text, setText] = useState(DEFAULT_PROMPT);
  const [ratio, setRatio] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [checkpoint, setCheckpoint] = useState("");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelError, setModelError] = useState("");

  useEffect(() => {
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => {
        setCheckpoint(data.checkpoint || "");
        setModelLoaded(Boolean(data.model_loaded));
        setModelError(data.model_error || "");
      })
      .catch(() => setCheckpoint("unavailable"));
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch("/api/compress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, ratio }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Compression failed.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>LLMLingua-2 Compressor</h1>
        <p>
          Enter a long prompt and choose a compression ratio. The model scores each
          word and keeps the most important ones in original order.
        </p>
        {checkpoint && (
          <span className="badge">Checkpoint: {checkpoint}</span>
        )}
        {!modelLoaded && !modelError && (
          <span className="badge">Loading model...</span>
        )}
        {modelError && (
          <span className="badge" style={{ borderColor: "rgba(239, 68, 68, 0.4)", color: "#fca5a5" }}>
            Model error: {modelError}
          </span>
        )}
      </header>

      <div className="card">
        <form className="form-grid" onSubmit={handleSubmit}>
          <div>
            <label htmlFor="prompt">Prompt</label>
            <textarea
              id="prompt"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your long prompt here..."
              required
            />
          </div>

          <div>
            <label htmlFor="ratio">Compression ratio</label>
            <div className="ratio-row">
              <input
                id="ratio"
                type="range"
                min="0.05"
                max="1"
                step="0.05"
                value={ratio}
                onChange={(e) => setRatio(parseFloat(e.target.value))}
              />
              <span className="ratio-value">{(ratio * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="actions">
            <button type="submit" disabled={loading || !text.trim()}>
              {loading ? "Compressing..." : "Compress"}
            </button>
          </div>
        </form>

        {error && <div className="error">{error}</div>}

        <section className="result">
          <h2>Output</h2>
          {result ? (
            <>
              <div className="stats">
                <span className="stat">
                  Words: <strong>{result.original_word_count}</strong> →{" "}
                  <strong>{result.compressed_word_count}</strong>
                </span>
                <span className="stat">
                  Target: <strong>{(result.target_ratio * 100).toFixed(0)}%</strong>
                </span>
                <span className="stat">
                  Actual: <strong>{(result.actual_ratio * 100).toFixed(1)}%</strong>
                </span>
              </div>
              <div className="output-box">{result.compressed}</div>
            </>
          ) : (
            <div className="output-box empty-output">
              Compressed text will appear here after you submit.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
