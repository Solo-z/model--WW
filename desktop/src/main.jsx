import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const examples = [
  "dark aggressive trap, D minor, 140 BPM, heavy 808 bass, rolling hi-hats",
  "piano ballad, E minor, 70 BPM, emotional, soft vocals",
  "lo-fi chill beat, A minor, 85 BPM, vinyl crackle, mellow piano",
  "epic cinematic orchestral, G minor, 100 BPM, dramatic strings",
  "smooth R&B, Eb major, 90 BPM, soulful chords, warm keys",
];

function fileSummary(files) {
  const counts = files.reduce((acc, f) => {
    acc[f.kind] = (acc[f.kind] || 0) + 1;
    return acc;
  }, {});
  return [
    counts.audio ? `${counts.audio} audio` : null,
    counts.drums || counts.bass || counts.vocals || counts.other
      ? `${(counts.drums || 0) + (counts.bass || 0) + (counts.vocals || 0) + (counts.other || 0)} stems`
      : null,
    counts.midi ? `${counts.midi} midi` : null,
  ].filter(Boolean).join(" · ");
}

function App() {
  const [prompt, setPrompt] = useState(examples[0]);
  const [messages, setMessages] = useState([
    {
      role: "room",
      title: "ROOM ready",
      body: "Describe the track you want. I will generate audio, stems, and MIDI, then save everything locally.",
    },
  ]);
  const [stems, setStems] = useState(true);
  const [midi, setMidi] = useState(true);
  const [duration, setDuration] = useState(30);
  const [steps, setSteps] = useState(8);
  const [guidance, setGuidance] = useState(7);
  const [seed, setSeed] = useState(-1);
  const [progress, setProgress] = useState({ stage: "idle", pct: 0, detail: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [generation, setGeneration] = useState(null);
  const [settings, setSettings] = useState({ reaperDir: "", autoOpenFolder: true });

  useEffect(() => {
    window.room.getSettings().then(setSettings).catch(() => {});
    const off = window.room.onProgress(setProgress);
    return off;
  }, []);

  const generatedSummary = useMemo(() => {
    if (!generation?.files?.length) return "";
    return fileSummary(generation.files);
  }, [generation]);

  async function generate() {
    setBusy(true);
    setError("");
    setGeneration(null);
    setMessages((prev) => [
      ...prev,
      { role: "user", title: "You", body: prompt },
      { role: "room", title: "ROOM", body: "Generating a production-ready direction..." },
    ]);
    try {
      const result = await window.room.generate({
        prompt,
        stems,
        midi,
        duration,
        steps,
        guidance,
        seed,
      });
      setGeneration(result);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          role: "room",
          title: "ROOM",
          body: `Generated ${fileSummary(result.files) || "audio"}. Files saved locally.`,
          generation: result,
        },
      ]);
      if (settings.autoOpenFolder) {
        await window.room.revealGeneration(result.id);
      }
    } catch (err) {
      const message = err?.message || "Generation failed.";
      setError(message);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "room", title: "ROOM error", body: message, error: true },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function sendToReaper() {
    if (!generation?.id) return;
    setError("");
    const res = await window.room.sendToReaper(generation.id);
    if (!res.ok) {
      setError(res.error || "Could not send to REAPER.");
    } else {
      setProgress({ stage: "reaper", pct: 100, detail: `Sent ${res.count} tracks to REAPER` });
    }
  }

  async function installReaperScript() {
    setError("");
    const res = await window.room.installReaperScript();
    if (!res.ok) {
      setError(res.error || "Could not install REAPER script.");
      return;
    }
    setProgress({
      stage: "reaper",
      pct: 100,
      detail: "REAPER script copied — load it from the opened folder",
    });
  }

  async function saveSettings(next) {
    const merged = { ...settings, ...next };
    setSettings(merged);
    await window.room.saveSettings(merged);
  }

  return (
    <main className="workspace">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">ROOM</span>
          <span>Music workspace</span>
        </div>

        <button type="button" className="new-chat" onClick={() => setMessages([{ role: "room", title: "ROOM ready", body: "New session. Describe the track." }])}>
          New session
        </button>

        <div className="side-section">
          <div className="side-label">Today</div>
          {examples.slice(0, 4).map((ex, index) => (
            <button type="button" className="session" key={ex} onClick={() => setPrompt(ex)}>
              <span>{`0${index + 1}`}</span>
              <p>{ex}</p>
            </button>
          ))}
        </div>

        <div className="side-footer">
          <span>Backend</span>
          <strong>HuggingFace Space</strong>
        </div>
      </aside>

      <section className="chat">
        <header className="topbar">
          <div>
            <span className="crumb">ROOM / Generate</span>
            <h1>Prompt to production session</h1>
          </div>
          <span className={`status-dot ${busy ? "busy" : ""}`}>{busy ? "Generating" : "Ready"}</span>
        </header>

        <div className="thread">
          {messages.map((message, index) => (
            <article className={`message ${message.role} ${message.error ? "error-msg" : ""}`} key={`${message.title}-${index}`}>
              <div className="avatar">{message.role === "user" ? "YOU" : "RM"}</div>
              <div className="bubble">
                <div className="message-title">{message.title}</div>
                <p>{message.body}</p>
                {message.generation?.audioPath && (
                  <div className="inline-result">
                    <audio src={`file://${message.generation.audioPath}`} controls>
                      <track kind="captions" />
                    </audio>
                    <div className="result-actions">
                      <button type="button" onClick={() => window.room.revealGeneration(message.generation.id)}>Open folder</button>
                      <button type="button" onClick={sendToReaper}>Send to REAPER</button>
                    </div>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>

        <section className="composer">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask ROOM for a track..."
          />
          <div className="composer-row">
            <label><input type="checkbox" checked={stems} onChange={(e) => setStems(e.target.checked)} /> Stems</label>
            <label><input type="checkbox" checked={midi} onChange={(e) => setMidi(e.target.checked)} /> MIDI</label>
            <button type="button" className="generate" disabled={busy} onClick={generate}>
              {busy ? "Generating" : "Generate"}
            </button>
          </div>
        </section>
      </section>

      <aside className="rightbar">
        <div className="panel-card">
          <h2>Current output</h2>
          {generation?.audioPath ? (
            <>
              <audio src={`file://${generation.audioPath}`} controls>
                <track kind="captions" />
              </audio>
              <p>{generatedSummary || generation.info || "Ready"}</p>
              <div className="result-actions vertical">
                <button type="button" onClick={() => window.room.revealGeneration(generation.id)}>Open output folder</button>
                <button type="button" onClick={sendToReaper}>Send to REAPER</button>
              </div>
            </>
          ) : (
            <p>No generation yet.</p>
          )}
        </div>

        {(busy || progress.stage !== "idle") && (
          <div className="panel-card">
            <h2>Status</h2>
            <div className="progress">
              <div className="progress-top">
                <span>{progress.detail || progress.stage}</span>
                <span>{Math.round(progress.pct || 0)}%</span>
              </div>
              <div className="bar"><div style={{ width: `${progress.pct || 0}%` }} /></div>
            </div>
          </div>
        )}

        <details className="panel-card" open>
          <summary>Advanced</summary>
          <div className="grid">
            <label>Duration <input type="number" min="10" max="300" value={duration} onChange={(e) => setDuration(Number(e.target.value))} /></label>
            <label>Steps <input type="number" min="4" max="50" value={steps} onChange={(e) => setSteps(Number(e.target.value))} /></label>
            <label>Guidance <input type="number" min="1" max="15" step="0.5" value={guidance} onChange={(e) => setGuidance(Number(e.target.value))} /></label>
            <label>Seed <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></label>
          </div>
          <label className="wide">
            REAPER command folder
            <input value={settings.reaperDir || ""} onChange={(e) => saveSettings({ reaperDir: e.target.value })} />
          </label>
          <button type="button" className="secondary-action" onClick={installReaperScript}>
            Install REAPER Script
          </button>
          <label className="check">
            <input type="checkbox" checked={settings.autoOpenFolder} onChange={(e) => saveSettings({ autoOpenFolder: e.target.checked })} />
            Open folder after generation
          </label>
        </details>

        {error && <div className="error">{error}</div>}
      </aside>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
