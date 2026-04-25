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
      if (settings.autoOpenFolder) {
        await window.room.revealGeneration(result.id);
      }
    } catch (err) {
      setError(err?.message || "Generation failed.");
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

  async function saveSettings(next) {
    const merged = { ...settings, ...next };
    setSettings(merged);
    await window.room.saveSettings(merged);
  }

  return (
    <main className="app">
      <section className="hero">
        <h1>ROOM</h1>
        <p>A Foundation Model for Music Production</p>
      </section>

      <section className="panel">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="describe the track — genre, key, BPM, mood, instruments…"
        />

        <div className="toggles">
          <label><input type="checkbox" checked={stems} onChange={(e) => setStems(e.target.checked)} /> Stems</label>
          <label><input type="checkbox" checked={midi} onChange={(e) => setMidi(e.target.checked)} /> MIDI</label>
        </div>

        <button className="generate" disabled={busy} onClick={generate}>
          {busy ? "Generating" : "⏵ Generate"}
        </button>

        {(busy || progress.stage !== "idle") && (
          <div className="progress">
            <div className="progress-top">
              <span>{progress.detail || progress.stage}</span>
              <span>{Math.round(progress.pct || 0)}%</span>
            </div>
            <div className="bar"><div style={{ width: `${progress.pct || 0}%` }} /></div>
          </div>
        )}

        {generation?.audioPath && (
          <div className="result">
            <audio src={`file://${generation.audioPath}`} controls />
            <div className="result-actions">
              <button onClick={() => window.room.revealGeneration(generation.id)}>Open Folder</button>
              <button onClick={sendToReaper}>Send to REAPER</button>
            </div>
            <p>{generatedSummary || generation.info || "Ready"}</p>
          </div>
        )}

        {error && <div className="error">{error}</div>}
      </section>

      <section className="settings">
        <details>
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
          <label className="check">
            <input type="checkbox" checked={settings.autoOpenFolder} onChange={(e) => saveSettings({ autoOpenFolder: e.target.checked })} />
            Open output folder after generation
          </label>
        </details>

        <details>
          <summary>Examples</summary>
          <div className="examples">
            {examples.map((ex) => <button key={ex} onClick={() => setPrompt(ex)}>{ex}</button>)}
          </div>
        </details>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
