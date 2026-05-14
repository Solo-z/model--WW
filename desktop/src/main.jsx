import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import roomLogoUrl from "../assets/room_logo.png";

/** Activity rail icon */
function IconChat() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.85" strokeLinecap="round" aria-hidden="true">
      <path d="M8 11h8M8 15h5" />
      <path d="M6.5 4h11A2.5 2.5 0 0 1 20 6.5v8a2.5 2.5 0 0 1-2.5 2.5h-7L6 21v-4H6.5A2.5 2.5 0 0 1 4 14.5v-8A2.5 2.5 0 0 1 6.5 4z" />
    </svg>
  );
}

const examples = [
  "dark aggressive trap, D minor, 140 BPM, heavy 808 bass, rolling hi-hats",
  "piano ballad, E minor, 70 BPM, emotional, soft vocals",
  "lo-fi chill beat, A minor, 85 BPM, vinyl crackle, mellow piano",
  "epic cinematic orchestral, G minor, 100 BPM, dramatic strings",
  "smooth R&B, Eb major, 90 BPM, soulful chords, warm keys",
];

/** Sidebar “previous chats” — tap to load the idea into the prompt (melody vs preset). */
const mockSidebarIdeas = [
  {
    id: "sb1",
    title: "Teenage love piano",
    mode: "songs",
    prompt: "Piano song for teenage love, tender verses, huge emotional chorus, 82 BPM, soft strings, bedroom warmth",
  },
  {
    id: "sb2",
    title: "3am bedroom trap",
    mode: "songs",
    prompt: "Late-night trap, lonely melodic hook, 142 BPM, filtered 808, minimal hats, rain sample bed",
  },
  {
    id: "sb3",
    title: "Summer festival EDM",
    mode: "songs",
    prompt: "Festival progressive house, F major, 126 BPM, saw leads, sidechain pads, big snare fills into drop",
  },
  {
    id: "sb4",
    title: "Neo-soul groove",
    mode: "songs",
    prompt: "Neo-soul session, Rhodes, live bass, dotted hats, 92 BPM, warm tape, stacked harmonies",
  },
  {
    id: "sb5",
    title: "Cinematic trailer",
    mode: "songs",
    prompt: "Epic trailer brass, huge taiko pulse, 100 BPM, tension risers, drops into silence then one big hit",
  },
  {
    id: "sb6",
    title: "Bright pluck preset",
    mode: "sounds",
    prompt: "Bright plucky one-shot for house chords, short decay, tiny chorus shimmer, playable from C3 up",
  },
  {
    id: "sb7",
    title: "808 slide preset",
    mode: "sounds",
    prompt: "West-coast 808 bounce, long decay sub, pitch glide between notes, click transient, mono punch",
  },
  {
    id: "sb8",
    title: "Ambient pad layer",
    mode: "sounds",
    prompt: "Long evolving pad, slow filter sweep, wide stereo shimmer, low mid body for layering under vocals",
  },
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
  ].filter(Boolean).join(" / ");
}

function formatTime(totalSeconds) {
  const seconds = Math.max(0, Number(totalSeconds) || 0);
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function audioSource(generation) {
  return generation?.audioUrl || (generation?.audioPath ? `file://${generation.audioPath}` : "");
}

function mockAudioUrl(frequency = 220) {
  const sampleRate = 22050;
  const seconds = 5;
  const samples = sampleRate * seconds;
  const dataSize = samples * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const write = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));

  write(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, dataSize, true);

  for (let i = 0; i < samples; i += 1) {
    const value = Math.sin((2 * Math.PI * frequency * i) / sampleRate) * 0.25;
    view.setInt16(44 + i * 2, Math.floor(value * 32767), true);
  }

  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return `data:audio/wav;base64,${btoa(binary)}`;
}

function createPreviewRoomApi() {
  return {
    getSettings: async () => ({ reaperDir: "", autoOpenFolder: false }),
    saveSettings: async (settings) => settings,
    onProgress: () => () => {},
    revealGeneration: async () => ({ ok: true }),
    sendToReaper: async () => ({ ok: true, count: 1 }),
    installReaperScript: async () => ({ ok: true }),
    generate: async (payload) => {
      await new Promise((resolve) => setTimeout(resolve, 450 + Math.random() * 600));
      const seed = Number(payload.seed) >= 0 ? Number(payload.seed) : Math.floor(Math.random() * 9999);
      const id = `preview_${seed}_${Math.random().toString(16).slice(2, 8)}`;
      const files = [
        { name: `take_${seed}.wav`, kind: "audio", path: id, url: "" },
        ...(payload.stems ? ["drums", "bass", "vocals", "other"].map((kind) => ({ name: `${kind}_${seed}.wav`, kind, path: `${id}_${kind}`, url: "" })) : []),
        ...(payload.midi ? ["chords", "bass", "melody"].map((name) => ({ name: `${name}_${seed}.mid`, kind: "midi", path: `${id}_${name}`, url: "" })) : []),
      ];
      return {
        ok: true,
        id,
        info: "Preview mock ready",
        audioUrl: mockAudioUrl(180 + (seed % 220)),
        audioPath: "",
        files,
      };
    },
  };
}

const roomApi = window.room || createPreviewRoomApi();

function createPendingTakes(count = 6, type = "song") {
  return Array.from({ length: count }, (_, index) => ({
    id: `pending_${type}_${index + 1}`,
    takeNumber: index + 1,
    type,
    pending: true,
    info: "Generating...",
    files: [],
  }));
}

/** Gentle bow: control point sits *above* the chord so the arc stays in the open gap and does not dip onto the cards. */
function flowBowPath(hubX, hubY, endX, endY) {
  const mx = (hubX + endX) / 2;
  const my = (hubY + endY) / 2;
  const span = Math.hypot(endX - hubX, endY - hubY);
  const sag = 6 + Math.min(20, span * 0.055);
  const cy = my - sag;
  return `M${hubX} ${hubY} Q${mx} ${cy} ${endX} ${endY}`;
}

function MelodyFlow({ children, layoutKey }) {
  const wrapRef = useRef(null);
  const [geom, setGeom] = useState(null);

  const measure = useCallback(() => {
    void layoutKey;
    const wrap = wrapRef.current;
    if (!wrap) return;
    const cards = wrap.querySelectorAll(".take-card");
    if (cards.length < 2) {
      setGeom(null);
      return;
    }
    const wr = wrap.getBoundingClientRect();
    const w = wrap.offsetWidth;
    const h = wrap.offsetHeight;
    const hubX = w / 2;
    const hubY = 12;
    const gapAboveCard = 12;
    const ends = Array.from(cards).map((el) => {
      const r = el.getBoundingClientRect();
      return {
        id: el.dataset.flowId || "",
        x: r.left + r.width / 2 - wr.left,
        y: r.top - wr.top - gapAboveCard,
      };
    });
    setGeom({ w, h, hubX, hubY, ends });
  }, [layoutKey]);

  useLayoutEffect(() => {
    measure();
    const id = requestAnimationFrame(() => measure());
    const wrap = wrapRef.current;
    const ro = wrap ? new ResizeObserver(() => measure()) : null;
    if (wrap && ro) ro.observe(wrap);
    window.addEventListener("resize", measure);
    return () => {
      cancelAnimationFrame(id);
      ro?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [measure]);

  return (
    <div className="take-flow-wrap" ref={wrapRef}>
      {geom && geom.ends.length >= 2 && (
        <svg
          className="flow-overlay-svg"
          width={geom.w}
          height={geom.h}
          viewBox={`0 0 ${geom.w} ${geom.h}`}
          preserveAspectRatio="xMinYMin meet"
          style={{ width: geom.w, height: geom.h }}
          aria-hidden="true"
        >
          <circle className="flow-hub-pixel" cx={geom.hubX} cy={geom.hubY} r={2.5} />
          {geom.ends.map((end) => (
            <path
              key={`flow-line-${end.id}`}
              className="flow-line-path"
              d={flowBowPath(geom.hubX, geom.hubY, end.x, end.y)}
            />
          ))}
        </svg>
      )}
      {children}
    </div>
  );
}

function App() {
  const [mode, setMode] = useState("songs");
  const [prompt, setPrompt] = useState(examples[0]);
  const [references, setReferences] = useState("Mention songs, audio clips, synths, one-shots, or a track here...");
  const [targetContext, setTargetContext] = useState("Example: preset or one-shot for Track 4, lower bass, playable from MIDI");
  const [messages, setMessages] = useState([
    {
      role: "room",
      title: "ROOM ready",
      body: "Describe the track you want. I will generate audio, stems, and MIDI, then save everything locally.",
    },
  ]);
  const stems = true;
  const midi = true;
  const [duration, setDuration] = useState(30);
  const [steps, setSteps] = useState(8);
  const [guidance, setGuidance] = useState(7);
  const [seed, setSeed] = useState(-1);
  const [, setProgress] = useState({ stage: "idle", pct: 0, detail: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [generation, setGeneration] = useState(null);
  const [takeOptions, setTakeOptions] = useState([]);
  const [batchStatus, setBatchStatus] = useState("");
  const [timer, setTimer] = useState(null);
  const [settings, setSettings] = useState({ reaperDir: "", autoOpenFolder: true });
  /** Which sidebar “idea” chip is selected (only for mock list highlight). */
  const [sidebarIdeaId, setSidebarIdeaId] = useState("");

  useEffect(() => {
    roomApi.getSettings().then(setSettings).catch(() => {});
    const off = roomApi.onProgress(setProgress);
    return off;
  }, []);

  useEffect(() => {
    if (!timer?.startedAt) return undefined;
    const tick = window.setInterval(() => {
      setTimer((current) => current ? { ...current, now: Date.now() } : current);
    }, 1000);
    return () => window.clearInterval(tick);
  }, [timer?.startedAt]);

  const timerText = useMemo(() => {
    if (!timer?.startedAt) return "";
    const elapsed = Math.max(0, Math.floor(((timer.now || Date.now()) - timer.startedAt) / 1000));
    const remaining = Math.max(0, Math.ceil(timer.estimateSeconds - elapsed));
    return `${formatTime(elapsed)} elapsed / about ${formatTime(remaining)} left`;
  }, [timer]);

  function generationPayload(overrides = {}) {
    const contextPrompt = mode === "sounds"
      ? `Sound design request: ${prompt}. References/context: ${references}. Target use: ${targetContext}.`
      : prompt;
    return {
      prompt: contextPrompt,
      stems,
      midi,
      duration,
      steps,
      guidance,
      seed,
      ...overrides,
    };
  }

  function estimateSeconds(count = 1) {
    const base = 55;
    const durationCost = Math.max(10, Number(duration) || 30) * 1.7;
    const stepCost = Math.max(4, Number(steps) || 8) * 2.5;
    const stemCost = stems ? 35 : 0;
    const midiCost = midi ? 35 : 0;
    return Math.ceil((base + durationCost + stepCost + stemCost + midiCost) * count);
  }

  function startTimer(label, count = 1) {
    setTimer({
      label,
      startedAt: Date.now(),
      now: Date.now(),
      estimateSeconds: estimateSeconds(count),
    });
  }

  function stopTimer() {
    setTimer((current) => current ? { ...current, now: Date.now(), estimateSeconds: 0 } : current);
  }

  async function generate() {
    setBusy(true);
    setError("");
    setGeneration(null);
    startTimer("Generating take", 1);
    setMessages((prev) => [
      ...prev,
      { role: "user", title: "You", body: prompt },
      { role: "room", title: "ROOM", body: "Generating a production-ready direction..." },
    ]);
    try {
      const result = await roomApi.generate(generationPayload());
      const take = { ...result, takeNumber: 1 };
      setGeneration(take);
      setTakeOptions([take]);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          role: "room",
          title: "ROOM",
          body: `Generated ${fileSummary(result.files) || "audio"}. Playback is ready below.`,
        },
      ]);
      if (settings.autoOpenFolder) {
        await roomApi.revealGeneration(result.id);
      }
    } catch (err) {
      const message = err?.message || "Generation failed.";
      setError(message);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "room", title: "ROOM error", body: message, error: true },
      ]);
    } finally {
      stopTimer();
      setBusy(false);
    }
  }

  async function generateBatch(targetMode = mode) {
    const isSoundMode = targetMode === "sounds";
    const count = isSoundMode ? 4 : 6;
    const type = isSoundMode ? "sound" : "song";
    setMode(targetMode);
    setBusy(true);
    setError("");
    setGeneration(null);
    const pendingTakes = createPendingTakes(count, type);
    setTakeOptions(pendingTakes);
    setBatchStatus(`Starting ${count} ${type}s`);
    startTimer(`Generating ${count} ${type}s`, count);
    setMessages((prev) => [
      ...prev,
      { role: "user", title: "You", body: prompt },
      {
        role: "room",
        title: "ROOM",
        body: isSoundMode
          ? "Four preset / one-shot sound slots are live. Each one becomes playable as the sound preview finishes."
          : "Six melody/song idea slots are live. Each one becomes playable as it finishes.",
      },
    ]);

    try {
      const completed = [];
      const jobs = Array.from({ length: count }, async (_, index) => {
        const nextSeed = Number(seed) >= 0 ? Number(seed) + index : -1;
        const result = await roomApi.generate(generationPayload({ seed: nextSeed }));
        const take = { ...result, takeNumber: index + 1, type };
        completed.push(take);
        completed.sort((a, b) => a.takeNumber - b.takeNumber);
        setBatchStatus(`Finished ${completed.length}/${count}`);
        setGeneration(take);
        setTakeOptions((current) => current.map((option) => option.takeNumber === take.takeNumber ? take : option));
        return take;
      });

      const results = await Promise.all(jobs);
      results.sort((a, b) => a.takeNumber - b.takeNumber);
      setTakeOptions(results);
      setGeneration(results[0]);

      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          role: "room",
          title: "ROOM",
          body: isSoundMode
            ? "Generated 4 preset / one-shot options. Audition them, then send the selected sound direction."
            : `Generated 6 melody/song options. Play them back below, then pick the one you want to push forward${midi ? " with MIDI" : ""}.`,
        },
      ]);
    } catch (err) {
      const message = err?.message || "Batch generation failed.";
      setError(message);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "room", title: "ROOM error", body: message, error: true },
      ]);
    } finally {
      setBatchStatus("");
      stopTimer();
      setBusy(false);
    }
  }

  async function sendToReaper(target = generation) {
    if (!target?.id) return;
    setError("");
    const res = await roomApi.sendToReaper(target.id);
    if (!res.ok) {
      setError(res.error || "Could not send the selected take.");
    } else {
      setGeneration(target);
      setProgress({ stage: "send", pct: 100, detail: `Sent ${res.count} tracks` });
    }
  }

  return (
    <main className="workspace chat-only">
      <nav className="activity-rail" aria-label="ROOM sections">
        <div className="rail-logo" title="ROOM">
          <img src={roomLogoUrl} alt="" />
        </div>
        <button type="button" className="rail-item active" title="Agent workspace" aria-current="page">
          <IconChat />
          <span className="rail-sr-only">Workspace</span>
        </button>
      </nav>

      <aside className="sidebar">
        <div className="brand">
          <img src={roomLogoUrl} alt="ROOM" />
        </div>

        <button
          type="button"
          className="new-chat"
          onClick={() => {
            setMessages([{ role: "room", title: "ROOM ready", body: "New session. Describe the track." }]);
            setTakeOptions([]);
            setGeneration(null);
            setSidebarIdeaId("");
          }}
        >
          New chat
        </button>

        <div className="side-mock-section">
          <div className="side-mock-label">Ideas</div>
          <div className="side-mock-list">
            {mockSidebarIdeas.map((idea) => (
              <button
                type="button"
                key={idea.id}
                className={`mock-chat-row ${sidebarIdeaId === idea.id ? "kind-active" : ""}`}
                onClick={() => {
                  setMode(idea.mode);
                  setPrompt(idea.prompt);
                  setSidebarIdeaId(idea.id);
                }}
              >
                <span className="mock-chat-kind">{idea.mode === "sounds" ? "Preset" : "Melody"}</span>
                <span className="mock-chat-title">{idea.title}</span>
                <span className="mock-chat-snippet">{idea.prompt}</span>
              </button>
            ))}
          </div>
        </div>
      </aside>

      <section className="chat">
        <div className="thread">
          {mode === "sounds" && (
            <section className="sound-context">
              <label>
                References / songs / context
                <input value={references} onChange={(e) => setReferences(e.target.value)} />
              </label>
              <label>
                What should this preset do?
                <input value={targetContext} onChange={(e) => setTargetContext(e.target.value)} />
              </label>
            </section>
          )}

          {messages.map((message, index) => (
            <article className={`message ${message.role} ${message.error ? "error-msg" : ""}`} key={`${message.title}-${index}`}>
              <div className="avatar">{message.role === "user" ? "YOU" : "RM"}</div>
              <div className="bubble">
                <div className="message-title">{message.title}</div>
                <p>{message.body}</p>
              </div>
            </article>
          ))}

          {takeOptions.length > 0 && (() => {
            const cards = takeOptions.map((take, index) => (
              <article
                className={`take-card ${generation?.id === take.id ? "selected" : ""} ${take.pending ? "pending" : ""}`}
                key={take.id}
                data-flow-id={take.id}
              >
                <div className="take-title">
                  <span>{take.type === "sound" || mode === "sounds" ? "Sound" : "Melody"} {take.takeNumber || index + 1}</span>
                </div>
                {audioSource(take) && (
                  <audio src={audioSource(take)} controls preload="metadata">
                    <track kind="captions" />
                  </audio>
                )}
                {!audioSource(take) && (
                  <div className="take-loading">
                    <span />
                    <p>{take.pending ? `Generating ${mode === "sounds" ? "sound" : "option"}...` : "Waiting for audio..."}</p>
                  </div>
                )}
                <div className="result-actions">
                  <button type="button" disabled={take.pending} onClick={() => setGeneration(take)}>Select</button>
                  <button type="button" disabled={take.pending} onClick={() => sendToReaper(take)}>Send</button>
                </div>
              </article>
            ));
            return (
              <section className="take-options">
                <div className="take-head">
                  <span>{mode === "sounds" ? "4 preset / one-shot options" : (takeOptions.length > 1 ? "6 melody/song options" : "Playback")}</span>
                </div>
                {takeOptions.length > 1 ? (
                  <MelodyFlow layoutKey={takeOptions.map((t) => t.id).join("-")}>
                    <div
                      className="take-grid"
                      style={{ "--take-cols": takeOptions.length }}
                    >
                      {cards}
                    </div>
                  </MelodyFlow>
                ) : (
                  <div
                    className="take-grid"
                    style={{ "--take-cols": takeOptions.length }}
                  >
                    {cards}
                  </div>
                )}
              </section>
            );
          })()}
        </div>

        <section className="composer">
          {timerText && <div className="composer-title"><span>{timerText}</span></div>}
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={mode === "sounds" ? "Describe the sound or preset..." : "Ask ROOM for a track..."}
          />
          <div className="composer-row">
            <button type="button" className="generate dual" disabled={busy} onClick={() => generateBatch("sounds")}>
              Generate Preset
            </button>
            <button type="button" className="generate dual" disabled={busy} onClick={() => generateBatch("songs")}>
              Generate Melody
            </button>
          </div>
        </section>
      </section>
      {error && <div className="error floating-error">{error}</div>}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
