const { app, BrowserWindow, ipcMain, shell } = require("electron");
const path = require("path");
const fs = require("fs");
const os = require("os");
const { Client } = require("@gradio/client");

const SPACE_URL = "https://solo363614-room-v2.hf.space";
const DEV_URL = "http://127.0.0.1:5173";

let mainWindow;
let gradioClientPromise;
const generationIndex = new Map();

function userDataPath(...parts) {
  return path.join(app.getPath("userData"), ...parts);
}

function generationsRoot() {
  return path.join(app.getPath("documents"), "ROOM", "Generations");
}

function settingsPath() {
  return userDataPath("settings.json");
}

function defaultReaperDir() {
  return process.env.REAPER_AGENT_DIR || path.join(os.homedir(), "AIAGENT DAW");
}

function safeName(name) {
  return String(name || "room_file").replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").slice(0, 180);
}

function emitProgress(stage, pct, detail = "") {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("room:progress", { stage, pct, detail });
  }
}

async function getClient() {
  if (!gradioClientPromise) {
    gradioClientPromise = Client.connect(SPACE_URL);
  }
  return gradioClientPromise;
}

async function downloadFile(file, outDir, fallbackName) {
  if (!file) return null;
  const url = file.url || file.path;
  if (!url) return null;
  const name = safeName(file.orig_name || path.basename(String(file.path || fallbackName)));
  const outPath = path.join(outDir, name || fallbackName);

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  }
  const bytes = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(outPath, bytes);
  return outPath;
}

function fileKind(filePath) {
  const base = path.basename(filePath).toLowerCase();
  if (base.endsWith(".mid") || base.endsWith(".midi")) return "midi";
  if (base.includes("drum")) return "drums";
  if (base.includes("bass")) return "bass";
  if (base.includes("vocal")) return "vocals";
  if (base.includes("other")) return "other";
  return "audio";
}

function readSettings() {
  try {
    return JSON.parse(fs.readFileSync(settingsPath(), "utf8"));
  } catch {
    return {
      reaperDir: defaultReaperDir(),
      autoOpenFolder: true,
    };
  }
}

function writeSettings(settings) {
  fs.mkdirSync(path.dirname(settingsPath()), { recursive: true });
  const next = {
    ...readSettings(),
    ...settings,
  };
  fs.writeFileSync(settingsPath(), JSON.stringify(next, null, 2));
  return next;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 900,
    minHeight: 650,
    title: "ROOM",
    backgroundColor: "#000000",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  if (!app.isPackaged) {
    mainWindow.loadURL(DEV_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("room:get-settings", () => readSettings());
ipcMain.handle("room:save-settings", (_event, settings) => writeSettings(settings));

ipcMain.handle("room:reveal", async (_event, generationId) => {
  const gen = generationIndex.get(generationId);
  if (!gen) return { ok: false, error: "Generation not found" };
  await shell.openPath(gen.dir);
  return { ok: true };
});

ipcMain.handle("room:send-to-reaper", async (_event, generationId) => {
  const gen = generationIndex.get(generationId);
  if (!gen) return { ok: false, error: "Generation not found" };

  const settings = readSettings();
  const reaperDir = settings.reaperDir || defaultReaperDir();
  fs.mkdirSync(reaperDir, { recursive: true });
  const commandFile = path.join(reaperDir, "reaper_commands.txt");

  const audioFiles = gen.files.filter((f) => !f.endsWith(".mid") && !f.endsWith(".midi"));
  const commands = [];
  const trackByKind = {
    audio: 0,
    drums: 1,
    bass: 2,
    vocals: 3,
    other: 4,
  };

  for (const f of audioFiles) {
    const kind = fileKind(f);
    const track = trackByKind[kind] ?? 0;
    commands.push(`INSERT_AUDIO ${track} "${f}" 0.0`);
  }

  if (commands.length === 0) {
    return { ok: false, error: "No WAV files to send to REAPER" };
  }

  fs.writeFileSync(commandFile, commands.join("\n") + "\n", "utf8");
  return { ok: true, commandFile, count: commands.length };
});

ipcMain.handle("room:install-reaper-script", async () => {
  const source = path.join(__dirname, "reaper_agent.lua");
  const targetDir = path.join(app.getPath("documents"), "ROOM", "Reaper");
  const target = path.join(targetDir, "reaper_agent.lua");

  fs.mkdirSync(targetDir, { recursive: true });
  fs.copyFileSync(source, target);
  await shell.openPath(targetDir);

  return {
    ok: true,
    path: target,
    instructions: "In REAPER: Actions -> Show action list -> New action -> Load ReaScript -> choose reaper_agent.lua -> Run.",
  };
});

ipcMain.handle("room:generate", async (_event, payload) => {
  const prompt = String(payload.prompt || "").trim();
  if (!prompt) throw new Error("Write a prompt first.");

  const outputs = [];
  if (payload.stems) outputs.push("Stems");
  if (payload.midi) outputs.push("MIDI");

  emitProgress("connecting", 5, "Connecting to ROOM");
  const client = await getClient();

  emitProgress("generating", 15, "Generating track");
  const result = await client.predict("/_generate_impl", {
    prompt,
    outputs_select: outputs,
    duration: Number(payload.duration || 30),
    seed: Number.isFinite(Number(payload.seed)) ? Number(payload.seed) : -1,
    steps: Number(payload.steps || 8),
    guidance: Number(payload.guidance || 7),
  });

  emitProgress("downloading", 82, "Saving files");
  const data = result.data || [];
  const audio = data[0] || null;
  const fileList = Array.isArray(data[1]) ? data[1] : [];
  const info = data[3] || "";

  const generationId = `room_${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const outDir = path.join(generationsRoot(), generationId);
  fs.mkdirSync(outDir, { recursive: true });

  const savedFiles = [];
  const audioPath = await downloadFile(audio, outDir, "room_mix.wav");
  if (audioPath) savedFiles.push(audioPath);

  for (let i = 0; i < fileList.length; i += 1) {
    const saved = await downloadFile(fileList[i], outDir, `room_file_${i}`);
    if (saved) savedFiles.push(saved);
  }

  const manifest = {
    id: generationId,
    prompt,
    createdAt: new Date().toISOString(),
    info,
    files: savedFiles,
  };
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2));
  generationIndex.set(generationId, { id: generationId, dir: outDir, files: savedFiles, info });

  emitProgress("ready", 100, "Ready");
  return {
    ok: true,
    id: generationId,
    dir: outDir,
    info,
    audioPath,
    files: savedFiles.map((filePath) => ({
      path: filePath,
      name: path.basename(filePath),
      kind: fileKind(filePath),
    })),
  };
});
