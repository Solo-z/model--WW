const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("room", {
  generate: (payload) => ipcRenderer.invoke("room:generate", payload),
  revealGeneration: (generationId) => ipcRenderer.invoke("room:reveal", generationId),
  sendToReaper: (generationId) => ipcRenderer.invoke("room:send-to-reaper", generationId),
  getSettings: () => ipcRenderer.invoke("room:get-settings"),
  saveSettings: (settings) => ipcRenderer.invoke("room:save-settings", settings),
  onProgress: (callback) => {
    const handler = (_event, progress) => callback(progress);
    ipcRenderer.on("room:progress", handler);
    return () => ipcRenderer.removeListener("room:progress", handler);
  },
});
