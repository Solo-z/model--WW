# ROOM Desktop

Windows desktop client for ROOM.

## What it does

- Calls the hosted ROOM Hugging Face Space.
- Saves generated audio, stems, and MIDI to:
  `Documents/ROOM/Generations/<generation-id>/`
- Plays the generated track in the app.
- Opens the output folder after generation.
- Optional: sends generated WAV files into REAPER via the existing
  `reaper_agent.lua` command-file bridge.

## Development

```bash
npm install
npm run dev
```

## Build Windows app

```bash
npm run dist
```

Outputs go to `desktop/release/`.

## REAPER integration

1. Open REAPER.
2. Go to `Actions -> Show action list`.
3. Click `New action -> Load ReaScript`.
4. Load `desktop/electron/reaper_agent.lua`.
5. Run the script and leave it running.
6. In ROOM Desktop, generate a track and click `Send to REAPER`.

The app writes commands to:

```text
%USERPROFILE%\AIAGENT DAW\reaper_commands.txt
```

The Lua script imports the audio files into tracks.
