# ROOM REAPER Connector

ROOM Desktop can send generated WAV files into REAPER using a lightweight
ReaScript bridge.

## Install

1. Open ROOM Desktop.
2. Open `Advanced`.
3. Click `Install REAPER Script`.
4. ROOM opens a folder containing `reaper_agent.lua`.
5. In REAPER, go to `Actions -> Show action list`.
6. Click `New action -> Load ReaScript`.
7. Select `reaper_agent.lua`.
8. Click `Run` and leave the script running.

## Use

1. Generate a track in ROOM Desktop.
2. Click `Send to REAPER`.
3. ROOM writes import commands to:

```text
%USERPROFILE%\AIAGENT DAW\reaper_commands.txt
```

4. The ReaScript reads those commands and imports the WAV files into REAPER.

## Track layout

ROOM currently sends:

- Full mix / unknown audio -> track 1
- Drums -> track 2
- Bass -> track 3
- Vocals -> track 4
- Other -> track 5

MIDI files are saved in the generation folder. Direct MIDI import can be added
after the audio import workflow is stable.

## Troubleshooting

- If nothing imports, make sure `reaper_agent.lua` is running in REAPER.
- Keep REAPER open while clicking `Send to REAPER`.
- Check the folder `%USERPROFILE%\AIAGENT DAW` for:
  - `reaper_commands.txt`
  - `reaper_feedback.txt`
  - `reaper_state.txt`
