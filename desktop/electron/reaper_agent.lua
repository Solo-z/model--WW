-- Reaper AI Agent - Background Command Processor
-- Load once: Actions → Show action list → New action → Load ReaScript → select this file → Run
-- Keep running in background

-- Portable base dir using REAPER_AGENT_DIR or ~/AIAGENT DAW (match bridge)
local sep = package.config:sub(1,1)
local base = os.getenv("REAPER_AGENT_DIR")
if not base or base == "" then
  local home = os.getenv(sep == "\\" and "USERPROFILE" or "HOME") or "."
  base = home .. sep .. "AIAGENT DAW"
end
local COMMAND_FILE = base .. sep .. "reaper_commands.txt"
local STATE_FILE   = base .. sep .. "reaper_state.txt"
local FEEDBACK_FILE= base .. sep .. "reaper_feedback.txt"
local last_check = reaper.time_precise()
local last_state_export = reaper.time_precise()
local state_export_counter = 0

function msg(s) reaper.ShowConsoleMsg(tostring(s).."\n") end

-- If a sample library moved drives (common D:\ -> F:\), try a simple drive-letter fallback.
local function try_drive_fallback(path)
  if not path or path == "" then return path end
  if #path >= 3 and path:sub(2,3) == ":\\" then
    local drive = path:sub(1,1)
    if drive:upper() == "D" then
      local alt = "F" .. path:sub(2)
      local f = io.open(alt, "rb")
      if f then
        f:close()
        return alt
      end
    end
  end
  return path
end

-- Parameter conversion helpers
function db_to_normalized(target_db, min_db, max_db)
    min_db = min_db or -30
    max_db = max_db or 30
    return (target_db - min_db) / (max_db - min_db)
end

function normalized_to_db(normalized, min_db, max_db)
    min_db = min_db or -30
    max_db = max_db or 30
    return min_db + (normalized * (max_db - min_db))
end

local function clamp(value, minValue, maxValue)
    if value < minValue then return minValue end
    if value > maxValue then return maxValue end
    return value
end

-- Global function to ensure track exists (creates if needed)
function ensure_track_exists(index)
    local track = reaper.GetTrack(0, index)
    if track then return track end
    
    -- Insert new tracks until requested index is available
    while reaper.CountTracks(0) <= index do
        reaper.InsertTrackAtIndex(reaper.CountTracks(0), true)
    end
    reaper.TrackList_AdjustWindows(false)
    return reaper.GetTrack(0, index)
end

local note_base = {C=0, D=2, E=4, F=5, G=7, A=9, B=11}

local function parse_midi_pitch(token)
    if not token or token == "" then return nil end
    local numeric = tonumber(token)
    if numeric then
        return clamp(math.floor(numeric + 0.5), 0, 127)
    end

    local upper = token:upper():gsub("%s+", "")
    local note, accidental, octave = upper:match("^([A-G])([#B]?)(-?%d+)$")
    if not note then return nil end

    local semitone = note_base[note]
    if accidental == "#" then
        semitone = semitone + 1
    elseif accidental == "B" then
        semitone = semitone - 1
    end

    local oct = tonumber(octave) or 4
    local midi = (oct + 1) * 12 + semitone
    return clamp(midi, 0, 127)
end

local function ensure_midi_take(track, tStart, tEnd)
    -- 1. If a MIDI editor is open, use that take (ensure we write to what the user sees)
    local editor = reaper.MIDIEditor_GetActive()
    if editor then
        local take = reaper.MIDIEditor_GetTake(editor)
        if take and reaper.TakeIsMIDI(take) then
            -- Verify this take belongs to the requested track, OR just use it if track is generic
            local item = reaper.GetMediaItemTake_Item(take)
            local itemTrack = reaper.GetMediaItem_Track(item)
            if itemTrack == track then
                return take, item
            end
        end
    end

    -- 2. Otherwise find/create on the specific track
    if not track then return nil end

    local item = nil
    local count = reaper.CountTrackMediaItems(track)
    for i = 0, count - 1 do
        local it = reaper.GetTrackMediaItem(track, i)
        local pos = reaper.GetMediaItemInfo_Value(it, "D_POSITION")
        local len = reaper.GetMediaItemInfo_Value(it, "D_LENGTH")
        if tStart >= pos and tStart < (pos + len + 0.0001) then
            local tk = reaper.GetActiveTake(it)
            if tk and reaper.TakeIsMIDI(tk) then
                item = it
                break
            end
        end
    end

    local desiredEnd = math.max(tEnd or (tStart + 0.25), tStart + 0.25)
    if not item then
        item = reaper.CreateNewMIDIItemInProj(track, tStart, desiredEnd, false)
    else
        local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
        local len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
        local currentEnd = pos + len
        if desiredEnd > currentEnd then
            reaper.SetMediaItemLength(item, desiredEnd - pos, true)
        end
    end

    if not item then return nil end
    local take = reaper.GetActiveTake(item)
    if take and reaper.TakeIsMIDI(take) then
        return take, item
    end
    return nil
end

local chord_library = {}
local function register_chord(intervals, ...)
    for _, name in ipairs({...}) do
        chord_library[name:lower()] = intervals
    end
end

register_chord({0, 4, 7}, "maj", "major")
register_chord({0, 3, 7}, "min", "minor", "m")
register_chord({0, 3, 6}, "dim", "diminished")
register_chord({0, 4, 8}, "aug", "augmented")
register_chord({0, 2, 7}, "sus2")
register_chord({0, 5, 7}, "sus4")
register_chord({0, 4, 7, 10}, "7", "dom7", "dominant7")
register_chord({0, 4, 7, 11}, "maj7", "major7")
register_chord({0, 3, 7, 10}, "min7", "m7")
register_chord({0, 3, 6, 9}, "dim7")
register_chord({0, 3, 6, 10}, "m7b5", "halfdim", "half-dim")
register_chord({0, 7}, "power", "5")
register_chord({0, 4, 7, 14}, "add9")

local function resolve_chord_intervals(name)
    local key = (name or "maj"):lower():gsub("%s+", "")
    local intervals = chord_library[key]
    if intervals then
        return intervals, key
    end
    return chord_library["maj"], "maj"
end

-- Feedback buffer to report back to Python
local feedback_buffer = {}

function add_feedback(message)
    table.insert(feedback_buffer, message)
end

function write_feedback()
    if #feedback_buffer > 0 then
        local file = io.open(FEEDBACK_FILE, "w")
        if file then
            for _, msg in ipairs(feedback_buffer) do
                file:write(msg .. "\n")
            end
            file:close()
        end
        feedback_buffer = {}
    end
end

-- Generalized command execution wrapper
-- Automatically logs success/failure for ANY command
function execute_with_feedback(command_name, success, message)
    if success then
        local feedback_msg = string.format("✓ %s: %s", command_name, message or "success")
        msg(feedback_msg)
        add_feedback(feedback_msg)
        return true
    else
        local feedback_msg = string.format("✗ %s: %s", command_name, message or "failed")
        msg(feedback_msg)
        add_feedback(feedback_msg)
        return false
    end
end

function export_state()
    -- Export FULL Reaper project state to file (like Cursor reading entire codebase)
    local stateFile = io.open(STATE_FILE, "w")
    if not stateFile then return end
    
    state_export_counter = state_export_counter + 1
    local precise_time = reaper.time_precise()
    
    local numTracks = reaper.CountTracks(0)
    local playState = reaper.GetPlayState()
    local cursorPos = reaper.GetCursorPosition()
    local _, tempo = reaper.GetProjectTimeSignature2(0)
    
    -- Project-level info
    stateFile:write("=== PROJECT STATE ===\n")
    stateFile:write(string.format("Export Counter: %d\n", state_export_counter))
    stateFile:write(string.format("Precise Time: %.6f\n", precise_time))
    stateFile:write(string.format("Human Time: %s\n", os.date("%H:%M:%S")))
    stateFile:write(string.format("Unix Timestamp: %d\n", os.time()))
    stateFile:write(string.format("Playing: %s\n", ((playState & 1) ~= 0) and "Yes" or "No"))
    stateFile:write(string.format("Cursor Position: %.2fs\n", cursorPos))
    stateFile:write(string.format("Tempo: %.1f BPM\n", tempo))
    stateFile:write(string.format("Total Tracks: %d\n", numTracks))
    
    -- Time selection
    local timeStart, timeEnd = reaper.GetSet_LoopTimeRange(false, false, 0, 0, false)
    if timeStart ~= timeEnd then
        stateFile:write(string.format("Time Selection: %.2fs to %.2fs (%.2fs duration)\n", timeStart, timeEnd, timeEnd-timeStart))
    end
    
    -- Loop points
    local loopStart, loopEnd = reaper.GetSet_LoopTimeRange(false, true, 0, 0, false)
    if loopStart ~= loopEnd then
        stateFile:write(string.format("Loop: %.2fs to %.2fs\n", loopStart, loopEnd))
    end
    
    stateFile:write("\n=== TRACKS ===\n")
    
    -- Detailed track info
    for i = 0, numTracks - 1 do
        local track = reaper.GetTrack(0, i)
        local _, trackName = reaper.GetTrackName(track)
        local volume = reaper.GetMediaTrackInfo_Value(track, "D_VOL")
        local pan = reaper.GetMediaTrackInfo_Value(track, "D_PAN")
        local mute = reaper.GetMediaTrackInfo_Value(track, "B_MUTE")
        local solo = reaper.GetMediaTrackInfo_Value(track, "I_SOLO")
        local selected = reaper.IsTrackSelected(track)
        local numFX = reaper.TrackFX_GetCount(track)
        local numItems = reaper.CountTrackMediaItems(track)
        
        -- Use 0-based indexing to match Reaper API and command parameters
        stateFile:write(string.format("\n--- Track %d: %s ---\n", i, trackName))
        stateFile:write(string.format("  Volume: %.1f dB (%.0f%%)\n", 20*math.log(volume, 10), volume*100))
        stateFile:write(string.format("  Pan: %.0f%%\n", pan*100))
        stateFile:write(string.format("  Mute: %s | Solo: %s | Selected: %s\n", 
            mute == 1 and "YES" or "no", 
            solo > 0 and "YES" or "no",
            selected and "YES" or "no"))
        stateFile:write(string.format("  Media Items: %d\n", numItems))
        
        -- FX chain with parameters
        if numFX > 0 then
            stateFile:write(string.format("  FX Chain (%d plugins):\n", numFX))
            for j = 0, numFX - 1 do
                local _, fxName = reaper.TrackFX_GetFXName(track, j, "")
                local enabled = reaper.TrackFX_GetEnabled(track, j)
                stateFile:write(string.format("    [%d] %s %s\n", j, fxName, enabled and "" or "(BYPASSED)"))
                
                -- All parameters grouped by category
                local numParams = reaper.TrackFX_GetNumParams(track, j)
                if numParams > 0 then
                    local currentSection = ""
                    local maxShow = numParams  -- Show ALL params (no limit)
                    
                    for p = 0, maxShow - 1 do
                        local value = reaper.TrackFX_GetParam(track, j, p)
                        local _, paramName = reaper.TrackFX_GetParamName(track, j, p, "")
                        local _, displayValue = reaper.TrackFX_GetFormattedParamValue(track, j, p, "")
                        local nameLower = paramName:lower()
                        
                        -- Detect section changes
                        local newSection = ""
                        if nameLower:find("band %d+") or nameLower:find("eq") then
                            newSection = "EQ BANDS"
                        elseif nameLower:find("tap %d+") or nameLower:find("delay tap") then
                            newSection = "DELAY TAPS"
                        elseif nameLower:find("filter") then
                            newSection = "FILTERS"
                        elseif nameLower:find("lfo") or nameLower:find("modulation") then
                            newSection = "MODULATION"
                        elseif nameLower:find("dynamics") or nameLower:find("compress") or nameLower:find("threshold") then
                            newSection = "DYNAMICS"
                        elseif nameLower:find("output") or nameLower:find("mix") or nameLower:find("gain") and p > numParams - 10 then
                            newSection = "OUTPUT"
                        elseif p < 20 and currentSection == "" then
                            newSection = "MAIN CONTROLS"
                        end
                        
                        -- Print section header if changed
                        if newSection ~= "" and newSection ~= currentSection then
                            stateFile:write(string.format("\n        === %s ===\n", newSection))
                            currentSection = newSection
                        end
                        
                        stateFile:write(string.format("        p%d %s: %.0f%% [%s]\n", p, paramName, value*100, displayValue))
                    end
                    
                    -- All params shown, no truncation message needed
                end
                
                -- Check for FX parameter automation envelopes
                local hasAutomation = false
                for p = 0, numParams - 1 do
                    local env = reaper.GetFXEnvelope(track, j, p, false)
                    if env then
                        local numPoints = reaper.CountEnvelopePoints(env)
                        if numPoints > 0 then
                            if not hasAutomation then
                                stateFile:write("\n        === AUTOMATED PARAMETERS ===\n")
                                hasAutomation = true
                            end
                            local _, paramName = reaper.TrackFX_GetParamName(track, j, p, "")
                            stateFile:write(string.format("        p%d %s: %d automation points\n", p, paramName, numPoints))
                            -- Show first few points
                            local maxShow = math.min(numPoints, 3)
                            for pt = 0, maxShow - 1 do
                                local _, time, value = reaper.GetEnvelopePoint(env, pt)
                                stateFile:write(string.format("          %.2fs: %.0f%%\n", time, value*100))
                            end
                            if numPoints > 3 then
                                stateFile:write(string.format("          ... (%d more points)\n", numPoints - 3))
                            end
                        end
                    end
                end
            end
        else
            stateFile:write("  FX Chain: (empty)\n")
        end
        
        -- Volume envelope automation
        local volEnv = reaper.GetTrackEnvelopeByName(track, "Volume")
        if volEnv then
            local numPoints = reaper.CountEnvelopePoints(volEnv)
            if numPoints > 0 then
                stateFile:write(string.format("  Volume Automation: %d points\n", numPoints))
                -- Show first few points
                local maxShow = math.min(numPoints, 5)
                for p = 0, maxShow - 1 do
                    local _, time, value = reaper.GetEnvelopePoint(volEnv, p)
                    stateFile:write(string.format("    %.2fs: %.1fdB\n", time, 20*math.log(value, 10)))
                end
                if numPoints > 5 then
                    stateFile:write(string.format("    ... (%d more points)\n", numPoints - 5))
                end
            end
        end
    end
    
    stateFile:write("\n=== END STATE ===\n")
    stateFile:close()
end

function process_command(line)
    local parts = {}
    for word in line:gmatch("%S+") do
        table.insert(parts, word)
    end
    
    local cmd = parts[1]
    
    -- Special command to export state
    if cmd == "GET_STATE" then
        export_state()
        msg("📊 State exported")
        return
    end
    
    -- List all available plugins (Reaper stock only for demos)
    if cmd == "LIST_PLUGINS" then
        local plugins = {}
        local i = 0
        while true do
            local ret, name = reaper.EnumInstalledFX(i)
            if not ret then break end
            -- Filter to only Reaper/Cockos/JS plugins
            local nameLower = name:lower()
            if nameLower:find("cockos") or nameLower:find("reaper") or nameLower:find("reacomp") or 
               nameLower:find("reaeq") or nameLower:find("reagate") or nameLower:find("js:") then
                table.insert(plugins, name)
            end
            i = i + 1
        end
        
        local file = io.open(STATE_FILE, "w")
        if file then
            file:write("=== AVAILABLE PLUGINS (Reaper Stock) ===\n")
            file:write(string.format("Total: %d plugins\n\n", #plugins))
            for idx, name in ipairs(plugins) do
                file:write(string.format("[%d] %s\n", idx, name))
            end
            file:close()
        end
        
        msg(string.format("📋 Listed %d Reaper stock plugins to state file", #plugins))
        add_feedback(string.format("✓ Listed %d Reaper stock plugins", #plugins))
        return
    end
    
    -- Search for plugins by name (Reaper stock only for demos)
    if cmd == "SEARCH_PLUGIN" then
        local searchTerm = table.concat(parts, " ", 2):lower()
        local matches = {}
        local i = 0
        while true do
            local ret, name = reaper.EnumInstalledFX(i)
            if not ret then break end
            local nameLower = name:lower()
            -- Filter to only Reaper/Cockos/JS plugins
            if (nameLower:find("cockos") or nameLower:find("reaper") or nameLower:find("reacomp") or 
                nameLower:find("reaeq") or nameLower:find("reagate") or nameLower:find("js:")) and
               nameLower:find(searchTerm, 1, true) then
                table.insert(matches, name)
            end
            i = i + 1
        end
        
        local file = io.open(STATE_FILE, "w")
        if file then
            file:write(string.format("=== SEARCH RESULTS (Reaper Stock): '%s' ===\n", searchTerm))
            file:write(string.format("Found: %d matches\n\n", #matches))
            for idx, name in ipairs(matches) do
                file:write(string.format("[%d] %s\n", idx, name))
            end
            file:close()
        end
        
        msg(string.format("🔍 Found %d Reaper stock plugins matching '%s'", #matches, searchTerm))
        add_feedback(string.format("✓ Found %d Reaper stock plugins matching '%s'", #matches, searchTerm))
        return
    end
    
    -- Check if cmd is a numeric action ID
    local actionID = tonumber(cmd)
    if actionID then
        reaper.Main_OnCommand(actionID, 0)
        msg("✓ Executed action: " .. actionID)
        add_feedback("✓ Action " .. actionID .. " executed")
        return
    end
    
    -- Custom parametric commands below
    if cmd == "VOL_DIP" then
        -- VOL_DIP <trackIdx> <tStart> <tEnd> <value>
        local trackIdx = tonumber(parts[2]) or 0
        local tStart = tonumber(parts[3]) or 16
        local tEnd = tonumber(parts[4]) or 32
        local value = tonumber(parts[5]) or 0.5
        
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            execute_with_feedback("VOL_DIP", false, string.format("Track %d not found", trackIdx))
            return
        end
        
        local env = reaper.GetTrackEnvelopeByName(track, "Volume")
        if not env then
            -- Show volume envelope
            reaper.SetTrackSelected(track, true)
            reaper.Main_OnCommand(40406, 0) -- Track: Toggle volume envelope visible
            env = reaper.GetTrackEnvelopeByName(track, "Volume")
        end
        
        if env then
            local _, val_before = reaper.Envelope_Evaluate(env, tStart - 0.001, 0, 0)
            local _, val_after = reaper.Envelope_Evaluate(env, tEnd + 0.001, 0, 0)
            
            reaper.InsertEnvelopePoint(env, tStart-0.0005, val_before, 0, 0.0, true, false)
            reaper.InsertEnvelopePoint(env, tStart, value, 0, 0.0, true, false)
            reaper.InsertEnvelopePoint(env, tEnd, value, 0, 0.0, true, false)
            reaper.InsertEnvelopePoint(env, tEnd+0.0005, val_after, 0, 0.0, true, false)
            
            reaper.Envelope_SortPoints(env)
            execute_with_feedback("VOL_DIP", true, string.format("Track %d: %.1fs→%.1fs at %.0f%%", trackIdx, tStart, tEnd, value*100))
        else
            execute_with_feedback("VOL_DIP", false, string.format("Could not create volume envelope for track %d", trackIdx))
        end
        
    elseif cmd == "SET_TRACK_PAN" then
        -- SET_TRACK_PAN <trackIdx> <panValue>
        local trackIdx = tonumber(parts[2]) or 0
        local panValue = tonumber(parts[3]) or 0.0  -- -1.0 (left) to 1.0 (right)
        
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            -- Select track first (required for some operations)
            reaper.SetOnlyTrackSelected(track)
            reaper.SetMediaTrackInfo_Value(track, "D_PAN", panValue)
            execute_with_feedback("SET_TRACK_PAN", true, string.format("Track %d pan: %.0f%%", trackIdx, panValue*100))
        else
            execute_with_feedback("SET_TRACK_PAN", false, string.format("Track %d not found", trackIdx))
        end
        
    elseif cmd == "INSERT_INSTRUMENT" then
        -- INSERT_INSTRUMENT <trackIdx> <instrumentName>
        local trackIdx = tonumber(parts[2]) or 0
        if trackIdx < 0 then trackIdx = 0 end
        local instName = table.concat(parts, " ", 3)

        if instName == nil or instName == "" then
            execute_with_feedback("INSERT_INSTRUMENT", false, "No instrument name provided")
            return
        end

        local function ensure_track_exists(index)
            local track = reaper.GetTrack(0, index)
            if track then return track end

            local total = reaper.CountTracks(0)
            -- Insert new tracks until requested index is available
            while reaper.CountTracks(0) <= index do
                reaper.InsertTrackAtIndex(reaper.CountTracks(0), true)
            end
            reaper.TrackList_AdjustWindows(false)
            return reaper.GetTrack(0, index)
        end

        local track = ensure_track_exists(trackIdx)
        if not track then
            execute_with_feedback("INSERT_INSTRUMENT", false, string.format("Could not access track %d", trackIdx))
            return
        end

        reaper.SetOnlyTrackSelected(track)

        local function set_track_midi_input(track)
            local numInputs = reaper.GetNumMIDIInputs()
            local chosenIndex = nil
            local sourceName = nil

            for i = 0, numInputs - 1 do
                local ok, name = reaper.GetMIDIInputName(i, "")
                if ok and name and name:lower():find("virtual midi keyboard") then
                    chosenIndex = i
                    sourceName = name
                    break
                end
            end

            if not chosenIndex and numInputs > 0 then
                chosenIndex = 0
                local _, firstName = reaper.GetMIDIInputName(0, "")
                sourceName = firstName ~= "" and firstName or "MIDI input 1"
            end

            if chosenIndex then
                local channelAll = 0
                local inputValue = 4096 + (chosenIndex * 32) + channelAll
                reaper.SetMediaTrackInfo_Value(track, "I_RECINPUT", inputValue)
                return true, sourceName
            end

            return false, "No MIDI inputs detected"
        end

        -- Build candidate names (favor instrument variants)
        local function append_unique(tbl, value, seen)
            if value and value ~= "" and not seen[value] then
                table.insert(tbl, value)
                seen[value] = true
            end
        end

        local namesToTry = {}
        local seenNames = {}
        append_unique(namesToTry, instName, seenNames)

        local prefixes = {"VST3i: ", "VSTi: ", "VST3: ", "VST: ", "CLAPi: ", "CLAP: "}
        for _, prefix in ipairs(prefixes) do
            append_unique(namesToTry, prefix .. instName, seenNames)
        end

        local vendors = {
            "Arturia", "Native Instruments", "u-he", "Spectrasonics",
            "XLN Audio", "Spitfire Audio", "Tone2.com", "Initial Audio",
            "Ample Sound", "FabFilter", "Waves", "Cockos", "iZotope"
        }
        for _, vendor in ipairs(vendors) do
            for _, prefix in ipairs(prefixes) do
                append_unique(namesToTry, prefix .. instName .. " (" .. vendor .. ")", seenNames)
            end
        end

        -- If the name already includes a vendor prefix, strip it and retry
        local stripped = instName
        stripped = stripped:gsub("^%s*", "")
        stripped = stripped:gsub("^VST3[iI]?:%s*", "")
        stripped = stripped:gsub("^VST[iI]?:%s*", "")
        stripped = stripped:gsub("^CLAP[iI]?:%s*", "")
        if stripped ~= instName then
            append_unique(namesToTry, stripped, seenNames)
            for _, prefix in ipairs(prefixes) do
                append_unique(namesToTry, prefix .. stripped, seenNames)
            end
            for _, vendor in ipairs(vendors) do
                for _, prefix in ipairs(prefixes) do
                    append_unique(namesToTry, prefix .. stripped .. " (" .. vendor .. ")", seenNames)
                end
            end
        end

        local matchedName = nil
        local fxIdx = -1
        for _, tryName in ipairs(namesToTry) do
            fxIdx = reaper.TrackFX_AddByName(track, tryName, false, -1)
            if fxIdx >= 0 then
                matchedName = tryName
                break
            end
        end

        if fxIdx >= 0 then
            -- Move instrument to top slot so it behaves like a normal instrument chain
            if fxIdx > 0 then
                reaper.TrackFX_CopyToTrack(track, fxIdx, track, 0, true)
                fxIdx = 0
            end

            reaper.TrackFX_Show(track, fxIdx, 3)

            -- Arm and enable monitoring so the instrument is immediately playable
            reaper.SetMediaTrackInfo_Value(track, "I_RECARM", 1)
            reaper.SetMediaTrackInfo_Value(track, "I_RECMON", 1)
            local midiSet, midiSource = set_track_midi_input(track)

            local _, finalName = reaper.TrackFX_GetFXName(track, fxIdx, "")
            local successMsg = string.format("🎹 Inserted instrument '%s' on track %d (matched '%s')", finalName ~= "" and finalName or instName, trackIdx, matchedName or instName)
            if midiSet then
                successMsg = successMsg .. string.format(" | MIDI input: %s", midiSource or "All inputs")
            else
                successMsg = successMsg .. " | ⚠️ MIDI input not set (no MIDI devices found)"
            end
            msg(successMsg)
            add_feedback(successMsg)
        else
            -- Instrument not found - try fallback instruments based on type
            msg(string.format("⚠️ '%s' not found, searching for similar instrument...", instName))
            
            -- Determine instrument type from name and pick fallbacks
            local lowerName = string.lower(instName)
            local fallbacks = {}
            
            if string.find(lowerName, "drum") or string.find(lowerName, "kit") or string.find(lowerName, "perc") then
                -- Drum fallbacks - ONLY FREE Reaper plugins!
                fallbacks = {
                    "ReaSynDr",
                    "VSTi: ReaSynDr (Cockos)",
                    "VST3i: ReaSynDr (Cockos)",
                    "ReaSynDr (Cockos)",
                    "ReaSynth",
                    "VSTi: ReaSynth (Cockos)"
                }
            elseif string.find(lowerName, "bass") then
                -- Bass fallbacks - ONLY FREE Reaper plugins!
                fallbacks = {
                    "ReaSynth",
                    "VSTi: ReaSynth (Cockos)",
                    "VST3i: ReaSynth (Cockos)",
                    "ReaSynth (Cockos)"
                }
            elseif string.find(lowerName, "piano") or string.find(lowerName, "keys") or string.find(lowerName, "rhodes") or string.find(lowerName, "wurli") then
                -- Piano/Keys fallbacks - ONLY FREE Reaper plugins!
                fallbacks = {
                    "ReaSynth",
                    "VSTi: ReaSynth (Cockos)",
                    "VST3i: ReaSynth (Cockos)",
                    "ReaSynth (Cockos)"
                }
            elseif string.find(lowerName, "synth") or string.find(lowerName, "lead") or string.find(lowerName, "pad") then
                -- Synth fallbacks - ONLY FREE Reaper plugins!
                fallbacks = {
                    "ReaSynth",
                    "VSTi: ReaSynth (Cockos)",
                    "VST3i: ReaSynth (Cockos)",
                    "ReaSynth (Cockos)"
                }
            else
                -- Generic fallbacks - ONLY FREE Reaper plugins!
                fallbacks = {
                    "ReaSynth",
                    "VSTi: ReaSynth (Cockos)",
                    "VST3i: ReaSynth (Cockos)",
                    "ReaSynth (Cockos)"
                }
            end
            
            -- Try each fallback
            local fallbackIdx = -1
            local fallbackName = nil
            for _, fb in ipairs(fallbacks) do
                fallbackIdx = reaper.TrackFX_AddByName(track, fb, false, -1)
                if fallbackIdx >= 0 then
                    fallbackName = fb
                    break
                end
            end
            
            if fallbackIdx >= 0 then
                -- Move to top slot
                if fallbackIdx > 0 then
                    reaper.TrackFX_CopyToTrack(track, fallbackIdx, track, 0, true)
                    fallbackIdx = 0
                end
                
                reaper.TrackFX_Show(track, fallbackIdx, 3)
                reaper.SetMediaTrackInfo_Value(track, "I_RECARM", 1)
                reaper.SetMediaTrackInfo_Value(track, "I_RECMON", 1)
                local midiSet, midiSource = set_track_midi_input(track)
                
                local _, finalName = reaper.TrackFX_GetFXName(track, fallbackIdx, "")
                local successMsg = string.format("🎹 Inserted FALLBACK '%s' on track %d (requested '%s' not found)", finalName ~= "" and finalName or fallbackName, trackIdx, instName)
                if midiSet then
                    successMsg = successMsg .. string.format(" | MIDI input: %s", midiSource or "All inputs")
                end
                msg(successMsg)
                add_feedback(successMsg)
            else
                local failMsg = string.format("❌ Could not find instrument '%s' or any fallback", instName)
                msg(failMsg)
                add_feedback(failMsg)
            end
        end

    elseif cmd == "ADD_FX" then
        -- ADD_FX <trackIdx> <fxName>
        local trackIdx = tonumber(parts[2]) or 0
        local fxName = table.concat(parts, " ", 3)
       
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            -- First check if plugin already exists on track
            local numFX = reaper.TrackFX_GetCount(track)
            local existingIdx = -1
            for i = 0, numFX - 1 do
                local _, existingName = reaper.TrackFX_GetFXName(track, i, "")
                -- Check if name contains our search term (case insensitive fuzzy match)
                local searchLower = fxName:lower()
                local existingLower = existingName:lower()
                if existingLower:find(searchLower, 1, true) then
                    existingIdx = i
                    break
                end
            end
            
            if existingIdx >= 0 then
                -- Plugin already exists, just show it
                reaper.TrackFX_Show(track, existingIdx, 3)
                local _, existingName = reaper.TrackFX_GetFXName(track, existingIdx, "")
                local successMsg = string.format("🎛️ Opened existing FX '%s' on track %d", existingName, trackIdx)
                msg(successMsg)
                add_feedback(successMsg)
            else
                -- Plugin not found, try to add it
                -- Build list of variations to try
                local namesToTry = {fxName}  -- Exact match first
                
                -- Try with common vendor wrappers
                local vendors = {"FabFilter", "Waves", "Cockos", "iZotope"}
                for _, vendor in ipairs(vendors) do
                    table.insert(namesToTry, "VST3: " .. fxName .. " (" .. vendor .. ")")
                    table.insert(namesToTry, "VST: " .. fxName .. " (" .. vendor .. ")")
                end
                
                -- Try without wrappers
                table.insert(namesToTry, "VST3: " .. fxName)
                table.insert(namesToTry, "VST: " .. fxName)
                
                -- If name starts with vendor prefix, strip it and try again
                local strippedName = fxName:gsub("^FabFilter ", ""):gsub("^Waves ", ""):gsub("^iZotope ", "")
                if strippedName ~= fxName then
                    table.insert(namesToTry, strippedName)
                    for _, vendor in ipairs(vendors) do
                        table.insert(namesToTry, "VST3: " .. strippedName .. " (" .. vendor .. ")")
                        table.insert(namesToTry, "VST: " .. strippedName .. " (" .. vendor .. ")")
                    end
                end
                
                local matchedName = nil
                local fxIdx = -1
                for _, tryName in ipairs(namesToTry) do
                    fxIdx = reaper.TrackFX_AddByName(track, tryName, false, -1)
                    if fxIdx >= 0 then
                        matchedName = tryName
                        break
                    end
                end
                
                if fxIdx >= 0 then
                    reaper.TrackFX_Show(track, fxIdx, 3)
                    -- Wait for plugin to fully load
                    reaper.defer(function() end)
                    local successMsg = string.format("🎛️ Added FX '%s' (matched as '%s') to track %d", fxName, matchedName, trackIdx)
                    msg(successMsg)
                    add_feedback(successMsg)
                else
                    local failMsg = string.format("❌ Could not find FX: %s", fxName)
                    msg(failMsg)
                    add_feedback(failMsg)
                end
            end
        end
        
    elseif cmd == "SET_FX_PARAM" then
        -- SET_FX_PARAM <trackIdx> <fxIdx> <paramIdx> <value0-1>
        local trackIdx = tonumber(parts[2]) or 0
        local fxIdx = tonumber(parts[3]) or 0
        local paramIdx = tonumber(parts[4]) or 0
        local value = tonumber(parts[5]) or 0.5
        
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            local numFX = reaper.TrackFX_GetCount(track)
            if fxIdx < numFX then
                reaper.TrackFX_SetParam(track, fxIdx, paramIdx, value)
                local _, fxName = reaper.TrackFX_GetFXName(track, fxIdx, "")
                local _, paramName = reaper.TrackFX_GetParamName(track, fxIdx, paramIdx, "")
                local _, displayValue = reaper.TrackFX_GetFormattedParamValue(track, fxIdx, paramIdx, "")
                local successMsg = string.format("🎚️ Track %d FX#%d '%s' - %s → %.0f%% [%s]", trackIdx, fxIdx, fxName, paramName, value*100, displayValue)
                msg(successMsg)
                add_feedback(successMsg)
            else
                local failMsg = string.format("❌ Track %d only has %d FX (tried to access FX#%d)", trackIdx, numFX, fxIdx)
                msg(failMsg)
                add_feedback(failMsg)
            end
        end
        
    elseif cmd == "EQ_NEUTRALIZE_BAND" then
        -- EQ_NEUTRALIZE_BAND <trackIdx> <fxIdx> <bandNumber>
        local trackIdx = tonumber(parts[2]) or 0
        local fxIdx = tonumber(parts[3]) or 0
        local bandNum = tonumber(parts[4]) or 1
        
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            msg(string.format("❌ Track %d not found", trackIdx))
            add_feedback(string.format("❌ Track %d not found", trackIdx))
            return
        end
        
        local numFX = reaper.TrackFX_GetCount(track)
        if fxIdx >= numFX then
            msg(string.format("❌ Track %d only has %d FX (tried FX#%d)", trackIdx, numFX, fxIdx))
            add_feedback(string.format("❌ Track %d only has %d FX (tried FX#%d)", trackIdx, numFX, fxIdx))
            return
        end
        
        local _, fxName = reaper.TrackFX_GetFXName(track, fxIdx, "")
        local fxLower = (fxName or ""):lower()
        if not fxLower:find("pro-q") then
            msg(string.format("⚠️ EQ_NEUTRALIZE_BAND currently only supports Pro-Q (fx: %s)", fxName))
            add_feedback(string.format("⚠️ Skipped neutralizing band on %s (unsupported EQ)", fxName))
            return
        end
        
        local numParams = reaper.TrackFX_GetNumParams(track, fxIdx)
        local bandTag = string.format("band %d", bandNum)
        local bandTagCompact = string.format("band%d", bandNum)
        local usedIdx, gainIdx, shapeIdx, freqIdx, qIdx, slopeIdx = nil, nil, nil, nil, nil, nil
        
        for p = 0, numParams - 1 do
            local _, pname = reaper.TrackFX_GetParamName(track, fxIdx, p, "")
            local lname = (pname or ""):lower()
            if lname:find(bandTag) or lname:find(bandTagCompact) then
                if lname:find("used") or lname:find("enable") or lname:find("on/off") then
                    usedIdx = usedIdx or p
                elseif lname:find("gain") then
                    gainIdx = gainIdx or p
                elseif lname:find("shape") or lname:find("type") then
                    shapeIdx = shapeIdx or p
                elseif lname:find("freq") then
                    freqIdx = freqIdx or p
                elseif lname:find("slope") then
                    slopeIdx = slopeIdx or p
                elseif lname:find(" q") or lname:match("q$") or lname:find("resonance") then
                    qIdx = qIdx or p
                end
            end
        end
        
        local function set_param(idx, value)
            if idx ~= nil then
                reaper.TrackFX_SetParam(track, fxIdx, idx, value)
            end
        end
        
        -- Keep band enabled
        if usedIdx ~= nil then
            set_param(usedIdx, 1.0)
        end
        -- Neutral bell shape, 0dB gain
        if shapeIdx ~= nil then
            set_param(shapeIdx, 0.0)
        end
        if gainIdx ~= nil then
            set_param(gainIdx, 0.5)
        end
        if qIdx ~= nil then
            set_param(qIdx, 0.5)
        end
        if slopeIdx ~= nil then
            set_param(slopeIdx, 0.0)
        end
        if freqIdx ~= nil then
            set_param(freqIdx, 0.5)
        end
        
        msg(string.format("🧼 Neutralized Pro-Q band %d on track %d FX#%d", bandNum, trackIdx, fxIdx))
        add_feedback(string.format("✓ Neutralized Pro-Q band %d (track %d)", bandNum, trackIdx))
        
    elseif cmd == "REMOVE_FX" then
        -- REMOVE_FX <trackIdx> <fxIdx>
        local trackIdx = tonumber(parts[2]) or 0
        local fxIdx = tonumber(parts[3]) or 0
        
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            local numFX = reaper.TrackFX_GetCount(track)
            if fxIdx < numFX then
                local fxName = reaper.TrackFX_GetFXName(track, fxIdx, "")
                reaper.TrackFX_Delete(track, fxIdx)
                local successMsg = string.format("🗑️ Removed FX#%d '%s' from track %d", fxIdx, fxName, trackIdx)
                msg(successMsg)
                add_feedback(successMsg)
            else
                local failMsg = string.format("❌ Track %d only has %d FX (tried to remove FX#%d)", trackIdx, numFX, fxIdx)
                msg(failMsg)
                add_feedback(failMsg)
            end
        end
        
    elseif cmd == "SELECT_TRACK" then
        -- SELECT_TRACK <trackNumber>  (treat input as 1-based; REAPER API is 0-based)
        local trackNumber = tonumber(parts[2]) or 1
        local idx0 = math.max(trackNumber - 1, 0)
        
        -- Deselect all tracks first
        local numTracks = reaper.CountTracks(0)
        for i = 0, numTracks - 1 do
            local track = reaper.GetTrack(0, i)
            reaper.SetTrackSelected(track, false)
        end
        
        -- Select the target track
        local track = reaper.GetTrack(0, idx0)
        if track then
            reaper.SetTrackSelected(track, true)
            -- ALSO "touch" the track by setting volume to current volume (makes it "last touched")
            local currentVol = reaper.GetMediaTrackInfo_Value(track, "D_VOL")
            reaper.SetMediaTrackInfo_Value(track, "D_VOL", currentVol)
            execute_with_feedback("SELECT_TRACK", true, string.format("Selected track %d", trackNumber))
        else
            execute_with_feedback("SELECT_TRACK", false, string.format("Track %d not found", trackNumber))
        end
        
    elseif cmd == "SET_TRACK_VOL" then
        -- SET_TRACK_VOL <trackIdx> <volumeDB>
        local trackIdx = tonumber(parts[2]) or 0
        local volDB = tonumber(parts[3]) or 0
        
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            local volume = 10^(volDB/20)  -- Convert dB to linear
            reaper.SetMediaTrackInfo_Value(track, "D_VOL", volume)
            execute_with_feedback("SET_TRACK_VOL", true, string.format("Track %d → %.1fdB", trackIdx, volDB))
        else
            execute_with_feedback("SET_TRACK_VOL", false, string.format("Track %d not found", trackIdx))
        end
        
    elseif cmd == "CLEAR_AUTOMATION" then
        -- CLEAR_AUTOMATION <trackIdx> <envelopeName>
        local trackIdx = tonumber(parts[2]) or 0
        local envName = parts[3] or "Volume"
        
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            local env = reaper.GetTrackEnvelopeByName(track, envName)
            if env then
                -- Delete all automation points using DeleteEnvelopePointRange
                local numPoints = reaper.CountEnvelopePoints(env)
                if numPoints > 0 then
                    -- Delete entire range of points
                    reaper.DeleteEnvelopePointRange(env, 0, 999999)
                    local successMsg = string.format("🗑️ Cleared %d automation points from %s on track %d", numPoints, envName, trackIdx)
                    msg(successMsg)
                    add_feedback(successMsg)
                else
                    local infoMsg = string.format("ℹ️ No automation points on %s envelope of track %d", envName, trackIdx)
                    msg(infoMsg)
                    add_feedback(infoMsg)
                end
            else
                local failMsg = string.format("❌ No %s envelope on track %d", envName, trackIdx)
                msg(failMsg)
                add_feedback(failMsg)
            end
        else
            local failMsg = string.format("❌ Track %d not found", trackIdx)
            msg(failMsg)
            add_feedback(failMsg)
        end
    
    elseif cmd == "CLEAR_AUTOMATION_RANGE" then
        -- CLEAR_AUTOMATION_RANGE <trackIdx> <envelopeName> <tStart> <tEnd>
        local trackIdx = tonumber(parts[2]) or 0
        local envName = parts[3] or "Volume"
        local tStart = tonumber(parts[4]) or 0
        local tEnd = tonumber(parts[5]) or tStart

        local track = reaper.GetTrack(0, trackIdx)
        if track then
            local env = reaper.GetTrackEnvelopeByName(track, envName)
            if env then
                reaper.DeleteEnvelopePointRange(env, tStart, tEnd)
                local successMsg = string.format("🧹 Cleared automation range %.2fs–%.2fs on %s (track %d)", tStart, tEnd, envName, trackIdx)
                msg(successMsg)
                add_feedback(successMsg)
            else
                local failMsg = string.format("❌ No %s envelope on track %d", envName, trackIdx)
                msg(failMsg)
                add_feedback(failMsg)
            end
        else
            local failMsg = string.format("❌ Track %d not found", trackIdx)
            msg(failMsg)
            add_feedback(failMsg)
        end

    elseif cmd == "MIDI_CLEAR_SELECTION" then
        -- MIDI_CLEAR_SELECTION <trackIdx> [tStart] [tEnd]
        local trackIdx = tonumber(parts[2]) or 0
        local tStart = tonumber(parts[3]) or nil
        local tEnd = tonumber(parts[4]) or nil
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            msg(string.format("❌ Track %d not found", trackIdx))
            return
        end
        local itemCount = reaper.CountTrackMediaItems(track)
        for i = 0, itemCount - 1 do
            local item = reaper.GetTrackMediaItem(track, i)
            local take = reaper.GetMediaItemTake(item, 0)
            if take and reaper.TakeIsMIDI(take) then
                reaper.MIDI_SelectAll(take, false)
                if tStart and tEnd then
                    -- Unselect notes in time range to ensure clean slate
                    local noteCount = reaper.MIDI_CountNotes(take)
                    for n = noteCount - 1, 0, -1 do
                        local retval, selected, muted, startppqpos, endppqpos, chan, pitch, vel = reaper.MIDI_GetNote(take, n)
                        if selected then
                            local startTime = reaper.MIDI_GetProjTimeFromPPQPos(take, startppqpos)
                            if startTime >= tStart and startTime <= tEnd then
                                reaper.MIDI_SetNote(take, n, false, muted, startppqpos, endppqpos, chan, pitch, vel, true)
                            end
                        end
                    end
                end
            end
        end
        local msgText = tStart and string.format("🧹 Cleared MIDI selection on track %d (%.2f–%.2fs)", trackIdx, tStart, tEnd)
            or string.format("🧹 Cleared MIDI selection on track %d", trackIdx)
        msg(msgText)
        add_feedback(msgText)

    elseif cmd == "MIDI_SELECT_PITCHES" then
        -- MIDI_SELECT_PITCHES <trackIdx> <pitchesCsv> [tStart] [tEnd]
        local trackIdx = tonumber(parts[2]) or 0
        local pitchList = {}
        for pitch in string.gmatch(parts[3] or "", "([^,]+)") do
            local n = tonumber(pitch)
            if n then table.insert(pitchList, n) end
        end
        if #pitchList == 0 then
            msg("❌ MIDI_SELECT_PITCHES: no pitches provided")
            return
        end
        local tStart = tonumber(parts[4]) or nil
        local tEnd = tonumber(parts[5]) or nil
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            msg(string.format("❌ Track %d not found", trackIdx))
            return
        end
        local itemCount = reaper.CountTrackMediaItems(track)
        local selectedCount = 0
        for i = 0, itemCount - 1 do
            local item = reaper.GetTrackMediaItem(track, i)
            local take = reaper.GetMediaItemTake(item, 0)
            if take and reaper.TakeIsMIDI(take) then
                -- Clear existing selection in range
                reaper.MIDI_SelectAll(take, false)
                local noteCount = reaper.MIDI_CountNotes(take)
                for n = 0, noteCount - 1 do
                    local retval, selected, muted, startppq, endppq, chan, pitch, vel = reaper.MIDI_GetNote(take, n)
                    local startTime = reaper.MIDI_GetProjTimeFromPPQPos(take, startppq)
                    local inRange = true
                    if tStart and startTime < tStart - 1e-6 then inRange = false end
                    if tEnd and startTime > tEnd + 1e-6 then inRange = false end
                    if inRange then
                        for _, targetPitch in ipairs(pitchList) do
                            if pitch == targetPitch then
                                reaper.MIDI_SetNote(take, n, true, muted, startppq, endppq, chan, pitch, vel, false)
                                selectedCount = selectedCount + 1
                                break
                            end
                        end
                    end
                end
            end
        end
        local msgText = tStart and string.format("✅ Selected %d MIDI notes on track %d pitches %s (%.2f–%.2fs)", selectedCount, trackIdx, parts[3], tStart, tEnd)
            or string.format("✅ Selected %d MIDI notes on track %d pitches %s", selectedCount, trackIdx, parts[3])
        msg(msgText); add_feedback(msgText)

    elseif cmd == "MIDI_SET_VELOCITY_CONST" then
        -- MIDI_SET_VELOCITY_CONST <trackIdx> <velocity 1-127>
        local trackIdx = tonumber(parts[2]) or 0
        local velocity = math.max(1, math.min(127, tonumber(parts[3]) or 100))
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            msg(string.format("❌ Track %d not found", trackIdx)); return
        end
        local itemCount = reaper.CountTrackMediaItems(track)
        local changed = 0
        for i = 0, itemCount - 1 do
            local item = reaper.GetTrackMediaItem(track, i)
            local take = reaper.GetMediaItemTake(item, 0)
            if take and reaper.TakeIsMIDI(take) then
                local noteCount = reaper.MIDI_CountNotes(take)
                for n = 0, noteCount - 1 do
                    local retval, selected, muted, startppq, endppq, chan, pitch, vel = reaper.MIDI_GetNote(take, n)
                    if selected then
                        reaper.MIDI_SetNote(take, n, true, muted, startppq, endppq, chan, pitch, velocity, true)
                        changed = changed + 1
                    end
                end
            end
        end
        msg(string.format("🎹 Set velocity=%d on %d MIDI notes (track %d)", velocity, changed, trackIdx))
        add_feedback(string.format("✓ Set velocity %d on %d notes (track %d)", velocity, changed, trackIdx))

    elseif cmd == "MIDI_SET_VELOCITY_RAMP" then
        -- MIDI_SET_VELOCITY_RAMP <trackIdx> <startVel> <endVel>
        local trackIdx = tonumber(parts[2]) or 0
        local startVel = math.max(1, math.min(127, tonumber(parts[3]) or 80))
        local endVel = math.max(1, math.min(127, tonumber(parts[4]) or 100))
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            msg(string.format("❌ Track %d not found", trackIdx)); return
        end
        local itemCount = reaper.CountTrackMediaItems(track)
        for i = 0, itemCount - 1 do
            local item = reaper.GetTrackMediaItem(track, i)
            local take = reaper.GetMediaItemTake(item, 0)
            if take and reaper.TakeIsMIDI(take) then
                -- Collect selected notes with their start time
                local selectedNotes = {}
                local noteCount = reaper.MIDI_CountNotes(take)
                for n = 0, noteCount - 1 do
                    local retval, selected, muted, startppq, endppq, chan, pitch, vel = reaper.MIDI_GetNote(take, n)
                    if selected then
                        local startTime = reaper.MIDI_GetProjTimeFromPPQPos(take, startppq)
                        table.insert(selectedNotes, {index = n, startTime = startTime, info = {startppq, endppq, chan, pitch, vel}})
                    end
                end
                table.sort(selectedNotes, function(a, b) return a.startTime < b.startTime end)
                local count = #selectedNotes
                for idx, note in ipairs(selectedNotes) do
                    local frac = (count <= 1) and 0 or (idx - 1) / (count - 1)
                    local vel = math.floor(startVel + (endVel - startVel) * frac + 0.5)
                    local startppq, endppq, chan, pitch, _ = table.unpack(note.info)
                    reaper.MIDI_SetNote(take, note.index, true, false, startppq, endppq, chan, pitch, vel, true)
                end
            end
        end
        msg(string.format("🎛️ Velocity ramp %d→%d applied to selected MIDI notes on track %d", startVel, endVel, trackIdx))
        add_feedback(string.format("✓ Velocity ramp %d→%d applied to selected notes (track %d)", startVel, endVel, trackIdx))

    elseif cmd == "MIDI_CREATE_ITEM" then
        -- MIDI_CREATE_ITEM <trackIdx> <start> <end>
        local trackIdx = tonumber(parts[2]) or 0
        local tStart = tonumber(parts[3]) or 0
        local tEnd = tonumber(parts[4]) or (tStart + 4.0)
        
        -- Auto-create tracks if needed
        local track = ensure_track_exists(trackIdx)
        
        local item = reaper.CreateNewMIDIItemInProj(track, tStart, tEnd, false)
        if item then
            msg(string.format("🎹 Created MIDI item on track %d (%.2fs-%.2fs)", trackIdx, tStart, tEnd))
            add_feedback(string.format("✓ Created MIDI item on track %d", trackIdx))
        else
            msg("❌ Failed to create MIDI item")
            add_feedback("❌ Failed to create MIDI item")
        end

    elseif cmd == "MIDI_ACTION" then
        -- MIDI_ACTION <commandID>
        -- Executes a command ID in the active MIDI Editor context
        local actionID = tonumber(parts[2])
        if not actionID then
            msg("❌ MIDI_ACTION requires a numeric command ID")
            return
        end
        
        local editor = reaper.MIDIEditor_GetActive()
        if editor then
            reaper.MIDIEditor_OnCommand(editor, actionID)
            local successMsg = string.format("🎹 Executed MIDI Editor Action %d", actionID)
            msg(successMsg)
            add_feedback("✓ " .. successMsg)
        else
            msg("❌ No active MIDI Editor found. Use MIDI_EDITOR_OPEN first.")
            add_feedback("✗ No active MIDI Editor")
        end

    elseif cmd == "MIDI_EDITOR_OPEN" then
        -- MIDI_EDITOR_OPEN <trackIdx> <time>
        -- Opens the MIDI item at the specified time in the piano roll
        local trackIdx = tonumber(parts[2]) or 0
        local time = tonumber(parts[3]) or 0
        
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            msg(string.format("❌ Track %d not found", trackIdx))
            return
        end
        
        -- Find item at time
        local item = nil
        local count = reaper.CountTrackMediaItems(track)
        for i=0, count-1 do
            local it = reaper.GetTrackMediaItem(track, i)
            local pos = reaper.GetMediaItemInfo_Value(it, "D_POSITION")
            local len = reaper.GetMediaItemInfo_Value(it, "D_LENGTH")
            if time >= pos and time < (pos + len) then
                item = it
                break
            end
        end
        
        if item then
            reaper.Main_OnCommand(40153, 0) -- Main: Open selected item in MIDI editor (requires selection)
            -- Better way: Select it, then open
            reaper.SetMediaItemSelected(item, true)
            reaper.Main_OnCommand(40153, 0) -- Item: Open in built-in MIDI editor
            msg(string.format("🎹 Opened MIDI editor for item on track %d at %.1fs", trackIdx, time))
            add_feedback(string.format("✓ Opened MIDI editor (track %d)", trackIdx))
        else
            msg(string.format("❌ No MIDI item found on track %d at %.1fs", trackIdx, time))
            add_feedback("✗ No MIDI item found to open")
        end

    elseif cmd == "MIDI_INSERT_NOTE" then
        -- MIDI_INSERT_NOTE <trackIdx> <pitch> <vel> <start> <end> <channel>
        local trackIdx = tonumber(parts[2]) or 0
        local pitchToken = parts[3] or "60"
        local vel = clamp(math.floor(tonumber(parts[4]) or 100), 1, 127)
        local tStart = tonumber(parts[5]) or 0
        local tEnd = tonumber(parts[6]) or (tStart + 0.25)
        local chan = (tonumber(parts[7]) or 1) - 1

        -- Auto-create tracks if needed
        local track = ensure_track_exists(trackIdx)

        local pitch = parse_midi_pitch(pitchToken)
        if not pitch then
            msg(string.format("❌ Invalid MIDI note '%s'", tostring(pitchToken)))
            add_feedback(string.format("✗ Invalid MIDI note '%s'", tostring(pitchToken)))
            return
        end

        if not tEnd or tEnd <= tStart then
            tEnd = tStart + 0.25
        end
        chan = clamp(math.floor(chan + 0.5), 0, 15)

        local take = ensure_midi_take(track, tStart, tEnd)
        if not take then
            msg("❌ Could not access MIDI take")
            add_feedback("❌ Could not access MIDI take")
            return
        end

        local startppq = reaper.MIDI_GetPPQPosFromProjTime(take, tStart)
        local endppq = reaper.MIDI_GetPPQPosFromProjTime(take, tEnd)
        reaper.MIDI_InsertNote(take, false, false, startppq, endppq, chan, pitch, vel, true)
        reaper.MIDI_Sort(take)
        add_feedback(string.format("✓ Inserted note pitch=%d on track %d", pitch, trackIdx))

    elseif cmd == "MIDI_INSERT_CHORD" then
        -- MIDI_INSERT_CHORD <trackIdx> <root> <quality> <velocity> <start> <duration> [channel]
        local trackIdx = tonumber(parts[2]) or 0
        local rootToken = parts[3] or "C4"
        local qualityRaw = parts[4] or "maj"
        local vel = clamp(math.floor(tonumber(parts[5]) or 100), 1, 127)
        local tStart = tonumber(parts[6]) or 0
        local duration = tonumber(parts[7]) or 1.0
        local chan = (tonumber(parts[8]) or 1) - 1

        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            msg(string.format("❌ Track %d not found", trackIdx))
            add_feedback(string.format("✗ Track %d not found", trackIdx))
            return
        end

        local rootPitch = parse_midi_pitch(rootToken)
        if not rootPitch then
            msg(string.format("❌ Invalid chord root '%s'", tostring(rootToken)))
            add_feedback(string.format("✗ Invalid chord root '%s'", tostring(rootToken)))
            return
        end

        local intervals, normalizedQuality = resolve_chord_intervals(qualityRaw)
        if not duration or duration <= 0 then
            duration = 1.0
        end
        local tEnd = tStart + duration
        chan = clamp(math.floor(chan + 0.5), 0, 15)

        local take = ensure_midi_take(track, tStart, tEnd)
        if not take then
            msg("❌ Could not access MIDI take for chord insertion")
            add_feedback("❌ Could not access MIDI take for chord insertion")
            return
        end

        local startppq = reaper.MIDI_GetPPQPosFromProjTime(take, tStart)
        local endppq = reaper.MIDI_GetPPQPosFromProjTime(take, tEnd)
        local inserted = {}
        for _, interval in ipairs(intervals) do
            local note = clamp(rootPitch + interval, 0, 127)
            reaper.MIDI_InsertNote(take, false, false, startppq, endppq, chan, note, vel, true)
            table.insert(inserted, note)
        end
        reaper.MIDI_Sort(take)

        local detail = table.concat(inserted, ",")
        local feedbackMsg = string.format("🎼 Inserted %s chord (%s) on track %d", normalizedQuality or qualityRaw, detail, trackIdx)
        msg(feedbackMsg)
        add_feedback(feedbackMsg)

    elseif cmd == "MIDI_GET_NOTES" then
        -- MIDI_GET_NOTES <trackIdx> [start] [end]
        local trackIdx = tonumber(parts[2]) or 0
        local tStart = tonumber(parts[3])
        local tEnd = tonumber(parts[4])
        
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            local note_list = {}
            local count = reaper.CountTrackMediaItems(track)
            for i=0, count-1 do
                local item = reaper.GetTrackMediaItem(track, i)
                local take = reaper.GetActiveTake(item)
                if take and reaper.TakeIsMIDI(take) then
                    local _, noteCount = reaper.MIDI_CountNotes(take)
                    for n=0, noteCount-1 do
                        local _, sel, muted, startppq, endppq, chan, pitch, vel = reaper.MIDI_GetNote(take, n)
                        local startTime = reaper.MIDI_GetProjTimeFromPPQPos(take, startppq)
                        local endTime = reaper.MIDI_GetProjTimeFromPPQPos(take, endppq)
                        
                        if (not tStart or startTime >= tStart) and (not tEnd or startTime <= tEnd) then
                            table.insert(note_list, string.format("%d,%d,%.3f,%.3f", pitch, vel, startTime, endTime))
                        end
                    end
                end
            end
            local out = "MIDI_NOTES: " .. table.concat(note_list, ";")
            add_feedback(out)
            msg("🎼 Read " .. #note_list .. " notes from track " .. trackIdx)
        end

    elseif cmd == "MIDI_QUANTIZE" then
        -- MIDI_QUANTIZE <trackIdx> <grid> [strength]
        local trackIdx = tonumber(parts[2]) or 0
        local grid = tonumber(parts[3]) or 1.0 -- 1.0 = 1/4 note (QN)
        local strength = tonumber(parts[4]) or 1.0
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            msg(string.format("❌ Track %d not found", trackIdx)); return
        end
        
        local itemCount = reaper.CountTrackMediaItems(track)
        for i = 0, itemCount - 1 do
            local item = reaper.GetTrackMediaItem(track, i)
            local take = reaper.GetMediaItemTake(item, 0)
            if take and reaper.TakeIsMIDI(take) then
                local noteCount = reaper.MIDI_CountNotes(take)
                for n = 0, noteCount - 1 do
                    local retval, selected, muted, startppq, endppq, chan, pitch, vel = reaper.MIDI_GetNote(take, n)
                    if selected then
                        local qn_start = reaper.MIDI_GetProjQNFromPPQPos(take, startppq)
                        local qn_target = math.floor(qn_start / grid + 0.5) * grid
                        local diff = qn_target - qn_start
                        local new_qn = qn_start + (diff * strength)
                        local new_ppq = reaper.MIDI_GetPPQPosFromProjQN(take, new_qn)
                        local len_ppq = endppq - startppq
                        reaper.MIDI_SetNote(take, n, true, muted, new_ppq, new_ppq + len_ppq, chan, pitch, vel, true)
                    end
                end
                reaper.MIDI_Sort(take)
            end
        end
        msg(string.format("📏 Quantized selected MIDI notes on track %d to %.2f grid (%.0f%%)", trackIdx, grid, strength*100))
        add_feedback(string.format("✓ Quantized selected notes on track %d", trackIdx))

    elseif cmd == "MIDI_CORRECT_PITCH" then
        -- MIDI_CORRECT_PITCH <trackIdx> <scale> <key>
        local trackIdx = tonumber(parts[2]) or 0
        local scaleName = (parts[3] or "chromatic"):lower()
        local keyName = (parts[4] or "C"):upper()
        
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            msg(string.format("❌ Track %d not found", trackIdx)); return
        end
        
        local scales = {
            major = {0, 2, 4, 5, 7, 9, 11},
            minor = {0, 2, 3, 5, 7, 8, 10},
            chromatic = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
        }
        local scale = scales[scaleName] or scales.chromatic
        local note_map = {C=0, ["C#"]=1, Db=1, D=2, ["D#"]=3, Eb=3, E=4, F=5, ["F#"]=6, Gb=6, G=7, ["G#"]=8, Ab=8, A=9, ["A#"]=10, Bb=10, B=11}
        local root = note_map[keyName] or 0
        
        local function get_closest_scale_note(pitch, root, scale)
            local p = pitch % 12
            local min_dist = 100
            local best_p = p
            for _, offset in ipairs(scale) do
                local target = (root + offset) % 12
                local dist = math.abs(target - p)
                if dist > 6 then dist = 12 - dist end
                if dist < min_dist then min_dist = dist; best_p = target end
            end
            local octave = math.floor(pitch / 12)
            return octave * 12 + best_p
        end
        
        local itemCount = reaper.CountTrackMediaItems(track)
        local fixedCount = 0
        for i = 0, itemCount - 1 do
            local item = reaper.GetTrackMediaItem(track, i)
            local take = reaper.GetMediaItemTake(item, 0)
            if take and reaper.TakeIsMIDI(take) then
                local noteCount = reaper.MIDI_CountNotes(take)
                for n = 0, noteCount - 1 do
                    local retval, selected, muted, startppq, endppq, chan, pitch, vel = reaper.MIDI_GetNote(take, n)
                    if selected then
                        local new_pitch = get_closest_scale_note(pitch, root, scale)
                        if new_pitch ~= pitch then
                            reaper.MIDI_SetNote(take, n, true, muted, startppq, endppq, chan, new_pitch, vel, true)
                            fixedCount = fixedCount + 1
                        end
                    end
                end
            end
        end
        msg(string.format("🎤 Corrected %d notes to %s %s on track %d", fixedCount, keyName, scaleName, trackIdx))
        add_feedback(string.format("✓ Pitch corrected %d notes to %s %s", fixedCount, keyName, scaleName))

    elseif cmd == "CLEAR_FX_PARAM_AUTOMATION" then
        -- CLEAR_FX_PARAM_AUTOMATION <trackIdx> <fxIdx> <paramIdx> [tStart] [tEnd]
        local trackIdx = tonumber(parts[2]) or 0
        local fxIdx = tonumber(parts[3]) or 0
        local paramIdx = tonumber(parts[4]) or 0
        local tStart = tonumber(parts[5] or "0") or 0
        local tEnd = tonumber(parts[6] or "999999") or 999999

        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            local failMsg = string.format("❌ Track %d not found", trackIdx)
            msg(failMsg); add_feedback(failMsg); return
        end
        local env = reaper.GetFXEnvelope(track, fxIdx, paramIdx, false)
        if env then
            reaper.DeleteEnvelopePointRange(env, tStart, tEnd)
            local _, fxName = reaper.TrackFX_GetFXName(track, fxIdx, "")
            local _, paramName = reaper.TrackFX_GetParamName(track, fxIdx, paramIdx, "")
            local successMsg = string.format("🧹 Cleared automation on %s > %s (p%d) track %d in %.2f–%.2fs",
                fxName, paramName, paramIdx, trackIdx, tStart, tEnd)
            msg(successMsg); add_feedback(successMsg)
        else
            local infoMsg = string.format("ℹ️ No automation envelope for FX#%d p%d on track %d", fxIdx, paramIdx, trackIdx)
            msg(infoMsg); add_feedback(infoMsg)
        end
        
    elseif cmd == "FX_PARAM_AUTO" then
        -- FX_PARAM_AUTO <trackIdx> <fxIdx> <paramIdx> <tStart> <tEnd> <startValue> <endValue>
        local trackIdx = tonumber(parts[2]) or 0
        local fxIdx = tonumber(parts[3]) or 0
        local paramIdx = tonumber(parts[4]) or 0
        local tStart = tonumber(parts[5]) or 0
        local tEnd = tonumber(parts[6]) or 0
        local startValue = tonumber(parts[7]) or 0
        local endValue = tonumber(parts[8]) or 1
        
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            local failMsg = string.format("❌ Track %d not found", trackIdx)
            msg(failMsg)
            add_feedback(failMsg)
            return
        end
        
        local numFX = reaper.TrackFX_GetCount(track)
        if fxIdx >= numFX then
            local failMsg = string.format("❌ Track %d only has %d FX (tried to access FX#%d)", trackIdx, numFX, fxIdx)
            msg(failMsg)
            add_feedback(failMsg)
            return
        end
        
        -- Get FX and parameter info
        local _, fxName = reaper.TrackFX_GetFXName(track, fxIdx, "")
        local _, paramName = reaper.TrackFX_GetParamName(track, fxIdx, paramIdx, "")
        
        -- Get the parameter envelope (create if doesn't exist)
        local env = reaper.GetFXEnvelope(track, fxIdx, paramIdx, true)
        if not env then
            local failMsg = string.format("❌ Could not get envelope for param %d on FX#%d", paramIdx, fxIdx)
            msg(failMsg)
            add_feedback(failMsg)
            return
        end
        
        -- Get values before and after to preserve existing automation
        local _, val_before = reaper.Envelope_Evaluate(env, tStart - 0.001, 0, 0)
        local _, val_after = reaper.Envelope_Evaluate(env, tEnd + 0.001, 0, 0)
        
        -- Add automation points
        -- Add edge point before automation starts (preserve existing)
        reaper.InsertEnvelopePoint(env, tStart - 0.0005, val_before, 0, 0.0, true, false)
        -- Add start point
        reaper.InsertEnvelopePoint(env, tStart, startValue, 0, 0.0, true, false)
        -- Add end point
        reaper.InsertEnvelopePoint(env, tEnd, endValue, 0, 0.0, true, false)
        -- Add edge point after automation ends (preserve existing)
        reaper.InsertEnvelopePoint(env, tEnd + 0.0005, val_after, 0, 0.0, true, false)
        
        -- Sort points to ensure proper order
        reaper.Envelope_SortPoints(env)
        
        local successMsg = string.format("🎛️ Automated %s > %s: %.2fs→%.2fs (%.0f%%→%.0f%%)", 
            fxName, paramName, tStart, tEnd, startValue*100, endValue*100)
        msg(successMsg)
        add_feedback(successMsg)
        
    elseif cmd == "GOTO" then
        -- GOTO <seconds>
        local pos = tonumber(parts[2]) or 0
        reaper.SetEditCurPos(pos, true, true)
        execute_with_feedback("GOTO", true, string.format("Jump to %.1fs", pos))
    
    elseif cmd == "SET_TEMPO" then
        -- SET_TEMPO <bpm>
        local bpm = tonumber(parts[2]) or 120
        if bpm < 20 then bpm = 20 end
        if bpm > 300 then bpm = 300 end
        reaper.SetCurrentBPM(0, bpm, false)
        msg(string.format("🎵 Set tempo to %.0f BPM", bpm))
        add_feedback(string.format("✓ Tempo set to %.0f BPM", bpm))
    
    elseif cmd == "SET_TIME_SIGNATURE" then
        -- SET_TIME_SIGNATURE <numerator> <denominator> [bar]
        local num = tonumber(parts[2]) or 4
        local denom = tonumber(parts[3]) or 4
        local bar = tonumber(parts[4]) or 1
        local time_pos = reaper.TimeMap_GetMeasureInfo(0, bar - 1)
        reaper.SetTempoTimeSigMarker(0, -1, time_pos, -1, -1, num, denom, false)
        msg(string.format("🎼 Set time signature to %d/%d at bar %d", num, denom, bar))
        add_feedback(string.format("✓ Time signature: %d/%d", num, denom))
    
    elseif cmd == "SET_PROJECT_SAMPLERATE" then
        -- SET_PROJECT_SAMPLERATE <hz>
        local sr = tonumber(parts[2]) or 48000
        reaper.SetProjectSampleRate2(0, sr)
        msg(string.format("🎛️ Set sample rate to %d Hz", sr))
        add_feedback(string.format("✓ Sample rate: %d Hz", sr))
    
    elseif cmd == "SET_TRACK_NAME" then
        -- SET_TRACK_NAME <trackIdx> <name>
        local trackIdx = tonumber(parts[2]) or 0
        local name = table.concat(parts, " ", 3)
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            reaper.GetSetMediaTrackInfo_String(track, "P_NAME", name, true)
            msg(string.format("📝 Named track %d: %s", trackIdx, name))
            add_feedback(string.format("✓ Track %d named: %s", trackIdx, name))
        else
            msg(string.format("❌ Track %d not found", trackIdx))
            add_feedback(string.format("✗ Track %d not found", trackIdx))
        end
    
    elseif cmd == "SET_TRACK_COLOR" then
        -- SET_TRACK_COLOR <trackIdx> <r> <g> <b>
        local trackIdx = tonumber(parts[2]) or 0
        local r = tonumber(parts[3]) or 128
        local g = tonumber(parts[4]) or 128
        local b = tonumber(parts[5]) or 128
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            local color = reaper.ColorToNative(r, g, b) | 0x1000000
            reaper.SetTrackColor(track, color)
            msg(string.format("🎨 Set track %d color: RGB(%d,%d,%d)", trackIdx, r, g, b))
            add_feedback(string.format("✓ Track %d color set", trackIdx))
        else
            msg(string.format("❌ Track %d not found", trackIdx))
            add_feedback(string.format("✗ Track %d not found", trackIdx))
        end
    
    elseif cmd == "CREATE_SEND_BUS" then
        -- CREATE_SEND_BUS <name> <type>
        -- Creates a new track to use as a send bus (reverb, delay, etc)
        local name = parts[2] or "Bus"
        local bus_type = parts[3] or "reverb"
        
        -- Insert new track at end
        local track_count = reaper.CountTracks(0)
        reaper.InsertTrackAtIndex(track_count, true)
        local bus_track = reaper.GetTrack(0, track_count)
        
        if bus_track then
            -- Name it
            reaper.GetSetMediaTrackInfo_String(bus_track, "P_NAME", name, true)
            
            -- Color it (blue for reverb, green for delay, purple for other)
            local color
            if bus_type:lower():find("verb") then
                color = reaper.ColorToNative(50, 100, 200) | 0x1000000  -- Blue
            elseif bus_type:lower():find("delay") then
                color = reaper.ColorToNative(50, 200, 100) | 0x1000000  -- Green
            else
                color = reaper.ColorToNative(150, 50, 200) | 0x1000000  -- Purple
            end
            reaper.SetTrackColor(bus_track, color)
            
            msg(string.format("🎚️ Created send bus: %s (track %d)", name, track_count))
            add_feedback(string.format("✓ Bus created: %s", name))
        end
    
    elseif cmd == "SET_TRACK_SEND" then
        -- SET_TRACK_SEND <fromTrack> <toTrack> <sendLevel_db>
        local fromIdx = tonumber(parts[2]) or 0
        local toIdx = tonumber(parts[3]) or 0
        local level_db = tonumber(parts[4]) or -12
        
        local from_track = reaper.GetTrack(0, fromIdx)
        local to_track = reaper.GetTrack(0, toIdx)
        
        if from_track and to_track then
            -- Create send
            local send_idx = reaper.CreateTrackSend(from_track, to_track)
            if send_idx >= 0 then
                -- Set level
                local level_vol = 10 ^ (level_db / 20)
                reaper.SetTrackSendInfo_Value(from_track, 0, send_idx, "D_VOL", level_vol)
                msg(string.format("📤 Created send: Track %d → Track %d (%.1f dB)", fromIdx, toIdx, level_db))
                add_feedback(string.format("✓ Send: %d→%d @ %.1fdB", fromIdx, toIdx, level_db))
            else
                msg(string.format("❌ Failed to create send"))
                add_feedback(string.format("✗ Send creation failed"))
            end
        else
            msg(string.format("❌ Track not found (from:%d to:%d)", fromIdx, toIdx))
            add_feedback(string.format("✗ Track not found"))
        end
    
    elseif cmd == "SET_SIDECHAIN" then
        -- SET_SIDECHAIN <fromTrack> <toTrack> <fxName> <amount>
        -- Sets up sidechain compression (e.g., kick sidechaining bass)
        local fromIdx = tonumber(parts[2]) or 0
        local toIdx = tonumber(parts[3]) or 0
        local fxName = parts[4] or "Compressor"
        local amount = tonumber(parts[5]) or 0.5
        
        local from_track = reaper.GetTrack(0, fromIdx)
        local to_track = reaper.GetTrack(0, toIdx)
        
        if from_track and to_track then
            -- Find compressor on target track
            local fx_count = reaper.TrackFX_GetCount(to_track)
            local comp_idx = -1
            for i = 0, fx_count - 1 do
                local _, name = reaper.TrackFX_GetFXName(to_track, i, "")
                if name:lower():find("comp") then
                    comp_idx = i
                    break
                end
            end
            
            -- If no compressor, add one
            if comp_idx < 0 then
                comp_idx = reaper.TrackFX_AddByName(to_track, "ReaComp", false, -1)
            end
            
            if comp_idx >= 0 then
                -- Set sidechain input
                reaper.TrackFX_SetPinMappings(to_track, comp_idx, 1, 0, 0, 0)
                -- Create audio send from source to sidechain input
                local send_idx = reaper.CreateTrackSend(from_track, to_track)
                if send_idx >= 0 then
                    reaper.SetTrackSendInfo_Value(from_track, 0, send_idx, "I_SRCCHAN", -1)
                    msg(string.format("🔗 Sidechain: Track %d → Track %d (%.0f%%)", fromIdx, toIdx, amount*100))
                    add_feedback(string.format("✓ Sidechain: %d→%d", fromIdx, toIdx))
                end
            end
        else
            msg(string.format("❌ Track not found"))
            add_feedback(string.format("✗ Sidechain failed"))
        end
    
    elseif cmd == "SET_SWING" then
        -- SET_SWING <amount>
        -- Sets global swing/groove (0.0 = none, 1.0 = full)
        local amount = tonumber(parts[2]) or 0.0
        local swing_val = math.floor(amount * 127)
        -- Reaper's groove settings are accessed via action IDs
        msg(string.format("🎵 Swing set to %.1f%%", amount * 100))
        add_feedback(string.format("✓ Swing: %.1f%%", amount * 100))
        
    elseif cmd == "EXPORT_AUDIO" then
        -- EXPORT_AUDIO <trackIdx> <outputPath>
        local trackIdx = tonumber(parts[2]) or 0
        local outputPath = table.concat(parts, " ", 3)
        
        -- Strip quotes if present (bridge sends quoted paths)
        if outputPath:match('^"') then
            outputPath = outputPath:match('^"(.-)"') or outputPath:gsub('"', '')
        end
        
        local track = reaper.GetTrack(0, trackIdx)
        if track then
            -- Get project length
            local projEnd = reaper.GetProjectLength(0)
            
            -- Solo this track
            reaper.SetMediaTrackInfo_Value(track, "I_SOLO", 2)  -- Solo in place
            
            -- Render to file
            reaper.GetSetProjectInfo_String(0, "RENDER_FILE", outputPath, true)
            reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "", true)
            reaper.GetSetProjectInfo(0, "RENDER_SETTINGS", 0, true)  -- WAV format
            reaper.GetSetProjectInfo(0, "RENDER_SRATE", 44100, true)
            reaper.GetSetProjectInfo(0, "RENDER_CHANNELS", 2, true)  -- Stereo
            
            -- Render
            reaper.Main_OnCommand(41824, 0)  -- Render project using last settings
            
            -- Wait for render (simplified - may need improvement)
            local start_time = reaper.time_precise()
            while reaper.time_precise() - start_time < 0.5 do
                -- Brief wait
            end
            
            -- Unsolo track
            reaper.SetMediaTrackInfo_Value(track, "I_SOLO", 0)
            
            msg(string.format("🎵 Exported track %d to %s", trackIdx, outputPath))
            add_feedback(string.format("✓ Exported track %d to %s", trackIdx, outputPath))
        else
            msg(string.format("❌ Track %d not found", trackIdx))
            add_feedback(string.format("✗ Track %d not found", trackIdx))
        end
    
    elseif cmd == "INSERT_AUDIO" then
        -- INSERT_AUDIO <trackIdx> <filePath> <startTime>
        -- Places an audio sample file on a track at the specified time
        local trackIdx = tonumber(parts[2]) or 0
        
        -- Reconstruct file path (may contain spaces, handle quoted paths)
        local filePath = ""
        local startTime = 0
        
        -- Find the quoted path or unquoted path
        local fullLine = table.concat(parts, " ", 3)
        if fullLine:match('^"') then
            -- Quoted path: INSERT_AUDIO 2 "D:\Samples\kick.wav" 0.5
            local pathEnd = fullLine:find('"', 2)
            if pathEnd then
                filePath = fullLine:sub(2, pathEnd - 1)
                local remaining = fullLine:sub(pathEnd + 1):match("^%s*(.*)$") or ""
                startTime = tonumber(remaining) or 0
            end
        else
            -- Try to find last number as start time
            local lastSpace = fullLine:match(".*()%s+[%d%.]+$")
            if lastSpace then
                filePath = fullLine:sub(1, lastSpace - 1)
                startTime = tonumber(fullLine:sub(lastSpace + 1)) or 0
            else
                filePath = fullLine
                startTime = 0
            end
        end
        
        -- Check if file exists (with D:\ -> F:\ fallback)
        filePath = try_drive_fallback(filePath)
        local f = io.open(filePath, "rb")
        if not f then
            -- Try replacing backslashes with forward slashes just in case
            local altPath = filePath:gsub("\\", "/")
            f = io.open(altPath, "rb")
            if f then
                filePath = altPath
            else
                msg(string.format("❌ Audio file not found: %s", filePath))
                add_feedback(string.format("✗ File not found: %s", filePath))
                return
            end
        end
        f:close()
        
        -- Get or create track
        local track = reaper.GetTrack(0, trackIdx)
        if not track then
            -- Create tracks up to the needed number
            local current_count = reaper.CountTracks(0)
            for i = current_count, trackIdx do
                reaper.InsertTrackAtIndex(i, true)
            end
            track = reaper.GetTrack(0, trackIdx)
        end
        
        if track then
            -- Select only this track
            reaper.SetOnlyTrackSelected(track)
            
            -- Set cursor position
            reaper.SetEditCurPos(startTime, false, false)
            
            -- Remember item count before insert
            local itemCountBefore = reaper.CountMediaItems(0)
            
            -- Insert the audio file
            reaper.InsertMedia(filePath, 0)  -- 0 = insert at edit cursor
            
            -- Check if item was added
            local itemCountAfter = reaper.CountMediaItems(0)
            if itemCountAfter > itemCountBefore then
                -- Get the new item and make sure it's at correct position
                local newItem = reaper.GetMediaItem(0, itemCountAfter - 1)
                if newItem then
                    reaper.SetMediaItemPosition(newItem, startTime, false)
                    local itemLength = reaper.GetMediaItemInfo_Value(newItem, "D_LENGTH")
                    
                    -- Extract just filename for logging
                    local filename = filePath:match("[^/\\]+$") or filePath
                    msg(string.format("🥁 Inserted %s on track %d at %.2fs (%.2fs)", filename, trackIdx, startTime, itemLength))
                    add_feedback(string.format("✓ Inserted audio: %s at %.2fs", filename, startTime))
                end
            else
                msg(string.format("❌ Failed to insert audio: %s", filePath))
                add_feedback(string.format("✗ Insert failed: %s", filePath))
            end
        else
            msg(string.format("❌ Could not create track %d", trackIdx))
            add_feedback(string.format("✗ Could not create track %d", trackIdx))
        end
        
        reaper.UpdateArrange()
    
    elseif cmd == "LOAD_SAMPLER" then
        -- LOAD_SAMPLER <trackIdx> <filePath> <baseNote> [noteRangeStart] [noteRangeEnd]
        -- Loads a sample into ReaSamplOmatic5000 for MIDI triggering
        -- baseNote: the MIDI note that plays sample at original pitch (e.g. 60 for C4)
        -- noteRange: allows pitching (e.g. 36-84 for full range, or just 60-60 for single note)
        
        local trackIdx = tonumber(parts[2]) or 0
        local baseNote = tonumber(parts[4]) or 60  -- Default C4
        local noteStart = tonumber(parts[5]) or 24  -- Default C1
        local noteEnd = tonumber(parts[6]) or 96    -- Default C7
        
        -- Parse file path (may be quoted)
        local fullLine = table.concat(parts, " ", 3)
        local filePath = ""
        if fullLine:match('^"') then
            local pathEnd = fullLine:find('"', 2)
            if pathEnd then
                filePath = fullLine:sub(2, pathEnd - 1)
                -- Re-parse remaining args after path
                local remaining = fullLine:sub(pathEnd + 2)
                local rparts = {}
                for p in remaining:gmatch("%S+") do table.insert(rparts, p) end
                baseNote = tonumber(rparts[1]) or baseNote
                noteStart = tonumber(rparts[2]) or noteStart
                noteEnd = tonumber(rparts[3]) or noteEnd
            end
        else
            filePath = parts[3] or ""
        end
        
        -- Check file exists (with D:\ -> F:\ fallback)
        filePath = try_drive_fallback(filePath)
        local f = io.open(filePath, "rb")
        if not f then
            local altPath = filePath:gsub("\\", "/")
            f = io.open(altPath, "rb")
            if f then filePath = altPath end
        end
        if not f then
            msg(string.format("❌ Sample file not found: %s", filePath))
            add_feedback(string.format("✗ File not found: %s", filePath))
        else
            f:close()
            
            -- Get or create track
            local track = ensure_track_exists(trackIdx)
            if track then
                reaper.SetOnlyTrackSelected(track)
                
                -- Insert ReaSamplOmatic5000
                local fxIdx = reaper.TrackFX_AddByName(track, "ReaSamplOmatic5000", false, -1)
                if fxIdx >= 0 then
                    -- Set the sample file using named config parm
                    reaper.TrackFX_SetNamedConfigParm(track, fxIdx, "FILE0", filePath)
                    reaper.TrackFX_SetNamedConfigParm(track, fxIdx, "DONE", "")
                    
                    -- Set note range parameters (normalized 0-1 for 0-127)
                    -- Param indices for RS5K: 3=Note range start, 4=Note range end, 5=Pitch for start note
                    reaper.TrackFX_SetParam(track, fxIdx, 3, noteStart/127)  -- Note range start
                    reaper.TrackFX_SetParam(track, fxIdx, 4, noteEnd/127)    -- Note range end  
                    reaper.TrackFX_SetParam(track, fxIdx, 5, baseNote/127)   -- Pitch for start note
                    
                    -- Arm track for MIDI input
                    reaper.SetMediaTrackInfo_Value(track, "I_RECARM", 1)
                    reaper.SetMediaTrackInfo_Value(track, "I_RECMON", 1)
                    reaper.SetMediaTrackInfo_Value(track, "I_RECINPUT", 4096+0) -- All MIDI
                    
                    local filename = filePath:match("[^/\\]+$") or filePath
                    
                    -- RENAME TRACK to match sample (like dragging it in!)
                    reaper.GetSetMediaTrackInfo_String(track, "P_NAME", filename, true)
                    
                    msg(string.format("🎹 Loaded %s into sampler on track %d (notes %d-%d, base=%d)", 
                        filename, trackIdx, noteStart, noteEnd, baseNote))
                    add_feedback(string.format("✓ Sampler loaded: %s", filename))
                else
                    msg("❌ Failed to add ReaSamplOmatic5000")
                    add_feedback("✗ Failed to add sampler")
                end
            end
        end
        
        reaper.UpdateArrange()
    
    elseif cmd == "DRUM_SAMPLE" then
        -- Legacy - bridge should convert to INSERT_AUDIO
        local trackIdx = parts[2] or "?"
        local category = parts[3] or "?"
        local startTime = parts[4] or "?"
        msg(string.format("⚠️ DRUM_SAMPLE not resolved: track %s, %s at %ss", trackIdx, category, startTime))
        add_feedback(string.format("⚠️ DRUM_SAMPLE needs bridge: %s", category))

    elseif cmd == "ELEVEN_VOCALS" then
        -- Bridge-only command (Python must fetch MP3 + rewrite to INSERT_AUDIO).
        msg("⚠️ ELEVEN_VOCALS reached Lua. Start/restart your Python bridge so it rewrites ELEVEN_VOCALS -> INSERT_AUDIO.")
        add_feedback("⚠️ ELEVEN_VOCALS needs bridge rewrite (cloud_bridge.py/local_bridge.pyw).")

    elseif cmd == "EL1_SONG" then
        -- Bridge-only command (Python must fetch full song + stems + rewrite to INSERT_AUDIO commands).
        msg("⚠️ EL1_SONG reached Lua. Start/restart your Python bridge so it rewrites EL1_SONG -> INSERT_AUDIO commands.")
        add_feedback("⚠️ EL1_SONG needs bridge rewrite (cloud_bridge.py/local_bridge.pyw).")
        
    else
        msg("❓ Unknown command: "..tostring(cmd))
    end
end

function check_for_commands()
    local now = reaper.time_precise()
    if now - last_check < 0.1 then return end -- Check every 100ms
    last_check = now
    
    local file = io.open(COMMAND_FILE, "r")
    if not file then return end
    
    local content = file:read("*all")
    file:close()
    
    -- Delete file immediately
    os.remove(COMMAND_FILE)
    
    -- Process each line
    for line in content:gmatch("[^\r\n]+") do
        if line:match("%S") then
            process_command(line)
        end
    end
    
    -- Write feedback after all commands processed
    write_feedback()
    -- Export state so the bridge immediately sees changes
    export_state()
end

function loop()
    check_for_commands()
    -- Don't auto-export state (only on explicit GET_STATE to avoid spam)
    -- Bridge will request GET_STATE when needed
    reaper.defer(loop)
end

msg("🤖 Reaper AI Agent Started")
msg("Watching: "..COMMAND_FILE)
loop()

