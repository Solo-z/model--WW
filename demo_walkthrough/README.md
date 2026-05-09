# ROOM Product Walkthrough Mock

Local clickable walkthrough for demos while the signed desktop app is not ready.

It does not call HuggingFace or spend GPU. It simulates:

- Prompting ROOM
- Generating six directions
- Picking a favorite track
- Downloading generated files
- Sending the selected direction into an Ableton-style DAW view

## Run locally

From the repository root:

```bash
python -m http.server 8088
```

Then open:

```text
http://localhost:8088/demo_walkthrough/
```

Use this for investor/product walkthroughs without installing an unsigned app.
