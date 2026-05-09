const tracks = [
  { id: 1, title: "Direction 01", mood: "hardest drums" },
  { id: 2, title: "Direction 02", mood: "best melody" },
  { id: 3, title: "Direction 03", mood: "darkest" },
  { id: 4, title: "Direction 04", mood: "cleanest mix" },
  { id: 5, title: "Direction 05", mood: "most emotional" },
  { id: 6, title: "Direction 06", mood: "club ready" },
];

const dawTracks = [
  "ROOM Mix",
  "Drums",
  "Bass",
  "Vocals",
  "Other",
  "MIDI Bass",
  "MIDI Lead",
];

let selectedTrack = 1;

const $ = (id) => document.getElementById(id);

function show(id) {
  $(id).hidden = false;
  $(id).scrollIntoView({ behavior: "smooth", block: "start" });
}

function hide(id) {
  $(id).hidden = true;
}

function makeBars(count, seed) {
  let html = "";
  for (let i = 0; i < count; i += 1) {
    const h = 12 + ((i * 19 + seed * 7) % 62);
    html += `<span style="height:${h}px"></span>`;
  }
  return html;
}

function renderTracks() {
  $("track-grid").innerHTML = tracks.map((track) => `
    <article class="track-card ${track.id === selectedTrack ? "selected" : ""}" data-id="${track.id}">
      <div class="track-title">
        <span>${track.title}</span>
        <span>${track.mood}</span>
      </div>
      <div class="mini-wave">${makeBars(32, track.id)}</div>
      <audio controls preload="metadata"></audio>
      <button class="secondary pick">Select</button>
    </article>
  `).join("");

  document.querySelectorAll(".track-card").forEach((card) => {
    card.addEventListener("click", () => {
      selectedTrack = Number(card.dataset.id);
      renderTracks();
    });
  });
}

function renderAbleton() {
  $("plugin-direction").textContent = String(selectedTrack).padStart(2, "0");
  $("plugin-prompt").textContent = $("prompt").value.split(",").slice(0, 3).join(",");
  $("plugin-slots").innerHTML = tracks.map((track) => `
    <div class="plugin-slot ${track.id === selectedTrack ? "active" : ""}">
      ${String(track.id).padStart(2, "0")}
    </div>
  `).join("");

  $("ableton-tracks").innerHTML = dawTracks.map((name, idx) => `
    <div class="daw-track">
      <div class="track-label">${name}</div>
      <div class="clips">
        ${Array.from({ length: 6 }).map((_, i) => `
          <div class="clip ${name.startsWith("MIDI") ? "midi" : ""}" style="opacity:${0.35 + ((idx + i) % 4) * 0.16}"></div>
        `).join("")}
      </div>
    </div>
  `).join("");
}

function fakeGenerate() {
  hide("results");
  hide("ableton");
  hide("final");
  $("loader").hidden = false;
  $("step-pill").textContent = "Step 1 / 4 · Generating";

  let pct = 0;
  $("loader-fill").style.width = "0%";
  $("loader-percent").textContent = "0%";

  const timer = setInterval(() => {
    pct = Math.min(100, pct + (pct < 70 ? 7 : pct < 90 ? 4 : 2));
    $("loader-fill").style.width = `${pct}%`;
    $("loader-percent").textContent = `${pct}%`;
    if (pct >= 100) {
      clearInterval(timer);
      setTimeout(() => {
        $("loader").hidden = true;
        $("step-pill").textContent = "Step 1 / 4 · Prompt";
        renderTracks();
        show("results");
      }, 450);
    }
  }, 260);
}

$("generate").addEventListener("click", fakeGenerate);
$("download-all").addEventListener("click", () => {
  alert("Demo download: this would save full mix, stems, and MIDI for all six directions.");
});
$("send-ableton").addEventListener("click", () => {
  renderAbleton();
  show("ableton");
});
$("back-results").addEventListener("click", () => show("results"));
$("finish-demo").addEventListener("click", () => show("final"));
$("restart").addEventListener("click", () => {
  hide("results");
  hide("ableton");
  hide("final");
  $("room-view").scrollIntoView({ behavior: "smooth" });
});
$("play-session").addEventListener("click", (event) => {
  event.currentTarget.textContent = event.currentTarget.textContent === "▶" ? "Ⅱ" : "▶";
});
