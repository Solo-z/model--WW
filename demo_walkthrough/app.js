const tracks = [
  { id: 1, title: "Direction 01", mood: "hardest" },
  { id: 2, title: "Direction 02", mood: "melodic" },
  { id: 3, title: "Direction 03", mood: "dark" },
  { id: 4, title: "Direction 04", mood: "clean" },
  { id: 5, title: "Direction 05", mood: "emotional" },
  { id: 6, title: "Direction 06", mood: "club" },
];

const dawTracks = ["ROOM Mix", "Drums", "Bass", "Vocals", "Other", "MIDI Bass", "MIDI Lead"];

let selectedTrack = 1;
const $ = (id) => document.getElementById(id);

function makeBars(count, seed) {
  let html = "";
  for (let i = 0; i < count; i += 1) {
    const h = 8 + ((i * 17 + seed * 11) % 28);
    html += `<span style="height:${h}px"></span>`;
  }
  return html;
}

function renderArrangement(committed = false) {
  $("ableton-tracks").innerHTML = dawTracks.map((name, idx) => `
    <div class="daw-track">
      <div class="track-label">${name}</div>
      <div class="clips">
        ${Array.from({ length: 6 }).map((_, i) => {
          const active = committed && i < 5;
          return `<div class="clip ${name.startsWith("MIDI") ? "midi" : ""}" style="opacity:${active ? 0.9 : 0.18}"></div>`;
        }).join("")}
      </div>
    </div>
  `).join("");
}

function renderDirections() {
  $("track-grid").innerHTML = tracks.map((track) => `
    <button class="direction-card ${track.id === selectedTrack ? "selected" : ""}" data-id="${track.id}">
      <span>${String(track.id).padStart(2, "0")}</span>
      <span>${track.mood}</span>
      <div class="mini-wave">${makeBars(18, track.id)}</div>
    </button>
  `).join("");

  document.querySelectorAll(".direction-card").forEach((card) => {
    card.addEventListener("click", () => {
      selectedTrack = Number(card.dataset.id);
      $("plugin-direction").textContent = String(selectedTrack).padStart(2, "0");
      renderDirections();
    });
  });
}

function fakeGenerate() {
  $("loader").hidden = false;
  $("commit").disabled = true;
  $("track-grid").innerHTML = "";
  renderArrangement(false);

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
        $("commit").disabled = false;
        renderDirections();
      }, 350);
    }
  }, 240);
}

$("generate").addEventListener("click", fakeGenerate);
$("commit").addEventListener("click", () => renderArrangement(true));
$("download-all").addEventListener("click", () => {
  alert("Demo export: full mix, stems, and MIDI would be saved from the plugin.");
});
$("play-session").addEventListener("click", (event) => {
  event.currentTarget.textContent = event.currentTarget.textContent === "▶" ? "Ⅱ" : "▶";
});

renderArrangement(false);
