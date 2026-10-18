"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  slide: null,
  catalog: null,
  preview: null,
  scale: 1,
  fitScale: 1,
  x: 0,
  y: 0,
  mode: "pan",
  drag: null,
  selection: null,
  region: null,
  heatmap: null,
  genes: [],
  tiles: new Map(),
  busy: false,
  offset: 0,
  libraryRows: [],
  version: 0,
  predictionInfo: null,
  libraryTissues: [],
  libraryHasMore: false,
  prototype: null,
  job: null,
  toast: null,
  uploading: false,
};
const canvas = $("slide-canvas"),
  ctx = canvas.getContext("2d");
const labels = { her2st: "HER2ST", cscc: "cSCC", dlpfc: "DLPFC" };
const t = (key, values) => I18n.t(key, values);
const number = (value, digits) => I18n.number(value, digits);
const tissueName = (value) => I18n.tissueName(value);
class LocalizedError extends Error {
  constructor(key, params = {}) {
    super(t(key, params));
    this.key = key;
    this.params = params;
  }
}
function messageValues(params = {}) {
  return Object.fromEntries(Object.entries(params).map(([key, value]) => [key,
    typeof value === "number" ? number(value) : value]));
}
function jobMessage(result) {
  return t(result.message_key || "error.serverFailure", messageValues(result.message_params));
}
function element(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}
async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok) throw new LocalizedError(data.error_key || "error.requestFailed", data.error_params);
  return data;
}
function url(path, args) {
  return path + "?" + new URLSearchParams(args);
}
let toastTimer;
function renderToast() {
  if (state.toast) $("toast").textContent = t(state.toast.key, messageValues(state.toast.params));
}
function toast(key, params = {}, error = false) {
  state.toast = { key, params };
  renderToast();
  $("toast").className = "toast" + (error ? " error" : "");
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("toast").hidden = true), 6500);
}
function guarded(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (error) {
      toast(error.key || "error.requestFailed", error.params || {}, true);
    }
  };
}
function empty(target, message) {
  target.replaceChildren();
  const box = element("div", "empty-state");
  box.append(element("span", "", "◈"), element("p", "", message));
  target.append(box);
}
function setMode(mode) {
  state.mode = mode;
  canvas.className = mode;
  for (const name of ["pan", "region", "spot"]) {
    $("tool-" + name).classList.toggle("active", name === mode);
    $("tool-" + name).setAttribute("aria-pressed", String(name === mode));
  }
}
function controls() {
  const predicted = !!state.genes.length;
  $("predict").disabled = !state.slide?.model_ready || state.busy;
  $("predict").textContent = state.busy
    ? t("processing")
    : predicted
      ? t("predictAgain")
      : t("predict");
  for (const id of [
    "whole-slide",
    "index-open",
    "download",
    "heatmap-toggle",
    "gene-input",
  ])
    $(id).disabled = !predicted || state.busy;
}
function refreshTissues(libraries) {
  const names = [
    ...new Set([
      ...(state.catalog?.models || []).map((m) => m.tissue),
      ...libraries.map((l) => l.tissue),
    ]),
  ];
  for (const id of ["search-tissue", "library-tissue"]) {
    const select = $(id),
      old = select.value;
    select.replaceChildren();
    if (id === "library-tissue") select.append(new Option(t("allTissues"), ""));
    for (const name of names) select.append(new Option(tissueName(name), name));
    if (names.includes(old)) select.value = old;
  }
  $("tissue-options").replaceChildren(
    ...names.map((name) => new Option(tissueName(name), tissueName(name))),
  );
}
async function catalog() {
  state.catalog = await api("/api/catalog");
  renderCatalog();
}
function renderCatalog() {
  if (!state.catalog) return;
  const select = $("slide-select"),
    old = select.value;
  select.replaceChildren();
  for (const dataset of ["her2st", "cscc", "dlpfc"]) {
    const group = element("optgroup");
    group.label = labels[dataset];
    for (const slide of state.catalog.slides.filter(
      (s) => s.dataset === dataset,
    )) {
      group.append(
        new Option(
          slide.name +
            (slide.uploaded ? " · " + t("uploaded") : "") +
            (slide.predicted ? " · " + t("predicted") : ""),
          slide.id,
        ),
      );
    }
    select.append(group);
  }
  if (state.catalog.slides.some((s) => s.id === old)) select.value = old;
  refreshTissues(state.catalog.libraries);
  for (const option of $("upload-model").options) {
    option.disabled = !state.catalog.models.find(
      (m) => m.dataset === option.value,
    )?.ready;
  }
  $("connection").textContent = t("connected");
}
function renderSelection() {
  if (state.region) {
    const result = state.region, selection = state.selection;
    $("selection-count").textContent = t("spotCount", { count: number(result.spots) });
    $("selection-title").textContent = t(selection.spot !== undefined ? "singleExpression" :
      selection.bounds ? "selectedExpression" : "wholeExpression");
    $("selection-help").textContent = t("regionSummary", {
      genes: number(result.ranking.length), spots: number(result.spots),
    });
    $("ranking").replaceChildren(rankingRows(result.ranking));
    renderRetrieval(result.prototypes);
  } else {
    $("selection-count").textContent = t("noSelection");
    $("selection-title").textContent = t("regionExpression");
    $("selection-help").textContent = state.slide && !state.slide.model_ready ? t("error.modelNotReady") :
      state.predictionInfo ? t("predictionReady", {
        spots: number(state.predictionInfo.spots), genes: number(state.predictionInfo.genes.length),
      }) : t("selectionHelp");
    empty($("ranking"), t("rankingEmpty"));
    empty($("retrieval"), t("retrievalEmpty"));
  }
}
function clearSelection() {
  state.selection = null;
  state.region = null;
  renderSelection();
  draw();
}
function renderSlideDetails() {
  const slide = state.slide;
  if (!slide) return;
  $("tissue-badge").textContent = tissueName(slide.tissue);
  $("slide-details").textContent = t("slideDetails", {
    name: slide.name, width: number(slide.width), height: number(slide.height), spots: number(slide.spots),
  });
  $("canvas-status").textContent = labels[slide.dataset] + " / " + slide.name;
  $("coordinate-note").hidden = !slide.coordinate_note;
  $("coordinate-note").textContent = slide.coordinate_note ? t(slide.coordinate_note_key || "coordinateWarning") : "";
}
async function openSlide(id) {
  const version = ++state.version;
  state.slide = null;
  state.predictionInfo = null;
  state.genes = [];
  state.preview = null;
  state.heatmap = null;
  state.tiles.clear();
  state.busy = false;
  state.drag = null;
  clearSelection();
  controls();
  $("viewer-empty").hidden = false;
  $("viewer-empty").textContent = t("openingSlide");
  $("job-status").hidden = true;
  $("heatmap-toggle").checked = false;
  $("gene-input").value = "";
  $("gene-options").replaceChildren();
  const slide = await api(url("/api/slide", { id }));
  if (version !== state.version) return;
  state.slide = slide;
  $("slide-select").value = id;
  $("search-tissue").value = slide.tissue;
  $("index-tissue").value = tissueName(slide.tissue);
  renderSlideDetails();
  renderSelection();
  const preview = new Image();
  preview.src = url("/api/preview", { id });
  await preview.decode();
  if (version !== state.version) return;
  state.preview = preview;
  $("viewer-empty").hidden = true;
  fit();
  controls();
  if (slide.predicted) await loadPrediction(version);
}
async function loadPrediction(version = state.version) {
  const result = await api(url("/api/prediction", { id: state.slide.id }));
  if (version !== state.version) return;
  state.genes = result.genes;
  $("gene-options").replaceChildren(
    ...result.genes.map((g) => new Option(g, g)),
  );
  state.predictionInfo = result;
  renderSelection();
  controls();
  draw();
}
function dimensions() {
  const rect = canvas.getBoundingClientRect();
  return { width: rect.width, height: rect.height };
}
function resize() {
  const { width, height } = dimensions(),
    ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}
function fit() {
  if (!state.slide) return;
  const { width, height } = dimensions();
  state.scale = Math.min(
    (width - 40) / state.slide.width,
    (height - 40) / state.slide.height,
  );
  state.fitScale = state.scale;
  state.x = (width - state.slide.width * state.scale) / 2;
  state.y = (height - state.slide.height * state.scale) / 2;
  draw();
}
function position(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}
function world(point) {
  return {
    x: (point.x - state.x) / state.scale,
    y: (point.y - state.y) / state.scale,
  };
}
function zoom(factor, point) {
  if (!state.slide) return;
  const before = world(point),
    next = Math.max(state.fitScale * 0.35, Math.min(8, state.scale * factor));
  state.scale = next;
  state.x = point.x - before.x * next;
  state.y = point.y - before.y * next;
  draw();
}
function heatColor(t) {
  const stops = [
      [92, 90, 206],
      [48, 199, 187],
      [255, 217, 118],
    ],
    i = t < 0.5 ? 0 : 1,
    u = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  return stops[i]
    .map((v, k) => Math.round(v + (stops[i + 1][k] - v) * u))
    .join(",");
}
let tileRedraw = false;
function redrawSoon() {
  if (tileRedraw) return;
  tileRedraw = true;
  requestAnimationFrame(() => {
    tileRedraw = false;
    draw();
  });
}
function drawTiles(width, height) {
  const slide = state.slide;
  if (
    state.scale <=
    Math.min(
      state.preview.width / slide.width,
      state.preview.height / slide.height,
    ) *
      1.05
  )
    return;
  const level = Math.max(
      0,
      Math.min(8, Math.floor(Math.log2(1 / state.scale))),
    ),
    size = 256 * 2 ** level;
  const x0 = Math.max(0, Math.floor(-state.x / state.scale / size)),
    y0 = Math.max(0, Math.floor(-state.y / state.scale / size));
  const x1 = Math.min(
      Math.ceil(slide.width / size) - 1,
      Math.floor((width - state.x) / state.scale / size),
    ),
    y1 = Math.min(
      Math.ceil(slide.height / size) - 1,
      Math.floor((height - state.y) / state.scale / size),
    );
  let requested = 0;
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      const key = `${level}:${x}:${y}`;
      let tile = state.tiles.get(key);
      if (!tile && requested++ < 80) {
        tile = new Image();
        const version = state.version;
        tile.onload = () => {
          if (version === state.version) redrawSoon();
        };
        tile.src = url("/api/tile", { id: slide.id, level, x, y });
        state.tiles.set(key, tile);
      }
      if (tile?.complete && tile.naturalWidth)
        ctx.drawImage(
          tile,
          state.x + x * size * state.scale,
          state.y + y * size * state.scale,
          Math.min(size, slide.width - x * size) * state.scale,
          Math.min(size, slide.height - y * size) * state.scale,
        );
    }
  if (state.tiles.size > 800) {
    const entries = [...state.tiles.keys()];
    for (const key of entries.slice(0, 200)) state.tiles.delete(key);
  }
}
function draw() {
  const { width, height } = dimensions();
  ctx.clearRect(0, 0, width, height);
  if (!state.slide || !state.preview) return;
  ctx.drawImage(
    state.preview,
    state.x,
    state.y,
    state.slide.width * state.scale,
    state.slide.height * state.scale,
  );
  drawTiles(width, height);
  const map = state.heatmap,
    show = $("heatmap-toggle").checked && map;
  const selected = state.region ? new Set(state.region.indices) : new Set();
  state.slide.pixels.forEach(([px, py], i) => {
    const x = state.x + px * state.scale,
      y = state.y + py * state.scale;
    if (x < -10 || x > width + 10 || y < -10 || y > height + 10) return;
    const radius = Math.max(2.5, Math.min(16, 32 * state.scale));
    if (show) {
      const t =
        map.max === map.min
          ? 0.5
          : Math.max(
              0,
              Math.min(1, (map.values[i] - map.min) / (map.max - map.min)),
            );
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${heatColor(t)},${$("opacity").value / 100})`;
      ctx.fill();
    } else if (state.mode !== "pan" || selected.has(i)) {
      ctx.beginPath();
      ctx.arc(
        x,
        y,
        selected.has(i) ? Math.max(4, radius) : Math.max(2, radius * 0.35),
        0,
        Math.PI * 2,
      );
      ctx.fillStyle = selected.has(i) ? "#087f78" : "#148b7977";
      ctx.fill();
    }
    if (selected.has(i)) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  });
  const bounds =
    state.drag && state.mode === "region"
      ? [
          state.drag.start.x,
          state.drag.start.y,
          state.drag.current.x,
          state.drag.current.y,
        ]
      : state.selection?.bounds?.map(
          (v, i) => v * state.scale + (i % 2 === 0 ? state.x : state.y),
        );
  if (bounds) {
    const [x0, y0, x1, y1] = bounds;
    ctx.fillStyle = "#0b8a7920";
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.strokeStyle = "#078575";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
  }
  $("zoom-label").textContent =
    Math.round((state.scale / state.fitScale) * 100) + "%";
}
canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    zoom(Math.exp(-event.deltaY * 0.0015), position(event));
  },
  { passive: false },
);
canvas.addEventListener("pointerdown", (event) => {
  if (!state.slide) return;
  if (state.mode !== "pan" && !state.genes.length) {
    toast("error.predictFirst");
    return;
  }
  canvas.setPointerCapture(event.pointerId);
  const p = position(event);
  state.drag = { start: p, current: p, x: state.x, y: state.y };
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const p = position(event);
  state.drag.current = p;
  if (state.mode === "pan") {
    state.x = state.drag.x + p.x - state.drag.start.x;
    state.y = state.drag.y + p.y - state.drag.start.y;
  }
  draw();
});
canvas.addEventListener(
  "pointerup",
  guarded(async (event) => {
    if (!state.drag) return;
    const drag = state.drag;
    state.drag = null;
    const end = position(event);
    if (state.mode === "region") {
      if (Math.hypot(end.x - drag.start.x, end.y - drag.start.y) < 3) return;
      const a = world(drag.start),
        b = world(end);
      await selectRegion({ bounds: [a.x, a.y, b.x, b.y] });
    } else if (state.mode === "spot") {
      const p = world(end);
      let nearest = -1,
        distance = Infinity;
      state.slide.pixels.forEach(([x, y], i) => {
        const d = Math.hypot(x - p.x, y - p.y);
        if (d < distance) {
          distance = d;
          nearest = i;
        }
      });
      if (distance * state.scale > 25) {
        toast("error.clickSpot");
        return;
      }
      await selectRegion({ spot: nearest });
    }
    draw();
  }),
);
canvas.addEventListener("pointercancel", () => {
  state.drag = null;
  draw();
});
function rankingRows(genes, limit = 50) {
  const fragment = document.createDocumentFragment(),
    max = Math.max(...genes.map((g) => Math.abs(g.value)), 1e-8);
  for (const [i, gene] of genes.slice(0, limit).entries()) {
    const row = element("div", "rank-row");
    row.tabIndex = 0;
    row.title = t("showGene", { gene: gene.gene });
    const bar = element("span", "rank-bar"),
      inner = element("i");
    inner.style.width = (Math.max(0, gene.value) / max) * 100 + "%";
    bar.append(inner);
    row.append(
      element("span", "rank-number", String(i + 1).padStart(2, "0")),
      element("span", "rank-gene", gene.gene),
      bar,
      element("span", "rank-value", number(gene.value, 3)),
    );
    const show = guarded(async () => {
      $("gene-input").value = gene.gene;
      await heatmap(gene.gene);
    });
    row.addEventListener("click", show);
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter") show();
    });
    fragment.append(row);
  }
  return fragment;
}
async function selectRegion(selection) {
  const version = state.version;
  const result = await api("/api/region", {
    id: state.slide.id,
    selection: { ...selection, tissue: $("search-tissue").value },
  });
  if (version !== state.version) return;
  state.selection = selection;
  state.region = result;
  renderSelection();
  draw();
}
function sourceTag(source) {
  return element(
    "span",
    "source-tag " + (source === "predicted" ? "predicted" : ""),
    t(source === "predicted" ? "predicted" : "measured"),
  );
}
function thumbnail(row) {
  const img = element("img");
  img.src = url("/api/thumbnail", { id: row.id });
  img.alt = t("prototypeAlt", { tissue: tissueName(row.tissue) });
  img.loading = "lazy";
  return img;
}
function renderRetrieval(rows) {
  const box = $("retrieval");
  box.replaceChildren();
  if (!rows.length) {
    empty(box, t("noMatchedPrototypes"));
    return;
  }
  for (const row of rows) {
    const card = element("button", "prototype-mini");
    const info = element("div", "info"),
      line = element("div", "row");
    line.append(
      element("span", "similarity", number(row.similarity * 100, 1) + "%"),
      sourceTag(row.source),
    );
    info.append(
      line,
      element(
        "div",
        "name",
        (row.slide_name || row.slide) + " / " + t("spotLabel", { spot: number(row.spot) }),
      ),
    );
    card.append(thumbnail(row), info);
    card.addEventListener("click", () => prototypeDetail(row));
    box.append(card);
  }
}
function prototypeDetail(row) {
  state.prototype = row;
  const box = $("prototype-detail");
  box.replaceChildren();
  const top = element("div", "detail-top"),
    info = element("div");
  info.append(
    element("span", "badge", tissueName(row.tissue)),
    sourceTag(row.source),
    element("p", "", (row.slide_name || row.slide) + " · " + t("spotLabel", { spot: number(row.spot) })),
    element(
      "p",
      "",
      t("coordinates", { x: number(row.pixel_x), y: number(row.pixel_y) }),
    ),
  );
  if (row.similarity !== undefined)
    info.append(
      element("p", "", t("cosineSimilarity", { value: number(row.similarity * 100, 2) })),
    );
  top.append(thumbnail(row), info);
  const ranking = element("div", "detail-ranking");
  ranking.append(rankingRows(row.top_genes, 8));
  box.append(top, element("h3", "", t("representativeExpression")), ranking);
  const button = element("button", "button full", t("locatePrototype"));
  button.addEventListener(
    "click",
    guarded(async () => {
      $("prototype-dialog").close();
      showPage("workspace");
      const id =
        state.catalog.slides.find((s) => s.id === row.slide)?.id ||
        row.dataset + ":" + row.slide;
      await openSlide(id);
      const { width, height } = dimensions();
      state.scale = Math.max(state.fitScale, 0.9);
      state.x = width / 2 - row.pixel_x * state.scale;
      state.y = height / 2 - row.pixel_y * state.scale;
      setMode("spot");
      draw();
      if (state.genes.length) await selectRegion({ spot: row.spot });
    }),
  );
  box.append(button);
  if (!$("prototype-dialog").open) $("prototype-dialog").showModal();
}
async function heatmap(gene) {
  if (!state.genes.includes(gene)) throw new LocalizedError("error.geneNotInPanel");
  const version = state.version;
  const result = await api(url("/api/heatmap", { id: state.slide.id, gene }));
  if (version !== state.version) return;
  state.heatmap = result;
  $("heatmap-toggle").checked = true;
  $("heatmap-range").textContent =
    number(result.min, 2) + " → " + number(result.max, 2);
  draw();
}
async function job(kind, tissue) {
  const id = state.slide.id,
    version = state.version;
  state.busy = true;
  controls();
  $("job-status").hidden = false;
  state.job = { message_key: "job.submitting" };
  $("job-message").textContent = jobMessage(state.job);
  try {
    let result = await api("/api/" + kind, { id, tissue });
    state.job = result;
    while (["queued", "running"].includes(result.status)) {
      if (version === state.version)
        $("job-message").textContent = jobMessage(result);
      await new Promise((resolve) => setTimeout(resolve, 1200));
      result = await api(url("/api/jobs", { id: result.id }));
      state.job = result;
    }
    if (result.status === "failed") throw new LocalizedError(result.message_key || "error.serverFailure", result.message_params);
    toast(result.message_key, result.message_params);
    await catalog();
    if (version === state.version) {
      if (kind === "predict") {
        clearSelection();
        await loadPrediction();
      }
      $("job-message").textContent = jobMessage(result);
    }
  } finally {
    if (version === state.version) {
      state.busy = false;
      controls();
      $("job-status").hidden = true;
    }
  }
}
function showPage(page) {
  $("workspace-page").hidden = page !== "workspace";
  $("library-page").hidden = page !== "library";
  $("nav-workspace").classList.toggle("active", page === "workspace");
  $("nav-library").classList.toggle("active", page === "library");
  if (page === "workspace") resize();
}
function libraryCard(row) {
  const card = element("button", "prototype-card"),
    info = element("div", "info"),
    line = element("div", "row"),
    chips = element("div", "gene-chips");
  line.append(element("span", "badge", tissueName(row.tissue)), sourceTag(row.source));
  for (const gene of row.top_genes.slice(0, 3))
    chips.append(element("span", "", gene.gene));
  info.append(
    line,
    element("div", "name", row.slide_name || row.slide),
    element("p", "", t("spotLabel", { spot: number(row.spot) }) + " · " + labels[row.dataset]),
    chips,
  );
  card.append(thumbnail(row), info);
  card.addEventListener("click", () => prototypeDetail(row));
  return card;
}
async function loadLibrary(reset = true) {
  if (reset) {
    state.offset = 0;
    state.libraryRows = [];
    $("library-grid").replaceChildren();
  }
  const tissue = $("library-tissue").value,
    result = await api(url("/api/library", { tissue, offset: state.offset }));
  refreshTissues(result.tissues);
  $("library-tissue").value = tissue;
  state.libraryTissues = result.tissues;
  state.catalog.libraries = result.tissues;
  state.libraryRows.push(...result.prototypes);
  state.offset += result.prototypes.length;
  state.libraryHasMore = result.prototypes.length === 24;
  renderLibrary();
}
function renderLibrary() {
  const tissue = $("library-tissue").value;
  $("library-stats").replaceChildren();
  for (const item of state.libraryTissues) {
    const card = element("div", "stat-card");
    card.append(element("h3", "", tissueName(item.tissue)),
      element("div", "stat-number", number(item.count)),
      element("p", "", t("libraryStats", {
        slides: number(item.slides), measured: number(item.measured), predicted: number(item.predicted),
      })));
    $("library-stats").append(card);
  }
  $("library-grid").replaceChildren(...state.libraryRows.map(libraryCard));
  if (!state.libraryRows.length) empty($("library-grid"), t("libraryEmpty"));
  $("library-more").hidden = !state.libraryHasMore;
  const count = state.libraryTissues.filter((item) => !tissue || item.tissue === tissue)
    .reduce((sum, item) => sum + item.count, 0);
  $("library-count").textContent = t("libraryCount", { count: number(count), shown: number(state.libraryRows.length) });
}
function changeLanguage(language) {
  const pendingTissue = I18n.tissueValue($("index-tissue").value);
  I18n.setLanguage(language);
  renderCatalog();
  renderSlideDetails();
  renderSelection();
  if (state.libraryTissues.length) renderLibrary();
  controls();
  $("index-tissue").value = tissueName(pendingTissue);
  $("upload-submit").textContent = t(state.uploading ? "uploading" : "uploadSubmit");
  if (state.heatmap) $("heatmap-range").textContent = number(state.heatmap.min, 2) + " → " + number(state.heatmap.max, 2);
  if (state.job) $("job-message").textContent = jobMessage(state.job);
  if ($("prototype-dialog").open && state.prototype) prototypeDetail(state.prototype);
  renderToast();
  resize();
}
$("language-select").addEventListener("change", (event) => changeLanguage(event.target.value));

$("slide-select").addEventListener(
  "change",
  guarded((event) => openSlide(event.target.value)),
);
for (const mode of ["pan", "region", "spot"])
  $("tool-" + mode).addEventListener("click", () => setMode(mode));
$("fit").addEventListener("click", fit);
for (const [id, factor] of [
  ["zoom-in", 1.5],
  ["zoom-out", 1 / 1.5],
])
  $(id).addEventListener("click", () => {
    const { width, height } = dimensions();
    zoom(factor, { x: width / 2, y: height / 2 });
  });
$("predict").addEventListener(
  "click",
  guarded(() => job("predict")),
);
$("whole-slide").addEventListener(
  "click",
  guarded(() => selectRegion({})),
);
$("clear-selection").addEventListener("click", clearSelection);
$("search-tissue").addEventListener(
  "change",
  guarded(() => state.selection && selectRegion(state.selection)),
);
$("gene-input").addEventListener(
  "change",
  guarded((event) => heatmap(event.target.value.trim())),
);
$("heatmap-toggle").addEventListener(
  "change",
  guarded(async () => {
    if ($("heatmap-toggle").checked && !state.heatmap) {
      const gene = $("gene-input").value || state.genes[0];
      $("gene-input").value = gene;
      await heatmap(gene);
    }
    draw();
  }),
);
$("opacity").addEventListener("input", draw);
$("download").addEventListener("click", () => {
  if (state.slide)
    window.location.href = url("/api/download", { id: state.slide.id });
});
$("upload-open").addEventListener("click", () =>
  $("upload-dialog").showModal(),
);
for (const button of document.querySelectorAll(".close-dialog"))
  button.addEventListener("click", () => button.closest("dialog").close());
$("upload-form").addEventListener(
  "submit",
  guarded(async (event) => {
    event.preventDefault();
    if (!event.target.elements.image.files.length) throw new LocalizedError("error.selectImage");
    const button = $("upload-submit");
    button.disabled = true;
    state.uploading = true;
    button.textContent = t("uploading");
    try {
      const response = await fetch("/api/upload", {
          method: "POST",
          body: new FormData(event.target),
        }),
        result = await response.json();
      if (!response.ok) throw new LocalizedError(result.error_key || "error.requestFailed", result.error_params);
      $("upload-dialog").close();
      event.target.reset();
      await catalog();
      showPage("workspace");
      await openSlide(result.id);
      toast("uploadSuccess");
    } finally {
      button.disabled = false;
      state.uploading = false;
      button.textContent = t("uploadSubmit");
    }
  }),
);
$("index-open").addEventListener("click", () => {
  $("index-tissue").value = tissueName(state.slide.tissue);
  $("index-dialog").showModal();
});
$("index-form").addEventListener(
  "submit",
  guarded(async (event) => {
    event.preventDefault();
    const tissue = I18n.tissueValue($("index-tissue").value);
    if (!tissue || tissue.length > 80) throw new LocalizedError("error.tissueName");
    $("index-dialog").close();
    await job("index", tissue);
  }),
);
$("nav-workspace").addEventListener("click", () => showPage("workspace"));
$("nav-library").addEventListener(
  "click",
  guarded(async () => {
    showPage("library");
    await loadLibrary();
  }),
);
$("library-tissue").addEventListener(
  "change",
  guarded(() => loadLibrary()),
);
$("library-refresh").addEventListener(
  "click",
  guarded(() => loadLibrary()),
);
$("library-more").addEventListener(
  "click",
  guarded(() => loadLibrary(false)),
);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !document.querySelector("dialog[open]"))
    clearSelection();
});
new ResizeObserver(resize).observe($("canvas-wrap"));
guarded(async () => {
  await I18n.ready;
  await catalog();
  const preferred =
    state.catalog.slides.find((s) => s.id === "her2st:B2") ||
    state.catalog.slides[0];
  await openSlide(preferred.id);
})();
