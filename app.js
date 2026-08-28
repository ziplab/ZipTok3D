const TOKENS = [1, 2, 4, 8, 16, 32, 64, 128];
const PASSES = [1, 2, 3, 4, 5, 6];

const metricConfig = {
  query_iou: { label: "Query IoU", higher: true, residual: value => 100 - value, format: value => value.toFixed(2) },
  mesh_fscore_0p02: { label: "Mesh F1@0.02", higher: true, residual: value => 100 - value, format: value => value.toFixed(2) },
  mesh_cd: { label: "Mesh CD", higher: false, residual: value => value, format: value => value.toFixed(4) },
};

const explorerState = {
  dataset: "ShapeNet",
  metric: "query_iou",
  token: 1,
  passes: 5,
};

const metricData = { ShapeNet: {}, TRELLIS: {} };
const ASSET_VERSION = "media8";

let lazyVideoObserver;

function loadVideo(video) {
  if (!video.dataset.src) return;
  video.src = video.dataset.src;
  delete video.dataset.src;
  video.load();
}

function observeVideo(video) {
  if (!("IntersectionObserver" in window)) {
    loadVideo(video);
    video.play().catch(() => {});
    return;
  }
  if (!lazyVideoObserver) {
    lazyVideoObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        const target = entry.target;
        if (!target.isConnected) {
          lazyVideoObserver.unobserve(target);
          return;
        }
        if (entry.isIntersecting) {
          loadVideo(target);
          target.play().catch(() => {});
        } else {
          target.pause();
        }
      });
    }, { rootMargin: "240px 0px" });
  }
  lazyVideoObserver.observe(video);
}

function assetUrl(path) {
  return `${path}?v=${ASSET_VERSION}`;
}

const results = {
  ShapeNet: [
    { method: "3DILG", tokens: 512, iou: 95.9, cd: 0.013, f1: 98.0 },
    { method: "VecSet", tokens: 512, iou: 96.3, cd: 0.013, f1: 98.0 },
    { method: "COD-VAE", tokens: 32, iou: 97.1, cd: 0.012, f1: 97.8 },
    { method: "COD-VAE", tokens: 64, iou: 97.5, cd: 0.012, f1: 98.0 },
    { method: "ZipTok3D", tokens: 1, iou: 96.8, cd: 0.012, f1: 97.8, ours: true },
    { method: "ZipTok3D", tokens: 2, iou: 96.9, cd: 0.012, f1: 97.8, ours: true },
    { method: "ZipTok3D", tokens: 4, iou: 96.9, cd: 0.012, f1: 97.9, ours: true },
  ],
  TRELLIS: [
    { method: "VecSet", tokens: 512, iou: 71.47, cd: 0.0249, f1: 88.81 },
    { method: "COD-VAE", tokens: 2, iou: 45.85, cd: 0.0674, f1: 53.95 },
    { method: "COD-VAE", tokens: 32, iou: 75.25, cd: 0.0172, f1: 95.67 },
    { method: "COD-VAE", tokens: 64, iou: 75.75, cd: 0.0168, f1: 95.98 },
    { method: "ZipTok3D", tokens: 1, iou: 75.18, cd: 0.0168, f1: 95.81, ours: true },
    { method: "ZipTok3D", tokens: 2, iou: 75.22, cd: 0.0167, f1: 95.86, ours: true },
    { method: "ZipTok3D", tokens: 4, iou: 75.31, cd: 0.0166, f1: 95.92, ours: true },
  ],
};

const refinementTrajectories = {
  lamp: {
    label: "Lamp",
    dataset: "ShapeNet",
    gt: "assets/traj_shapenet_lamp_gt.png",
    l1: "assets/traj_shapenet_lamp_l1.png",
    l3: "assets/traj_shapenet_lamp_l3.png",
    l5: "assets/traj_shapenet_lamp_l5.png",
  },
  bridge: {
    label: "Bridge",
    dataset: "TRELLIS",
    gt: "assets/traj_trellis_bridge_gt.png",
    l1: "assets/traj_trellis_bridge_l1.png",
    l3: "assets/traj_trellis_bridge_l3.png",
    l5: "assets/traj_trellis_bridge_l5.png",
  },
  frame: {
    label: "Frame building",
    dataset: "TRELLIS",
    gt: "assets/traj_trellis_frame_gt.png",
    l1: "assets/traj_trellis_frame_l1.png",
    l3: "assets/traj_trellis_frame_l3.png",
    l5: "assets/traj_trellis_frame_l5.png",
  },
};

const depthSweepGroups = [
  { dataset: "TRELLIS", id: "6889cb7c4faea430ef6e8c32be4e9c38bd8f4e6a439c92012a4e230f2e9c1352", label: "Bridge" },
].map(group => ({ ...group, key: `${group.dataset.toLowerCase()}__${group.id}` }));

const sampleNames = {
  "5fc39e0ecc8e50f0902a571380e15334": "Side table",
  "862d685006637dfef630324ef3baae90": "Aircraft",
  "8c942a8e196a9371a782a4379556c7": "Bench",
  "aea5192a4a7bda94d33646b0990bb4a": "Jet",
  "f144e93fe2a11c1f4c3a35cee92bb95b": "Glider",
  "4be2461bad10aa82a875c848d0fb1664": "Boat",
  "753452a3a8f44bd38b69f185154696a3": "Chair",
  "89054836cd41bfb9820018801b237b3d": "Table",
  "b04647659c599ade7fb4fbe822d98e36": "Piano",
  "f1e439307b834015770a0ff1161fa15a": "Mug",
  "6889cb7c4faea430ef6e8c32be4e9c38bd8f4e6a439c92012a4e230f2e9c1352": "Bridge",
};

const fixedSampleIds = {
  1: {
    ShapeNet: ["5fc39e0ecc8e50f0902a571380e15334", "862d685006637dfef630324ef3baae90", "8c942a8e196a9371a782a4379556c7", "aea5192a4a7bda94d33646b0990bb4a", "f144e93fe2a11c1f4c3a35cee92bb95b"],
    TRELLIS: [],
  },
  4: {
    ShapeNet: ["753452a3a8f44bd38b69f185154696a3", "862d685006637dfef630324ef3baae90", "89054836cd41bfb9820018801b237b3d", "b04647659c599ade7fb4fbe822d98e36", "f1e439307b834015770a0ff1161fa15a"],
    TRELLIS: ["6889cb7c4faea430ef6e8c32be4e9c38bd8f4e6a439c92012a4e230f2e9c1352"],
  },
};

function mediaAsset(directory, group, stem, extension = "mp4") {
  const file = `${group}--${stem}.${extension}`;
  return `assets/media/${directory}/${file}`;
}

function posterAsset(directory, group, stem) {
  const file = `${group}--${stem}.jpg`;
  return `assets/posters/${directory}/${file}`;
}

const reconstructionSamples = Object.entries(fixedSampleIds).flatMap(([token, datasets]) =>
  Object.entries(datasets).flatMap(([dataset, ids]) => ids.map(id => {
    const setting = `fixed_k${token}_l5_vs_codvae32`;
    const group = `${dataset.toLowerCase()}__${id}`;
    const stem = `ziptok3d_k${token}_l5`;
    return {
      dataset,
      token: Number(token),
      id,
      label: sampleNames[id] || "Shape",
      kind: "video",
      config: `K = ${token}, L = 5`,
      file: mediaAsset("reconstructions", `${setting}--${group}`, stem),
      poster: posterAsset("reconstructions", `${setting}--${group}`, stem),
    };
  }))
);

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map(line => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

async function loadCsv(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Unable to load ${path}`);
  return parseCsv(await response.text());
}

function storeRows(dataset, queryRows, meshRows) {
  queryRows.filter(row => row.model === "flex").forEach(row => {
    const key = `${Number(row.token_count)}-${Number(row.loop_count)}`;
    metricData[dataset][key] = { query_iou: Number(row.query_iou) };
  });
  meshRows.filter(row => row.model === "flex").forEach(row => {
    const key = `${Number(row.token_count)}-${Number(row.loop_count)}`;
    metricData[dataset][key] = {
      ...metricData[dataset][key],
      mesh_cd: Number(row.mesh_cd),
      mesh_fscore_0p02: Number(row.mesh_fscore_0p02),
    };
  });
}

function mixColor(low, high, amount) {
  const channel = index => Math.round(low[index] + (high[index] - low[index]) * amount);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

function qualityScores(dataset, metric) {
  const config = metricConfig[metric];
  const residuals = Object.values(metricData[dataset]).map(row => config.residual(row[metric]));
  const minimum = Math.min(...residuals);
  const maximum = Math.max(...residuals);
  const denominator = Math.log(maximum / minimum);
  return value => {
    const residual = config.residual(value);
    if (denominator === 0) return 1;
    return Math.max(0, Math.min(1, Math.log(maximum / residual) / denominator));
  };
}

function selectedRow() {
  return metricData[explorerState.dataset][`${explorerState.token}-${explorerState.passes}`];
}

function updateExplorerControls() {
  document.querySelectorAll("[data-dataset]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.dataset === explorerState.dataset));
  });
  document.querySelectorAll("[data-metric]").forEach(button => {
    button.setAttribute("aria-selected", String(button.dataset.metric === explorerState.metric));
  });
  const tokenIndex = TOKENS.indexOf(explorerState.token);
  document.getElementById("explorer-k").value = String(tokenIndex);
  document.getElementById("explorer-l").value = String(explorerState.passes);
  document.getElementById("explorer-k-value").textContent = String(explorerState.token);
  document.getElementById("explorer-l-value").textContent = String(explorerState.passes);
}

function renderHeatmap() {
  const heatmap = document.getElementById("heatmap");
  const config = metricConfig[explorerState.metric];
  const quality = qualityScores(explorerState.dataset, explorerState.metric);
  heatmap.replaceChildren();

  const corner = document.createElement("div");
  corner.className = "heatmap-corner";
  corner.textContent = "L / K";
  heatmap.appendChild(corner);

  TOKENS.forEach(token => {
    const label = document.createElement("div");
    label.className = "heatmap-axis";
    label.textContent = String(token);
    heatmap.appendChild(label);
  });

  PASSES.forEach(passes => {
    const label = document.createElement("div");
    label.className = "heatmap-axis";
    label.textContent = String(passes);
    heatmap.appendChild(label);

    TOKENS.forEach(token => {
      const row = metricData[explorerState.dataset][`${token}-${passes}`];
      const value = row[explorerState.metric];
      const score = quality(value);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "heatmap-cell";
      button.setAttribute("role", "gridcell");
      button.setAttribute("aria-label", `${explorerState.dataset}, K ${token}, L ${passes}, ${config.label} ${config.format(value)}`);
      button.textContent = config.format(value);
      button.style.backgroundColor = mixColor([225, 238, 232], [8, 115, 99], score);
      button.style.color = score > 0.58 ? "#ffffff" : "#162020";
      if (token === explorerState.token && passes === explorerState.passes) button.classList.add("selected");
      if ((explorerState.dataset === "ShapeNet" && token === 1 && passes === 5) ||
          (explorerState.dataset === "TRELLIS" && token === 4 && passes === 5)) {
        button.classList.add("reported");
      }
      button.addEventListener("click", () => {
        explorerState.token = token;
        explorerState.passes = passes;
        renderExplorer();
      });
      heatmap.appendChild(button);
    });
  });
}

function renderSelection() {
  const row = selectedRow();
  document.getElementById("selection-dataset").textContent = `${explorerState.dataset} operating point`;
  document.getElementById("selection-k").textContent = `K = ${explorerState.token}`;
  document.getElementById("selection-l").textContent = `L = ${explorerState.passes}`;
  document.getElementById("selected-iou").textContent = row.query_iou.toFixed(2);
  document.getElementById("selected-cd").textContent = row.mesh_cd.toFixed(4);
  document.getElementById("selected-f1").textContent = row.mesh_fscore_0p02.toFixed(2);

  const base = metricData[explorerState.dataset][`${explorerState.token}-1`][explorerState.metric];
  const current = row[explorerState.metric];
  const config = metricConfig[explorerState.metric];
  let insight;
  if (explorerState.passes === 1) {
    insight = `Single-pass decoding makes quality depend most strongly on prefix length.`;
  } else {
    const signedChange = current - base;
    const favorable = config.higher ? signedChange : -signedChange;
    const unitChange = explorerState.metric === "mesh_cd" ? Math.abs(signedChange).toFixed(4) : Math.abs(signedChange).toFixed(2);
    const direction = favorable >= 0 ? "improves" : "changes";
    insight = `At K = ${explorerState.token}, ${explorerState.passes} passes ${direction} ${config.label} by ${unitChange} relative to L = 1.`;
  }
  document.getElementById("selection-insight").textContent = insight;
}

function renderExplorer() {
  updateExplorerControls();
  renderHeatmap();
  renderSelection();
}

function renderResults(dataset) {
  document.querySelectorAll("[data-result-dataset]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.resultDataset === dataset));
  });
  const rows = results[dataset];
  const bestIou = Math.max(...rows.map(row => row.iou));
  const bestCd = Math.min(...rows.map(row => row.cd));
  const bestF1 = Math.max(...rows.map(row => row.f1));
  const body = document.getElementById("results-body");
  body.replaceChildren();
  rows.forEach(row => {
    const tr = document.createElement("tr");
    if (row.ours) tr.className = "ours";
    const cells = [
      { value: row.method },
      { value: row.tokens },
      { value: dataset === "ShapeNet" ? row.iou.toFixed(1) : row.iou.toFixed(2), best: row.iou === bestIou },
      { value: dataset === "ShapeNet" ? row.cd.toFixed(3) : row.cd.toFixed(4), best: row.cd === bestCd },
      { value: dataset === "ShapeNet" ? row.f1.toFixed(1) : row.f1.toFixed(2), best: row.f1 === bestF1 },
    ];
    cells.forEach(cell => {
      const td = document.createElement("td");
      td.textContent = String(cell.value);
      if (cell.best) td.className = "best";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

function videoSource(file, poster, label) {
  const video = document.createElement("video");
  video.dataset.src = assetUrl(file);
  video.poster = assetUrl(poster);
  video.autoplay = true;
  video.muted = true;
  video.loop = true;
  video.controls = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.dataset.pauseOffscreen = "true";
  video.setAttribute("aria-label", label);
  video.addEventListener("error", () => video.classList.add("media-error"), { once: true });
  observeVideo(video);
  return video;
}

function commandButton(id, label, icon) {
  const button = document.createElement("button");
  button.type = "button";
  button.id = id;
  button.className = "media-command";
  button.innerHTML = `<i data-lucide="${icon}" aria-hidden="true"></i><span>${label}</span>`;
  return button;
}

function videoGroupControls(videos, controls) {
  let commandLock = false;
  let masterVideo = videos[0];

  const setToggleLabel = playing => {
    if (!controls.toggle) return;
    controls.toggle.innerHTML = `<i data-lucide="${playing ? "pause" : "play"}" aria-hidden="true"></i><span>${playing ? "Pause all" : "Play all"}</span>`;
    if (window.lucide) window.lucide.createIcons();
  };

  const updateTime = () => {
    if (!controls.status || !masterVideo || !Number.isFinite(masterVideo.duration)) return;
    const elapsed = masterVideo.currentTime.toFixed(1);
    const duration = masterVideo.duration.toFixed(1);
    controls.status.textContent = `${elapsed}s / ${duration}s`;
  };

  const playAll = () => {
    commandLock = true;
    videos.forEach(video => {
      loadVideo(video);
      video.play().catch(() => {});
    });
    setToggleLabel(true);
    window.setTimeout(() => { commandLock = false; }, 300);
  };

  const pauseAll = () => {
    commandLock = true;
    videos.forEach(video => video.pause());
    setToggleLabel(false);
    window.setTimeout(() => { commandLock = false; }, 300);
  };

  const seekAll = delta => {
    const current = masterVideo?.currentTime || 0;
    const duration = Number.isFinite(masterVideo?.duration) && masterVideo.duration > 0
      ? masterVideo.duration
      : current + delta;
    const nextTime = Math.max(0, Math.min(duration, current + delta));
    commandLock = true;
    videos.forEach(video => { video.currentTime = nextTime; });
    updateTime();
    window.setTimeout(() => { commandLock = false; }, 100);
  };

  videos.forEach(video => {
    video.addEventListener("play", () => {
      masterVideo = video;
      if (!commandLock) playAll();
    });
    video.addEventListener("pause", () => {
      if (!commandLock) pauseAll();
    });
    video.addEventListener("timeupdate", () => {
      if (video === masterVideo && !commandLock) {
        videos.forEach(other => {
          if (other !== video && Math.abs(other.currentTime - video.currentTime) > 0.08) {
            other.currentTime = video.currentTime;
          }
        });
      }
      updateTime();
    });
  });

  controls.toggle?.addEventListener("click", () => {
    if (videos.some(video => video.paused)) playAll();
    else pauseAll();
  });
  controls.back?.addEventListener("click", () => seekAll(-0.5));
  controls.forward?.addEventListener("click", () => seekAll(0.5));
  videos.forEach(video => video.addEventListener("loadedmetadata", updateTime, { once: true }));
  setToggleLabel(false);
}

function renderDepthSelectors() {
  const picker = document.getElementById("depth-sample-picker");
  picker.replaceChildren(...depthSweepGroups.map((group, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.depthSample = String(index);
    button.setAttribute("aria-pressed", String(index === 0));
    button.innerHTML = `<strong>${group.label}</strong><span>${group.dataset}</span>`;
    return button;
  }));
}

function renderDepthSweep(index) {
  const group = depthSweepGroups[index];
  document.querySelectorAll("[data-depth-sample]").forEach(button => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.depthSample) === index));
  });
  document.getElementById("depth-sample-name").textContent = group.label;
  document.getElementById("depth-sample-config").textContent = "K = 2, shared decoder";
  const grid = document.getElementById("depth-video-grid");
  const videos = [];
  grid.replaceChildren(...[1, 3, 5].map(depth => {
    const figure = document.createElement("figure");
    const source = {
      file: mediaAsset("depth-sweep", group.key, `ziptok3d_k2_l${depth}`),
      poster: posterAsset("depth-sweep", group.key, `ziptok3d_k2_l${depth}`),
    };
    const video = videoSource(source.file, source.poster, `${group.label}, ${group.dataset}, reconstruction at L = ${depth}`);
    videos.push(video);
    const caption = document.createElement("figcaption");
    caption.innerHTML = `<strong>L = ${depth}</strong><span>${depth === 1 ? "coarse structure" : depth === 3 ? "structure recovered" : "surface refined"}</span>`;
    figure.append(video, caption);
    return figure;
  }));
  videoGroupControls(videos, {
    toggle: document.getElementById("depth-play-toggle"),
    back: document.getElementById("depth-seek-back"),
    forward: document.getElementById("depth-seek-forward"),
    status: document.getElementById("depth-time"),
  });
  if (window.lucide) window.lucide.createIcons();
}

function renderReconstructionSample(sample) {
  const article = document.createElement("article");
  article.className = `reconstruction-sample ${sample.kind === "video" ? "reconstruction-video-sample" : ""} ${sample.dataset.toLowerCase()}`;

  const header = document.createElement("header");
  const title = document.createElement("strong");
  title.textContent = sample.label;
  const config = document.createElement("span");
  config.textContent = sample.config;
  header.append(title, config);
  article.appendChild(header);

  if (sample.kind === "video") {
    const video = videoSource(sample.file, sample.poster, `${sample.label}, ZipTok3D reconstruction`);
    article.appendChild(video);
  } else {
    const comparison = document.createElement("div");
    comparison.className = "sample-comparison-grid";
    [
      ["Ground truth", sample.gt],
      ["COD-VAE, K = 32", sample.codvae],
      ["ZipTok3D", sample.ziptok],
    ].forEach(([label, source]) => {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.src = assetUrl(source);
      image.alt = `${sample.label}, ${label}`;
      image.loading = "lazy";
      const caption = document.createElement("figcaption");
      caption.textContent = label;
      figure.append(image, caption);
      comparison.appendChild(figure);
    });
    article.appendChild(comparison);
  }
  return article;
}

const galleryState = { dataset: "all", token: "all" };
let depthSweepIndex = 0;

function setGallery(dataset = galleryState.dataset, token = galleryState.token) {
  galleryState.dataset = dataset;
  galleryState.token = token;
  document.querySelectorAll("[data-gallery-dataset]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.galleryDataset === dataset));
  });
  document.querySelectorAll("[data-gallery-token]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.galleryToken === token));
  });
  const gallery = document.getElementById("reconstruction-gallery");
  const dynamicSamples = reconstructionSamples.filter(sample =>
    (dataset === "all" || sample.dataset === dataset) &&
    (token === "all" || String(sample.token) === token),
  );
  gallery.replaceChildren(...dynamicSamples.map(renderReconstructionSample));
  document.getElementById("gallery-summary").textContent =
    `${dynamicSamples.length} ZipTok3D reconstructions | ${dataset === "all" ? "ShapeNet + TRELLIS" : dataset}${token === "all" ? "" : ` | K = ${token}`} | L = 5.`;
}

function bindControls() {
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");
  const closeNav = () => {
    navToggle?.setAttribute("aria-expanded", "false");
    navToggle?.setAttribute("aria-label", "Open navigation");
    navLinks?.classList.remove("is-open");
  };
  navToggle?.addEventListener("click", () => {
    const open = navToggle.getAttribute("aria-expanded") === "true";
    navToggle.setAttribute("aria-expanded", String(!open));
    navToggle.setAttribute("aria-label", open ? "Open navigation" : "Close navigation");
    navLinks?.classList.toggle("is-open", !open);
  });
  navLinks?.querySelectorAll("a").forEach(link => link.addEventListener("click", closeNav));
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeNav();
  });
  document.addEventListener("click", event => {
    if (!navLinks?.classList.contains("is-open")) return;
    if (!navLinks.contains(event.target) && !navToggle?.contains(event.target)) closeNav();
  });

  document.querySelectorAll("[data-dataset]").forEach(button => {
    button.addEventListener("click", () => {
      explorerState.dataset = button.dataset.dataset;
      explorerState.token = explorerState.dataset === "ShapeNet" ? 1 : 4;
      explorerState.passes = 5;
      renderExplorer();
    });
  });
  document.querySelectorAll("[data-metric]").forEach(button => {
    button.addEventListener("click", () => {
      explorerState.metric = button.dataset.metric;
      renderExplorer();
    });
  });
  document.getElementById("explorer-k").addEventListener("input", event => {
    explorerState.token = TOKENS[Number(event.target.value)];
    renderExplorer();
  });
  document.getElementById("explorer-l").addEventListener("input", event => {
    explorerState.passes = Number(event.target.value);
    renderExplorer();
  });
  document.querySelectorAll("[data-result-dataset]").forEach(button => {
    button.addEventListener("click", () => renderResults(button.dataset.resultDataset));
  });
  document.querySelectorAll("[data-depth-sample]").forEach(button => {
    button.addEventListener("click", () => {
      depthSweepIndex = Number(button.dataset.depthSample);
      renderDepthSweep(depthSweepIndex);
    });
  });
  document.getElementById("depth-prev").addEventListener("click", () => {
    depthSweepIndex = (depthSweepIndex + depthSweepGroups.length - 1) % depthSweepGroups.length;
    renderDepthSweep(depthSweepIndex);
  });
  document.getElementById("depth-next").addEventListener("click", () => {
    depthSweepIndex = (depthSweepIndex + 1) % depthSweepGroups.length;
    renderDepthSweep(depthSweepIndex);
  });
  document.querySelectorAll("[data-gallery-dataset]").forEach(button => {
    button.addEventListener("click", () => setGallery(button.dataset.galleryDataset, galleryState.token));
  });
  document.querySelectorAll("[data-gallery-token]").forEach(button => {
    button.addEventListener("click", () => setGallery(galleryState.dataset, button.dataset.galleryToken));
  });
}

async function initialize() {
  renderDepthSelectors();
  bindControls();
  renderResults("ShapeNet");
  renderDepthSweep(depthSweepIndex);
  setGallery("all", "all");
  if (window.lucide) window.lucide.createIcons();

  try {
    const [shapeQuery, shapeMesh, trellisQuery, trellisMesh] = await Promise.all([
      loadCsv("data/shapenet-query.csv"),
      loadCsv("data/shapenet-mesh.csv"),
      loadCsv("data/trellis-query.csv"),
      loadCsv("data/trellis-mesh.csv"),
    ]);
    storeRows("ShapeNet", shapeQuery, shapeMesh);
    storeRows("TRELLIS", trellisQuery, trellisMesh);
    renderExplorer();
  } catch (error) {
    const heatmap = document.getElementById("heatmap");
    heatmap.textContent = "Metric data could not be loaded.";
    console.error(error);
  }
}

document.addEventListener("DOMContentLoaded", initialize);
