/* T-SAO command client. All imagery + metrics = live API (real M1 inference).
   Contracts: /api/health, /api/scenes, POST /api/predict/{scene} (SSE),
   /api/scenes/{scene}/layer/{vv,vh,dem,prob,mask,gt}, /api/scenes/{scene}/export. */
(function () {
  "use strict";
  var RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var FINE = window.matchMedia("(pointer: fine)").matches;
  var $ = function (id) { return document.getElementById(id); };
  var state = { scenes: [], selected: null, result: null, mode: "source", src: "vv", zoom: 1, pan: [0, 0] };

  var LEGENDS = {
    source: '<span><i class="sw" style="background:#8a8f8c"></i>SAR backscatter (dB) / terrain (m)</span><span class="muted">Grey = NoData</span>',
    prob: '<span>Low</span><span class="prob-bar" role="img" aria-label="Probability scale from low to high"></span><span>High</span><span class="muted">Flood probability</span>',
    mask: '<span><i class="sw" style="background:#3ee6c4"></i>Flood</span><span><i class="sw" style="background:#10161d"></i>Non-flood</span><span><i class="sw" style="background:#2a3438"></i>NoData</span>',
    gt: '<span><i class="sw" style="background:#3ee6c4"></i>Flood (label)</span><span><i class="sw" style="background:#10161d"></i>Non-flood</span><span class="tag">benchmark only</span>',
    compare: '<span>Drag across the scene — SAR source vs live prediction.</span>',
  };
  var CAPTIONS = {
    source: { vv: "VV · Sentinel-1 backscatter", vh: "VH · Sentinel-1 backscatter", dem: "DEM · Copernicus terrain (m)" },
    prob: "PROB · Flood probability (M1, live)",
    mask: "FLOOD · Predicted mask (M1, live)",
    gt: "TRUTH · Hand-labeled ground truth",
    compare: "COMPARE · Source vs prediction",
  };

  function fmt(n) { return n.toLocaleString("en-US"); }

  /* ---------- nav ---------- */
  var nav = $("topNav");
  function onScroll() { nav.classList.toggle("scrolled", window.scrollY > 40); }
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  var secIO = new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      if (!e.isIntersecting) return;
      document.querySelectorAll("[data-nav]").forEach(function (a) {
        a.classList.toggle("active", a.getAttribute("data-nav") === e.target.id);
      });
    });
  }, { rootMargin: "-40% 0px -55% 0px" });
  ["overview", "analyze", "method"].forEach(function (id) {
    var el = document.getElementById(id); if (el) secIO.observe(el);
  });

  /* ---------- pointer parallax: env + status card drift ---------- */
  if (!RM && FINE) {
    var env = $("envImg"), card = document.querySelector("[data-depth]");
    var tx = 0, ty = 0, cx = 0, cy = 0, raf = null;
    function loop() {
      cx += (tx - cx) * 0.06; cy += (ty - cy) * 0.06;
      if (env) env.style.translate = (cx * -14) + "px " + (cy * -10) + "px";
      if (card) card.style.translate = (cx * 14) + "px " + (cy * 10) + "px";
      if (Math.abs(tx - cx) > 0.0005 || Math.abs(ty - cy) > 0.0005) raf = requestAnimationFrame(loop);
      else raf = null;
    }
    document.querySelector(".hero").addEventListener("pointermove", function (e) {
      var r = e.currentTarget.getBoundingClientRect();
      tx = (e.clientX - r.left) / r.width - 0.5;
      ty = (e.clientY - r.top) / r.height - 0.5;
      if (!raf) raf = requestAnimationFrame(loop);
    });
    /* ---------- magnetic buttons ---------- */
    document.querySelectorAll(".magnetic").forEach(function (b) {
      b.addEventListener("pointermove", function (e) {
        var r = b.getBoundingClientRect();
        var x = (e.clientX - r.left - r.width / 2) / r.width;
        var y = (e.clientY - r.top - r.height / 2) / r.height;
        b.style.transform = "translate(" + (x * 6) + "px," + (y * 5) + "px)";
      });
      b.addEventListener("pointerleave", function () { b.style.transform = ""; });
    });
  }

  /* ---------- status ---------- */
  fetch("/api/health").then(function (r) { return r.json(); }).then(function () {
    $("sysDot").classList.add("ok");
    $("sysText").textContent = "M1 · READY";
    var hs = $("heroStatus"); if (hs) hs.textContent = "● Ready";
  }).catch(function () {
    $("sysText").textContent = "M1 · OFFLINE";
    var hs2 = $("heroStatus"); if (hs2) { hs2.textContent = "● Offline"; hs2.classList.remove("lime"); }
  });

  /* ---------- reveals ---------- */
  var io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach(function (el) { io.observe(el); });

  /* ---------- count-ups + ablation bars ---------- */
  var cio = new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      if (!e.isIntersecting) return;
      cio.unobserve(e.target);
      countUp(e.target, parseFloat(e.target.dataset.count), 4);
    });
  }, { threshold: 0.5 });
  document.querySelectorAll("[data-count]").forEach(function (el) { cio.observe(el); });
  var bio = new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      if (!e.isIntersecting) return;
      bio.unobserve(e.target);
      e.target.querySelectorAll(".bfill").forEach(function (f, i) {
        setTimeout(function () { f.style.width = f.dataset.w + "%"; }, RM ? 0 : i * 120);
      });
    });
  }, { threshold: 0.35 });
  if ($("ablBars")) bio.observe($("ablBars"));

  /* ---------- scenes ---------- */
  fetch("/api/scenes").then(function (r) { return r.json(); }).then(function (scenes) {
    state.scenes = scenes;
    var list = $("sceneList");
    scenes.forEach(function (s, i) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "scene" + (i === 0 ? " sel" : "");
      b.setAttribute("role", "option");
      b.setAttribute("aria-selected", i === 0 ? "true" : "false");
      b.innerHTML = '<img loading="lazy" alt="">' +
        '<span><span class="sn"></span><span class="sm"></span><span class="sa">● SELECTED</span></span>';
      b.querySelector("img").src = "/api/scenes/" + s.scene + "/layer/vv";
      b.querySelector("img").alt = "SAR preview of " + s.scene;
      b.querySelector(".sn").textContent = s.scene;
      b.querySelector(".sm").textContent = s.event + " · " + s.event_date;
      b.addEventListener("click", function () { selectScene(s.scene); });
      b.dataset.scene = s.scene;
      list.appendChild(b);
    });
    if (scenes.length) selectScene(scenes[0].scene, true);
  });
  function sceneById(id) {
    for (var i = 0; i < state.scenes.length; i++) if (state.scenes[i].scene === id) return state.scenes[i];
    return null;
  }
  function setDeck(status, cls) {
    var d = $("deckStatus");
    d.innerHTML = '<span class="dot ' + cls + '" aria-hidden="true"></span>' + status;
  }
  function selectScene(id, silent) {
    state.selected = id; state.result = null;
    document.querySelectorAll(".scene").forEach(function (el) {
      var on = el.dataset.scene === id;
      el.classList.toggle("sel", on);
      el.setAttribute("aria-selected", on ? "true" : "false");
    });
    var s = sceneById(id);
    if (s) {
      $("sceneMeta").innerHTML = "EVENT <b>" + s.event + "</b> · " + s.event_date +
        "<br>CENTRE <b>" + s.centre_lat + ", " + s.centre_lon + "</b>" +
        "<br>VALID PX <b>" + fmt(s.valid_pixels) + "</b>";
      $("deckTitle").textContent = "SCENE · " + s.scene;
      $("hudScene").textContent = "SCENE " + s.scene;
      $("hudEvent").textContent = "EVENT " + s.event + " · " + s.event_date;
      $("hudCoords").textContent = "CENTRE " + s.centre_lat + " / " + s.centre_lon + " · EPSG:4326";
    }
    $("resultBody").hidden = true; $("resultError").hidden = true; $("resultEmpty").hidden = false;
    $("benchBox").hidden = true;
    setDeck("IDLE", "idle");
    showMain("/api/scenes/" + id + "/layer/vv", "Sentinel-1 VV backscatter for " + id);
    setLegend(); setCaption(); resetZoom(); setStages(null);
    if (!silent) $("runState").textContent = id + " selected — run analysis to map flooding.";
  }
  $("sceneList").addEventListener("keydown", function (e) {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    var btns = Array.prototype.slice.call(document.querySelectorAll(".scene"));
    var i = btns.indexOf(document.activeElement);
    var n = e.key === "ArrowDown" ? i + 1 : i - 1;
    if (btns[n]) { e.preventDefault(); btns[n].focus(); selectScene(btns[n].dataset.scene); }
  });

  /* ---------- modes ---------- */
  var tabs = Array.prototype.slice.call(document.querySelectorAll('[role="tab"]'));
  function moveInk() {
    var cur = document.querySelector('[role="tab"][aria-selected="true"]');
    var ink = $("modesInk");
    if (!cur || !ink) return;
    ink.style.left = cur.offsetLeft + "px";
    ink.style.width = cur.offsetWidth + "px";
  }
  tabs.forEach(function (t, i) {
    t.addEventListener("click", function () { setMode(t.dataset.mode); });
    t.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      var n = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
      tabs[n].focus(); setMode(tabs[n].dataset.mode);
    });
  });
  window.addEventListener("resize", moveInk);
  window.addEventListener("load", moveInk);
  function setMode(m) {
    state.mode = m;
    tabs.forEach(function (t) { t.setAttribute("aria-selected", t.dataset.mode === m ? "true" : "false"); });
    moveInk();
    $("srcSub").style.display = m === "source" ? "flex" : "none";
    var cmp = m === "compare";
    $("viewport").classList.toggle("cmp-on", cmp);
    $("cmpSlider").hidden = !cmp;
    $("cmpLine").hidden = !cmp;
    refreshView();
  }
  document.querySelectorAll("[data-src]").forEach(function (b) {
    b.addEventListener("click", function () {
      document.querySelectorAll("[data-src]").forEach(function (x) {
        x.classList.remove("on"); x.setAttribute("aria-pressed", "false");
      });
      b.classList.add("on"); b.setAttribute("aria-pressed", "true");
      state.src = b.dataset.src;
      if (state.mode === "source") refreshView();
    });
  });
  function layerURL(kind) { return "/api/scenes/" + state.selected + "/layer/" + kind; }
  function refreshView() {
    var m = state.mode;
    if (m === "source") { showMain(layerURL(state.src), "Source " + state.src); $("viewTop").style.opacity = 0; }
    else if (m === "prob") { showMain(layerURL("prob"), "Flood probability"); }
    else if (m === "mask") { showMain(layerURL("mask"), "Predicted flood mask"); }
    else if (m === "gt") { showMain(layerURL("gt"), "Ground truth label"); }
    else if (m === "compare") {
      showMain(layerURL("vv"), "Source for comparison");
      var top = $("viewTop");
      top.style.opacity = "";
      top.src = layerURL("mask"); top.alt = "Prediction overlay";
      applyCmp();
    }
    setLegend(); setCaption();
  }
  function showMain(src, alt) {
    var main = $("viewMain");
    if (!RM) main.style.opacity = 0;
    main.onload = function () { main.style.opacity = 1; };
    main.src = src; main.alt = alt;
  }
  function setLegend() { $("legend").innerHTML = LEGENDS[state.mode] || ""; }
  function setCaption() {
    var c = CAPTIONS[state.mode];
    $("viewCap").textContent = typeof c === "string" ? c : (c[state.src] || c.vv);
  }
  $("cmpSlider").addEventListener("input", applyCmp);
  function applyCmp() {
    var v = parseFloat($("cmpSlider").value);
    $("viewTop").style.clipPath = "inset(0 0 0 " + v + "%)";
    var line = $("cmpLine");
    if (line) line.style.left = v + "%";
  }

  /* ---------- zoom / pan ---------- */
  function applyZoom() {
    $("zoomable").style.transform = "translate(" + state.pan[0] + "px," + state.pan[1] + "px) scale(" + state.zoom + ")";
  }
  function resetZoom() { state.zoom = 1; state.pan = [0, 0]; applyZoom(); }
  $("zin").addEventListener("click", function () { state.zoom = Math.min(4, state.zoom * 1.25); applyZoom(); });
  $("zout").addEventListener("click", function () { state.zoom = Math.max(1, state.zoom / 1.25); if (state.zoom === 1) state.pan = [0, 0]; applyZoom(); });
  $("zreset").addEventListener("click", resetZoom);
  (function () {
    var drag = null, el = $("viewport");
    el.addEventListener("pointerdown", function (e) {
      if (state.zoom === 1 || e.target.closest("button,input")) return;
      drag = [e.clientX, e.clientY]; el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", function (e) {
      if (!drag) return;
      state.pan[0] += e.clientX - drag[0]; state.pan[1] += e.clientY - drag[1];
      drag = [e.clientX, e.clientY]; applyZoom();
    });
    ["pointerup", "pointercancel"].forEach(function (ev) { el.addEventListener(ev, function () { drag = null; }); });
  })();

  /* ---------- pipeline hover ---------- */
  var pipeDesc = $("pipeDesc");
  document.querySelectorAll("#pipeList li").forEach(function (li) {
    function show() { if (pipeDesc) pipeDesc.textContent = li.dataset.desc; }
    li.addEventListener("pointerenter", show);
    li.addEventListener("focus", show);
  });

  /* ---------- stages ---------- */
  var STAGE_ORDER = ["preparing", "normalizing", "segmenting", "extent"];
  function setStages(current) {
    var box = $("stageSteps");
    if (!current) { box.hidden = true; return; }
    box.hidden = false;
    var idx = STAGE_ORDER.indexOf(current);
    Array.prototype.forEach.call(box.children, function (li) {
      var i = STAGE_ORDER.indexOf(li.dataset.stage);
      li.classList.toggle("doing", i === idx);
      li.classList.toggle("done", i < idx);
    });
  }

  /* ---------- run inference (SSE, real stages — never faked) ---------- */
  $("runBtn").addEventListener("click", run);
  $("retryBtn").addEventListener("click", run);
  function run() {
    if (!state.selected) return;
    var id = state.selected;
    $("runBtn").disabled = true;
    $("resultBody").hidden = true; $("resultError").hidden = true; $("resultEmpty").hidden = true;
    $("skeleton").hidden = false;
    setStages("preparing");
    setDeck("PROCESSING", "busy");
    fetch("/api/predict/" + id, { method: "POST" }).then(function (resp) {
      if (!resp.ok) throw new Error("bad");
      var rd = resp.body.getReader(), dec = new TextDecoder(), buf = "";
      function pump() {
        return rd.read().then(function (r) {
          buf += dec.decode(r.value || new Uint8Array(), { stream: !r.done });
          var parts = buf.split("\n\n"); buf = parts.pop();
          parts.forEach(function (p) {
            var line = p.replace(/^data:\s*/, "");
            if (!line.trim()) return;
            handleEvent(JSON.parse(line));
          });
          if (r.done) return;
          return pump();
        });
      }
      return pump();
    }).catch(function () { showError("Flood analysis couldn't be completed.", "The server didn't respond. Check the backend and try again."); });
  }
  function handleEvent(ev) {
    if (ev.error) { showError("Flood analysis couldn't be completed.", ev.error); return; }
    if (ev.stage && ev.stage !== "done") {
      setStages(ev.stage);
      $("runState").innerHTML = "" + ev.label + "…<span class='bar'><i></i></span>";
      setDeck("PROCESSING · " + ev.label.toUpperCase(), "busy");
    }
    if (ev.stage === "done") onResult(ev.result);
  }
  function onResult(res) {
    state.result = res;
    $("skeleton").hidden = true;
    $("runBtn").disabled = false;
    setStages(null);
    var box = $("stageSteps"); box.hidden = false;
    Array.prototype.forEach.call(box.children, function (li) { li.classList.remove("doing"); li.classList.add("done"); });
    setDeck("COMPLETE", "ok");
    $("runState").textContent = res.scene + " analyzed — live M1 inference.";
    var t = "?t=" + Date.now();
    document.querySelectorAll(".scene").forEach(function (el) {
      if (el.dataset.scene === res.scene) el.querySelector("img").src = layerURL("vv") + t;
    });
    refreshViewBusted(t);
    if (state.mode === "mask" || state.mode === "compare") {
      var top = state.mode === "compare" ? $("viewTop") : $("viewMain");
      top.style.transition = "none"; top.style.opacity = 0; top.style.filter = "blur(8px)";
      requestAnimationFrame(function () {
        top.style.transition = ""; top.style.opacity = ""; top.style.filter = "";
      });
    }
    $("resultEmpty").hidden = true; $("resultBody").hidden = false;
    var area = $("rArea");
    area.classList.remove("reveal-in"); void area.offsetWidth; area.classList.add("reveal-in");
    $("rScene").textContent = res.scene + " · T-SAO M1 · thr 0.50";
    countUp(area, res.flood_area_km2, 2);
    $("rCov").textContent = (res.flood_fraction * 100).toFixed(2) + " %";
    $("rFlood").textContent = fmt(res.flooded_pixels);
    $("rValid").textContent = fmt(res.valid_pixels);
    if (res.benchmark) {
      $("benchBox").hidden = false;
      $("bIou").textContent = res.benchmark.iou.toFixed(4);
      $("bF1").textContent = res.benchmark.f1.toFixed(4);
      $("bP").textContent = res.benchmark.precision.toFixed(4);
      $("bR").textContent = res.benchmark.recall.toFixed(4);
    } else { $("benchBox").hidden = true; }
    $("dlTif").href = res.export_geotiff;
    setLegend();
  }
  function refreshViewBusted(t) {
    var m = state.mode;
    var map = { source: state.src, prob: "prob", mask: "mask", gt: "gt" };
    if (map[m]) showMain(layerURL(map[m]) + t, "Analysis result");
    if (m === "compare") { showMain(layerURL("vv") + t, "Source"); $("viewTop").src = layerURL("mask") + t; applyCmp(); }
  }
  function showError(title, msg) {
    $("skeleton").hidden = true; $("runBtn").disabled = false;
    setStages(null); setDeck("ERROR", "idle");
    $("resultBody").hidden = true; $("resultEmpty").hidden = true;
    $("resultError").hidden = false;
    $("errTitle").textContent = title; $("errMsg").textContent = msg || "";
    $("runState").textContent = "Something went wrong — you can try again.";
  }
  function countUp(el, target, dec) {
    if (RM) { el.textContent = target.toFixed(dec); return; }
    var t0 = performance.now(), dur = 1100;
    function tick(t) {
      var k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = (target * e).toFixed(dec);
      if (k < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ---------- export ---------- */
  $("dlPng").addEventListener("click", function () {
    var src = $("viewMain").src;
    fetch(src).then(function (r) { return r.blob(); }).then(function (b) {
      var a = document.createElement("a");
      a.href = URL.createObjectURL(b);
      a.download = state.selected + "_" + state.mode + ".png";
      document.body.appendChild(a); a.click(); a.remove();
      $("dlPng").textContent = "Saved ✓";
      setTimeout(function () { $("dlPng").textContent = "Save view"; }, 1600);
    });
  });

  moveInk();
})();
