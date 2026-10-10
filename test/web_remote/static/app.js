/* AutoDoor Test Remote - vanilla JS (TEST ONLY) */
(function () {
  "use strict";

  var POLL_MS = 300;
  var MAX_CHANNEL = 20;

  // ------------------------------------------------------------------
  // helpers
  // ------------------------------------------------------------------

  function $(id) { return document.getElementById(id); }

  function api(path, method, body) {
    var opts = { method: method || "GET", headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(function (r) { return r.json(); });
  }

  var toastTimer = null;
  function toast(message, bad) {
    var el = $("toast");
    if (!el) return;
    el.textContent = message;
    el.className = "toast" + (bad ? " bad" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.className = "toast hidden"; }, 2500);
  }

  function setText(id, value) {
    var el = $(id);
    if (!el) return;
    el.textContent = (value === null || value === undefined || value === "") ? "-" : String(value);
  }

  function fmt(value, digits) {
    if (value === null || value === undefined) return "-";
    if (typeof value === "number") return value.toFixed(digits === undefined ? 4 : digits);
    return String(value);
  }

  function setStatusBox(id, text, cls) {
    var el = $(id);
    if (!el) return;
    el.textContent = text;
    el.className = "big-status" + (cls ? " " + cls : "");
  }

  function esc(s) {
    return String(s === null || s === undefined ? "-" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // quality checks table: one row per FaceQualityGate condition
  function checksRowsHtml(checks) {
    if (!checks || !checks.length) return "";
    return checks.map(function (c) {
      var res = c.ok ? "<span class='check-ok'>PASS</span>" : "<span class='check-fail'>FAIL</span>";
      var val = c.value === null ? (c.note ? esc(c.note) : "-") : fmt(c.value, c.metric === "sharpness" || c.metric.indexOf("brightness") === 0 ? 2 : 4);
      var thr = c.threshold === null ? "-" : fmt(c.threshold, c.metric === "sharpness" || c.metric.indexOf("brightness") === 0 ? 2 : 4);
      return "<tr><td>" + esc(c.metric) + "</td><td>" + val + "</td><td>" + esc(c.op) + "</td><td>" + thr + "</td><td>" + res + "</td></tr>";
    }).join("");
  }

  function renderChecks(tableId, checks) {
    var t = $(tableId);
    if (!t) return;
    var head = "<tr><th>metric</th><th>value</th><th>op</th><th>threshold</th><th>result</th></tr>";
    t.innerHTML = head + (checks ? checksRowsHtml(checks) : "<tr><td colspan='5' class='muted'>no single face</td></tr>");
  }

  var THRESHOLD_KEYS = ["YAW_THRESHOLD", "DOWN_THRESHOLD", "FAR_FACE_RATIO", "FACE_MIN_CONFIDENCE",
    "FACE_MIN_RATIO", "FACE_MIN_SHARPNESS", "FACE_MIN_BRIGHTNESS", "FACE_MAX_BRIGHTNESS"];

  function renderLoadedThresholds(sys) {
    var t = $("loaded-thresholds");
    if (!t || !sys.loaded_thresholds) return;
    t.innerHTML = "<tr><th>key</th><th>loaded</th></tr>" + THRESHOLD_KEYS.map(function (k) {
      return "<tr><td>" + k + "</td><td>" + fmt(sys.loaded_thresholds[k], 4) + "</td></tr>";
    }).join("");
    setText("thr-loaded-at", sys.thresholds_loaded_at);
  }

  function failReasons(checks) {
    if (!checks) return "";
    return checks.filter(function (c) { return !c.ok; }).map(function (c) { return c.metric; }).join(", ");
  }

  // ------------------------------------------------------------------
  // video stream
  // ------------------------------------------------------------------

  function refreshStream() {
    var img = $("video");
    if (!img) return;
    img.src = "/api/video/stream?t=" + Date.now();
  }

  // ------------------------------------------------------------------
  // system badge + camera selector (shared by all pages)
  // ------------------------------------------------------------------

  var cameraLocked = false;

  function applySystem(sys) {
    if (!sys) return;
    var modeEl = $("sys-mode");
    if (modeEl) {
      modeEl.textContent = sys.mode;
      modeEl.className = "badge badge-mode" + (sys.mode !== "IDLE" ? " busy" : "");
    }
    var camEl = $("sys-cam");
    if (camEl) {
      camEl.textContent = sys.source === "RTSP" ? "CH " + sys.channel : sys.source;
    }
    cameraLocked = !!sys.camera_locked;
    markActiveCamera(sys.source, sys.channel);
    var stale = $("env-stale");
    if (stale) {
      stale.classList.toggle("hidden", !sys.thresholds_reload_error);
      if (sys.thresholds_reload_error) stale.textContent = ".env reload failed: " + sys.thresholds_reload_error;
    }
    renderLoadedThresholds(sys);
    var errEl = $("camera-error");
    if (errEl) errEl.textContent = sys.camera_error || "";
  }

  function markActiveCamera(source, channel) {
    var buttons = document.querySelectorAll(".cam-btn");
    for (var i = 0; i < buttons.length; i++) {
      var b = buttons[i];
      var isActive = (b.dataset.kind === "laptop" && source === "LAPTOP") ||
                     (b.dataset.kind === "rtsp" && source === "RTSP" && Number(b.dataset.channel) === Number(channel));
      b.classList.toggle("active", isActive);
      b.disabled = cameraLocked;
    }
  }

  function buildCameraSelector() {
    var box = $("camera-selector");
    if (!box) return;
    box.innerHTML = "";

    var laptop = document.createElement("button");
    laptop.className = "cam-btn laptop";
    laptop.textContent = "LAPTOP";
    laptop.dataset.kind = "laptop";
    laptop.onclick = function () { selectCamera("/api/camera/laptop"); };
    box.appendChild(laptop);

    for (var ch = 1; ch <= MAX_CHANNEL; ch++) {
      var b = document.createElement("button");
      b.className = "cam-btn";
      b.textContent = String(ch);
      b.dataset.kind = "rtsp";
      b.dataset.channel = String(ch);
      b.onclick = (function (c) {
        return function () { selectCamera("/api/camera/channel/" + c); };
      })(ch);
      box.appendChild(b);
    }
  }

  var switching = false;
  function selectCamera(path) {
    if (switching) { toast("Switching camera, please wait"); return; }
    if (cameraLocked) { toast("Camera locked while mode is running", true); return; }
    switching = true;
    toast("Switching camera...");
    api(path, "POST").then(function (res) {
      switching = false;
      toast(res.message || (res.ok ? "OK" : "Failed"), !res.ok);
      applySystem(res);
      refreshStream();
    }).catch(function (e) {
      switching = false;
      toast("Request failed: " + e, true);
    });
  }

  // ------------------------------------------------------------------
  // page: index
  // ------------------------------------------------------------------

  function initIndex() {
    function poll() {
      api("/api/status").then(applySystem).catch(function () {});
    }
    poll();
    setInterval(poll, 1000);
  }

  // ------------------------------------------------------------------
  // page: calibration
  // ------------------------------------------------------------------

  function initCalibration() {
    var stageListEl = $("stage-list");
    var lastPhase = null;

    $("btn-cal-start").onclick = function () {
      api("/api/calibration/start", "POST").then(function (res) {
        toast(res.message, !res.ok);
        $("cal-result").hidden = true;
        setStatusBox("cal-save-status", "", "");
      });
    };
    $("btn-cal-ready").onclick = function () {
      api("/api/calibration/ready", "POST").then(function (res) { if (!res.ok) toast(res.message, true); });
    };
    $("btn-cal-cancel").onclick = function () {
      api("/api/calibration/cancel", "POST").then(function (res) { toast(res.message, !res.ok); });
    };
    var lastThresholds = null;

    function readSaveInputs() {
      var values = {};
      var inputs = document.querySelectorAll("#cal-thresholds input[data-key]");
      for (var i = 0; i < inputs.length; i++) {
        var v = parseFloat(inputs[i].value);
        if (isNaN(v)) { toast("Invalid number for " + inputs[i].dataset.key, true); return null; }
        values[inputs[i].dataset.key] = v;
      }
      return values;
    }

    function fillSaveInputs(source) {
      if (!source) return;
      var inputs = document.querySelectorAll("#cal-thresholds input[data-key]");
      for (var i = 0; i < inputs.length; i++) {
        inputs[i].value = Number(source[inputs[i].dataset.key]).toFixed(4);
      }
    }

    function renderThresholdTable(st) {
      var t = $("cal-thresholds");
      var html = "<tr><th>key</th><th>strict</th><th>relaxed</th><th>save</th></tr>";
      THRESHOLD_KEYS.forEach(function (k) {
        html += "<tr><td>" + k + "</td><td>" + fmt(st.thresholds[k], 4) + "</td><td>" + fmt(st.thresholds_relaxed[k], 4) +
          "</td><td><input class='thr-input' type='number' step='0.0001' inputmode='decimal' data-key='" + k +
          "' value='" + Number(st.thresholds_relaxed[k]).toFixed(4) + "'></td></tr>";
      });
      t.innerHTML = html;
    }

    $("btn-cal-fill-strict").onclick = function () { fillSaveInputs(lastThresholds && lastThresholds.thresholds); };
    $("btn-cal-fill-relaxed").onclick = function () { fillSaveInputs(lastThresholds && lastThresholds.thresholds_relaxed); };

    $("btn-cal-save").onclick = function () {
      var values = readSaveInputs();
      if (!values) return;
      if (!confirm("Write these thresholds to .env and apply immediately?")) return;
      api("/api/calibration/save", "POST", { values: values }).then(function (res) {
        toast(res.message, !res.ok);
        setStatusBox("cal-save-status", res.message, res.ok ? "ok" : "bad");
      });
    };

    $("btn-thr-reload").onclick = function () {
      api("/api/thresholds/reload", "POST").then(function (res) { toast(res.message, !res.ok); });
    };

    function renderStages(st) {
      stageListEl.innerHTML = "";
      st.stages.forEach(function (name, i) {
        var chip = document.createElement("span");
        chip.className = "chip" + (i < st.stage_index ? " done" : (i === st.stage_index && st.active ? " current" : ""));
        chip.textContent = name;
        stageListEl.appendChild(chip);
      });
    }

    function renderTable(tableId, rows, header) {
      var t = $(tableId);
      var html = "";
      if (header) {
        html += "<tr>" + header.map(function (h) { return "<th>" + h + "</th>"; }).join("") + "</tr>";
      }
      rows.forEach(function (r) {
        html += "<tr>" + r.map(function (c, i) { return i === 0 ? "<td>" + c + "</td>" : "<td>" + c + "</td>"; }).join("") + "</tr>";
      });
      t.innerHTML = html;
    }

    function render(data) {
      applySystem(data.system);
      var st = data.calibration;

      var running = !!(st && st.active);
      $("btn-cal-start").disabled = running || data.system.mode !== "IDLE";
      $("btn-cal-ready").disabled = !(running && st.phase === "WAIT");
      $("btn-cal-cancel").disabled = !running;

      if (!st) {
        setStatusBox("cal-message", "Press START CALIBRATION", "");
        return;
      }

      renderStages(st);
      setText("cal-stage", st.stage);
      setText("cal-phase", st.phase);
      setText("cal-countdown", st.countdown !== null ? st.countdown.toFixed(1) + "s" : "-");
      setText("cal-samples", st.samples + " / " + st.samples_per_stage);
      setText("cal-faces", st.face_count);
      setText("cal-guide", st.guide);

      var live = $("cal-live");
      var keys = ["yaw", "pitch", "face_ratio", "confidence", "sharpness", "brightness"];
      keys.forEach(function (k) {
        var el = live.querySelector("[data-k='" + k + "']");
        if (el) el.textContent = st.live_metrics ? fmt(st.live_metrics[k], (k === "sharpness" || k === "brightness") ? 2 : 4) : "-";
      });

      if (st.phase === "WAIT") {
        setStatusBox("cal-message", st.stage + ": " + st.guide + " — press READY", "warn");
      } else if (st.phase === "PREPARE") {
        setStatusBox("cal-message", "GET READY " + st.stage + ": " + st.countdown.toFixed(1) + "s", "warn");
      } else if (st.phase === "CAPTURE") {
        setStatusBox("cal-message", "CAPTURING " + st.stage + "  " + st.samples + " / " + st.samples_per_stage +
          (st.face_count !== 1 ? "  (need exactly 1 face)" : ""), "ok");
      } else if (st.phase === "DONE") {
        setStatusBox("cal-message", "CALIBRATION COMPLETE" + (st.saved ? " (saved)" : " - not saved yet"), "ok");
      } else if (st.phase === "ERROR") {
        setStatusBox("cal-message", "ERROR: " + st.error, "bad");
      } else if (st.phase === "CANCELLED") {
        setStatusBox("cal-message", "CANCELLED", "bad");
      }

      if (st.phase === "DONE" && st.medians && st.thresholds) {
        if (lastPhase !== "DONE") {
          var mkeys = ["yaw", "pitch", "face_ratio", "confidence", "sharpness", "brightness"];
          var rows = st.stages.map(function (s) {
            var m = st.medians[s];
            return [s].concat(mkeys.map(function (k) { return fmt(m[k], (k === "sharpness" || k === "brightness") ? 2 : 4); }));
          });
          renderTable("cal-medians", rows, ["stage"].concat(mkeys));
          renderThresholdTable(st);
          lastThresholds = { thresholds: st.thresholds, thresholds_relaxed: st.thresholds_relaxed };
          setText("cal-env-path", st.env_path);
        }
        $("cal-result").hidden = false;
        $("btn-cal-save").disabled = false;
      }
      lastPhase = st.phase;
    }

    function poll() {
      api("/api/calibration/status").then(render).catch(function () {});
    }
    poll();
    setInterval(poll, POLL_MS);
  }

  // ------------------------------------------------------------------
  // page: enrollment
  // ------------------------------------------------------------------

  function initEnrollment() {
    var posesEl = $("enr-poses");

    $("btn-enr-start").onclick = function () {
      var name = $("enr-name").value.trim();
      var room = $("enr-room").value.trim();
      if (!name) { toast("Họ tên không được để trống", true); return; }
      $("enr-result").hidden = true;
      api("/api/enrollment/start", "POST", { full_name: name, room_number: room || null })
        .then(function (res) { toast(res.message, !res.ok); });
    };
    $("btn-enr-cancel").onclick = function () {
      api("/api/enrollment/cancel", "POST").then(function (res) { toast(res.message, !res.ok); });
    };

    function renderPoses(st) {
      posesEl.innerHTML = "";
      st.required.forEach(function (p) {
        var chip = document.createElement("span");
        var done = st.collected.indexOf(p) >= 0;
        chip.className = "chip" + (done ? " done" : (p === st.next_pose && st.active ? " current" : ""));
        chip.textContent = p;
        posesEl.appendChild(chip);
      });
    }

    function render(data) {
      applySystem(data.system);
      var st = data.enrollment;
      var running = !!(st && st.active);

      $("btn-enr-start").disabled = running || data.system.mode !== "IDLE";
      $("btn-enr-cancel").disabled = !running;
      $("enr-name").disabled = running;
      $("enr-room").disabled = running;

      if (!st) {
        setStatusBox("enr-message", "Fill the form, pick a camera, press START", "");
        return;
      }

      setText("enr-s-name", st.name);
      setText("enr-s-room", st.room);
      setText("enr-s-source", st.camera ? st.camera.source : "-");
      setText("enr-s-channel", st.camera ? st.camera.channel : "-");
      renderPoses(st);
      setText("enr-next", st.next_pose);
      setText("enr-progress", st.progress + " / " + st.total);
      setText("enr-collected", st.collected.length ? st.collected.join(", ") : "-");

      var dbg = $("enr-debug");
      var d = st.debug || {};
      var q = d.quality;
      function setK(k, v) { var el = dbg.querySelector("[data-k='" + k + "']"); if (el) el.textContent = v; }
      setK("face_count", d.face_count === undefined ? "-" : d.face_count);
      setK("detected_pose", d.detected_pose || "-");
      setK("passed", q ? (q.passed ? "PASS" : "FAIL") : "-");
      setK("confidence", q ? fmt(q.confidence, 4) : "-");
      setK("face_ratio", q ? fmt(q.face_ratio, 4) : "-");
      setK("sharpness", q ? fmt(q.sharpness, 2) : "-");
      setK("brightness", q ? fmt(q.brightness, 2) : "-");
      renderChecks("enr-checks", q ? q.checks : null);
      setText("enr-started", data.system.thresholds_loaded_at);

      if (st.state === "COOLDOWN") {
        setStatusBox("enr-message", "GET READY: " + st.cooldown.toFixed(1) + "s  → " + st.next_pose, "warn");
      } else if (st.state === "CAPTURING") {
        setStatusBox("enr-message", "CAPTURING " + st.next_pose, "ok");
      } else if (st.state === "SUCCESS") {
        setStatusBox("enr-message", "ENROLLMENT SUCCESS", "ok");
      } else if (st.state === "ERROR") {
        setStatusBox("enr-message", "ERROR: " + st.error, "bad");
      } else if (st.state === "CANCELLED") {
        setStatusBox("enr-message", "CANCELLED", "bad");
      }

      if (st.state === "SUCCESS" && st.result) {
        $("enr-result").hidden = false;
        setText("enr-r-id", st.result.id);
        setText("enr-r-code", st.result.code);
        setText("enr-r-name", st.result.name);
        setText("enr-r-room", st.result.room);
      }
    }

    function poll() {
      api("/api/enrollment/status").then(render).catch(function () {});
    }
    poll();
    setInterval(poll, POLL_MS);
  }

  // ------------------------------------------------------------------
  // page: recognition
  // ------------------------------------------------------------------

  function initRecognition() {
    var listEl = $("rec-face-list");

    $("btn-rec-start").onclick = function () {
      api("/api/recognition/start", "POST").then(function (res) { toast(res.message, !res.ok); });
    };
    $("btn-rec-stop").onclick = function () {
      api("/api/recognition/stop", "POST").then(function (res) { toast(res.message, !res.ok); });
    };

    function kv(label, value) {
      return "<div class='kv'><span>" + label + "</span><b>" + value + "</b></div>";
    }

    function renderFace(f) {
      var v = f.verdict.toLowerCase();
      var html = "<section class='card face-card " + v + "'>";
      html += "<div class='face-head'><b>Face #" + f.index + "</b><span class='verdict " + v + "'>" + f.verdict + "</span></div>";

      var c = f.candidate;
      if (f.verdict === "RECOGNIZED") {
        html += "<div class='identity recognized'>" + esc(c.name) + "</div>";
        html += "<div class='kv-grid'>" +
          kv("Member code", esc(c.code)) + kv("Room", esc(c.room)) +
          kv("Best template pose", esc(c.pose)) + kv("SFace similarity", fmt(c.similarity, 4)) + "</div>";
      } else if (f.verdict === "REJECTED") {
        html += "<div class='identity rejected'>REJECTED (quality FAIL)</div>";
        html += "<div class='kv-grid'>" +
          kv("Candidate", c ? esc(c.name) : "-") + kv("Similarity", c ? fmt(c.similarity, 4) : "-") +
          kv("Candidate code", c ? esc(c.code) : "-") + kv("Candidate pose", c ? esc(c.pose) : "-") + "</div>";
      } else if (f.verdict === "UNKNOWN") {
        html += "<div class='identity unknown'>UNKNOWN</div>";
        html += "<div class='kv-grid'>" +
          kv("Best candidate", c ? esc(c.name) : "-") + kv("Similarity", c ? fmt(c.similarity, 4) : "-") +
          kv("Candidate code", c ? esc(c.code) : "-") + kv("Candidate pose", c ? esc(c.pose) : "-") + "</div>";
      } else {
        html += "<div class='identity rejected'>NO EMBEDDING" + (f.sface_error ? ": " + esc(f.sface_error) : "") + "</div>";
      }

      html += "<div class='kv-grid'>" +
        kv("YuNet confidence", fmt(f.yunet_confidence, 4)) +
        kv("BBox", f.bbox.width + " x " + f.bbox.height) +
        kv("Face ratio", fmt(f.face_ratio, 4)) +
        kv("Sharpness", fmt(f.sharpness, 2)) +
        kv("Brightness", fmt(f.brightness, 2)) +
        kv("Quality", f.quality_passed ? "<span class='ok-text'>PASS</span>" : "<span class='bad-text'>FAIL: " + esc(failReasons(f.quality_checks)) + "</span>") +
        kv("Embedding dim", f.embedding_dim) +
        kv("Embedding norm", fmt(f.embedding_norm, 4)) +
        kv("SFace time", fmt(f.sface_ms, 2) + " ms") +
        "</div>" +
        (f.quality_passed ? "" : "<table class='table'>" + checksRowsHtml(f.quality_checks) + "</table>") +
        "</section>";
      return html;
    }

    function render(data) {
      applySystem(data.system);
      var st = data.recognition;
      var running = !!(st && st.active);

      $("btn-rec-start").disabled = running || data.system.mode !== "IDLE";
      $("btn-rec-stop").disabled = !running;

      setText("rec-source", data.system.source);
      setText("rec-channel", data.system.channel);
      setText("rec-fps", fmt(data.system.fps, 1));
      setText("rec-frame", data.system.frame_size ? data.system.frame_size.join(" x ") : "-");

      if (!st) {
        setStatusBox("rec-message", "Press START RECOGNITION", "");
        listEl.innerHTML = "";
        return;
      }

      setText("rec-faces", st.face_count);
      setText("rec-yunet", fmt(st.yunet_ms, 2) + " ms");
      setText("rec-db", st.db_embedding_count + " (" + st.db_member_count + " members)");
      setText("rec-threshold", fmt(st.match_threshold, 2));

      if (st.error) {
        setStatusBox("rec-message", "ERROR: " + st.error, "bad");
      } else if (running) {
        setStatusBox("rec-message", "RUNNING - " + st.face_count + " face(s)", "ok");
      } else {
        setStatusBox("rec-message", "STOPPED", "");
      }

      if (running) {
        listEl.innerHTML = st.faces.map(renderFace).join("");
      }
    }

    function poll() {
      api("/api/recognition/status").then(render).catch(function () {});
    }
    poll();
    setInterval(poll, POLL_MS);
  }

  // ------------------------------------------------------------------
  // boot
  // ------------------------------------------------------------------

  document.addEventListener("DOMContentLoaded", function () {
    buildCameraSelector();
    refreshStream();

    var page = document.body.dataset.page;
    if (page === "calibration") initCalibration();
    else if (page === "enrollment") initEnrollment();
    else if (page === "recognition") initRecognition();
    else initIndex();
  });
})();
