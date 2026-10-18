/* 2MP — живой 3D-куб логотипа для сайта (CSS 3D + SVG, без библиотек).
 *
 * Подключение:  <div data-cube-loop></div>  +  <script src="/static/cube-loop.js" defer></script>
 * Атрибуты: data-theme="dark|light", data-static (без движения).
 *
 * Цикл 10 с: парит и покачивается → полный оборот вокруг оси с «доводкой» → блик по граням →
 * грани разлетаются в стороны и собираются с щелчком → кивок. Начало и конец цикла совпадают.
 * Мышь: куб следит за курсором, раскрывается при наведении, крутится при перетаскивании (с инерцией),
 * «взрывается» по клику. На скрытых гранях — объектив и рамка видоискателя,
 * их видно во время оборота. Пауза, когда блок не виден; prefers-reduced-motion — статичный куб.
 */
(function () {
  "use strict";
  var PERIOD = 10;
  var ISO_X = -35.264, ISO_Y = -45;          // изометрический ракурс логотипа
  var G = 7, SW = 11, A = G / 2, B = 100 - G / 2;
  var THEMES = {
    dark: { front: "#E4E0D6", right: "#2BB3A3", top: "#F2EFE8", back: "#F2EFE8", left: "#2BB3A3", bottom: "#C9C4B8",
            glow: "rgba(43,179,163,.34)", shadow: "rgba(0,0,0,.45)" },
    light: { front: "#13202B", right: "#1E9A8C", top: "#2A3A48", back: "#2A3A48", left: "#1E9A8C", bottom: "#0D1620",
             glow: "rgba(43,179,163,.22)", shadow: "rgba(13,22,32,.25)" },
    violet: { front: "#6E5BFF", right: "#4F7BFF", top: "#9C8BFF", back: "#6E5BFF", left: "#4F7BFF", bottom: "#3F4FD0",
            glow: "rgba(120,100,255,.32)", shadow: "rgba(70,60,180,.28)" },
    teal: { front: "#2AA898", right: "#1E8C80", top: "#45C2B2", back: "#2AA898", left: "#1E8C80", bottom: "#17736A",
            glow: "rgba(43,179,163,.26)", shadow: "rgba(90,70,40,.28)" }
  };

  function rrect(x1, y1, x2, y2, r) {
    var tl = r[0], tr = r[1], br = r[2], bl = r[3];
    return "M" + (x1 + tl) + " " + y1 + " L" + (x2 - tr) + " " + y1 + " Q" + x2 + " " + y1 + " " + x2 + " " + (y1 + tr) +
      " L" + x2 + " " + (y2 - br) + " Q" + x2 + " " + y2 + " " + (x2 - br) + " " + y2 + " L" + (x1 + bl) + " " + y2 +
      " Q" + x1 + " " + y2 + " " + x1 + " " + (y2 - bl) + " L" + x1 + " " + (y1 + tl) + " Q" + x1 + " " + y1 + " " + (x1 + tl) + " " + y1 + " Z";
  }
  function slot(x1, y1, x2, y2) {
    return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" stroke="#000" stroke-width="' + SW + '" stroke-linecap="round"/>';
  }
  // содержимое граней (вырезы в маске), координаты грани 0..100
  var CUTS = {
    two: { r: [10, 34, 6, 6], cuts: slot(-10, 38, 58, 38) + slot(42, 68, 110, 68) },
    p: { r: [6, 34, 6, 6], cuts: slot(46, 32, 60, 32) + '<path d="' + rrect(46, 62, 120, 120, [12, 0, 0, 0]) + '" fill="#000"/>' },
    m: { r: [8, 8, 8, 8], cuts: slot(30, 110, 30, 42) + slot(70, 110, 70, 42) + '<path d="M36 -10 L64 -10 L52 46 Q50 52 48 46 Z" fill="#000"/>' },
    lens: { r: [8, 8, 8, 8], cuts: '<circle cx="50" cy="50" r="27" fill="#000"/>', extra: function (c) {
      return '<circle cx="50" cy="50" r="17" fill="' + c + '"/><circle cx="50" cy="50" r="7" fill="#0D1620"/><circle cx="56" cy="44" r="3.2" fill="#F2EFE8"/>'; } },
    finder: { r: [8, 8, 8, 8], cuts: slot(24, 40, 24, 24) + slot(24, 24, 40, 24) + slot(76, 60, 76, 76) + slot(60, 76, 76, 76), extra: function () {
      return '<circle cx="74" cy="26" r="6" fill="#E8533D"/>'; } },
    plain: { r: [8, 8, 8, 8], cuts: "" }
  };
  // грани куба: какая буква, цвет, базовая ориентация (CSS), направление «разлёта»
  var FACES = [
    { k: "two", col: "front", tr: "rotateY(0deg)" },
    { k: "p", col: "right", tr: "rotateY(90deg)" },
    { k: "lens", col: "back", tr: "rotateY(180deg)" },
    { k: "finder", col: "left", tr: "rotateY(-90deg)" },
    { k: "m", col: "top", tr: "rotateX(90deg) rotateZ(-90deg)" },
    { k: "plain", col: "bottom", tr: "rotateX(-90deg)" }
  ];

  var clamp = function (x) { return x < 0 ? 0 : x > 1 ? 1 : x; };
  var seg = function (t, a, b) { return clamp((t - a) / (b - a)); };
  var eio = function (x) { return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
  var backOut = function (x) { var c1 = 1.7, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };
  var spinEase = function (x) {                 // разгон → оборот → лёгкий перелёт и возврат
    var c = 1.2;
    return x < 0.5 ? (Math.pow(2 * x, 2) * ((c + 1) * 2 * x - c)) / 2 : (Math.pow(2 * x - 2, 2) * ((c + 1) * (x * 2 - 2) + c) + 2) / 2;
  };

  var cssDone = false;
  function injectCss() {
    if (cssDone) return;
    cssDone = true;
    var s = document.createElement("style");
    s.textContent =
      ".c2-wrap{position:relative;width:100%;aspect-ratio:1.25;display:flex;align-items:center;justify-content:center;perspective:1100px;user-select:none}" +
      ".c2-glow{position:absolute;inset:6% 12%;border-radius:50%;filter:blur(30px);pointer-events:none}" +
      ".c2-shadow{position:absolute;left:50%;bottom:9%;width:46%;height:7%;margin-left:-23%;border-radius:50%;filter:blur(10px);pointer-events:none}" +
      ".c2-cube{position:relative;width:var(--s);height:var(--s);transform-style:preserve-3d;will-change:transform}" +
      ".c2-face{position:absolute;inset:0;backface-visibility:hidden;-webkit-backface-visibility:hidden;will-change:transform}" +
      ".c2-face svg{width:100%;height:100%;display:block;overflow:visible}" +
      ".c2-glare{mix-blend-mode:screen}";
    document.head.appendChild(s);
  }

  var uidN = 0;
  function faceSvg(f, theme, uid) {
    var d = CUTS[f.k], col = theme[f.col];
    return '<svg viewBox="0 0 100 100" aria-hidden="true"><defs>' +
      '<mask id="' + uid + '" maskUnits="userSpaceOnUse" x="-30" y="-30" width="160" height="160"><path d="' + rrect(A, A, B, B, d.r) + '" fill="#fff"/>' + d.cuts + "</mask>" +
      '<linearGradient id="' + uid + 'g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/>' +
      '<stop offset=".5" stop-color="#fff" stop-opacity=".55"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient></defs>' +
      '<rect x="-30" y="-30" width="160" height="160" fill="' + col + '" mask="url(#' + uid + ')"/>' +
      (d.extra ? d.extra(theme.right) : "") +
      '<g mask="url(#' + uid + ')"><rect class="c2-glare" x="-160" y="-60" width="70" height="220" fill="url(#' + uid + 'g)" transform="rotate(25)"/></g>' +
      '<rect class="c2-shade" x="-30" y="-30" width="160" height="160" fill="#000" opacity="0" mask="url(#' + uid + ')"/></svg>';
  }

  function mount(host) {
    if (host.__cube) return;
    injectCss();
    var theme = THEMES[host.getAttribute("data-theme") || "dark"] || THEMES.dark;
    host.innerHTML = "";
    var wrap = document.createElement("div");
    wrap.className = "c2-wrap";
    wrap.setAttribute("role", "img");
    wrap.setAttribute("aria-label", "Логотип 2MP");
    var glow = document.createElement("div"); glow.className = "c2-glow";
    glow.style.background = "radial-gradient(closest-side," + theme.glow + ",transparent)";
    var shadow = document.createElement("div"); shadow.className = "c2-shadow";
    shadow.style.background = "radial-gradient(closest-side," + theme.shadow + ",transparent)";
    var cube = document.createElement("div"); cube.className = "c2-cube";
    wrap.appendChild(glow); wrap.appendChild(shadow); wrap.appendChild(cube); host.appendChild(wrap);

    var faces = FACES.map(function (f) {
      var el = document.createElement("div");
      el.className = "c2-face";
      el.innerHTML = faceSvg(f, theme, "c2m" + (uidN++));
      cube.appendChild(el);
      return { el: el, tr: f.tr, glare: el.querySelector(".c2-glare"), shade: el.querySelector(".c2-shade") };
    });

    var size = 0;
    function resize() {
      size = Math.max(14, Math.min(wrap.clientWidth * 0.36, wrap.clientHeight * 0.46));
      cube.style.setProperty("--s", size + "px");
    }
    resize();
    if ("ResizeObserver" in window) new ResizeObserver(resize).observe(wrap);

    // ---------------- взаимодействие с мышью ----------------
    // look   — куб следит за курсором по всей странице (как взгляд);
    // hover  — при наведении куб приподнимается, грани чуть раскрываются, свечение ярче;
    // drag   — потяни мышью: куб вращается за рукой, после отпускания крутится по инерции
    //          и мягко возвращается в позу логотипа;
    // click  — короткий «взрыв» граней и рывок вращения.
    var I = { lx: 0, ly: 0, tlx: 0, tly: 0, hover: 0, thover: 0, yaw: 0, pitch: 0, vyaw: 0, vpitch: 0,
              burst: 0, auto: 1, drag: false, idle: 0, moved: 0, last: null };
    wrap.style.cursor = "grab";
    wrap.style.touchAction = "pan-y";
    window.addEventListener("pointermove", function (e) {
      var r = cube.getBoundingClientRect();
      var cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      var span = Math.max(window.innerWidth, 600) * 0.45;
      I.tlx = Math.max(-1, Math.min(1, (e.clientX - cx) / span));
      I.tly = Math.max(-1, Math.min(1, (e.clientY - cy) / span));
      if (I.drag && I.last) {
        var dx = e.clientX - I.last[0], dy = e.clientY - I.last[1], dt = Math.max((e.timeStamp - I.last[2]) / 1000, 0.008);
        I.yaw += dx * 0.45; I.pitch = Math.max(-70, Math.min(70, I.pitch - dy * 0.35));
        I.vyaw = I.vyaw * 0.5 + (dx * 0.45 / dt) * 0.5; I.vpitch = I.vpitch * 0.5 + (-dy * 0.35 / dt) * 0.5;
        I.moved += Math.abs(dx) + Math.abs(dy);
        I.last = [e.clientX, e.clientY, e.timeStamp];
      }
    }, { passive: true });
    wrap.addEventListener("pointerenter", function () { I.thover = 1; kick(); });
    wrap.addEventListener("pointerleave", function () { I.thover = 0; });
    wrap.addEventListener("pointerdown", function (e) {
      I.drag = true; I.moved = 0; I.last = [e.clientX, e.clientY, e.timeStamp]; I.vyaw = 0; I.vpitch = 0;
      wrap.style.cursor = "grabbing";
      try { wrap.setPointerCapture(e.pointerId); } catch (err) { /* не критично */ }
      kick();
    });
    function release() {
      if (!I.drag) return;
      I.drag = false; wrap.style.cursor = "grab";
      if (I.moved < 6) { I.burst = 1; I.vyaw += 720; }                     // клик без перетаскивания
    }
    wrap.addEventListener("pointerup", release);
    wrap.addEventListener("pointercancel", release);

    function shade(rx, ry) {
      var ryr = ry * Math.PI / 180, rxr = rx * Math.PI / 180;
      var normals = [[0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0], [0, -1, 0], [0, 1, 0]];
      faces.forEach(function (f, i) {
        var n = normals[i];
        var x1 = n[0] * Math.cos(ryr) + n[2] * Math.sin(ryr), z1 = -n[0] * Math.sin(ryr) + n[2] * Math.cos(ryr), y1 = n[1];
        var y2 = y1 * Math.cos(rxr) - z1 * Math.sin(rxr), z2 = y1 * Math.sin(rxr) + z1 * Math.cos(rxr);
        var light = clamp(0.55 * z2 - 0.65 * y2 + 0.25 * x1);             // свет сверху-спереди-справа
        f.shade.setAttribute("opacity", (0.2 * (1 - light)).toFixed(3));
      });
    }

    // t — время автоматического цикла; I — состояние мыши (без него — чистый цикл, для записи превью)
    function render(t, st) {
      st = st || { lx: 0, ly: 0, hover: 0, yaw: 0, pitch: 0, burst: 0, auto: 1 };
      t = ((t % PERIOD) + PERIOD) % PERIOD;
      var w = st.auto;                                                       // вес автоматической хореографии
      var float = Math.sin(t / PERIOD * Math.PI * 4) * 0.035 * w - st.hover * 0.05;
      var sway = Math.sin(t / PERIOD * Math.PI * 2) * 7 * w;
      var spin = spinEase(seg(t, 2.0, 3.6)) * 360 * w;
      var nod = Math.sin(seg(t, 7.4, 8.6) * Math.PI) * 14 * w;
      var apart = eio(seg(t, 5.2, 5.9)) * (1 - backOut(seg(t, 6.1, 6.8))) * w + st.hover * 0.12 + st.burst * 0.9;
      var pop = 1 + 0.06 * Math.sin(seg(t, 2.0, 3.6) * Math.PI) * w + st.hover * 0.05;
      var rx = ISO_X - nod - st.ly * 22 + apart * 10 - st.pitch;
      var ry = ISO_Y + sway + spin + st.lx * 32 + st.yaw;
      cube.style.transform = "translateY(" + (float * size).toFixed(2) + "px) scale(" + pop.toFixed(4) + ") rotateX(" + rx.toFixed(2) + "deg) rotateY(" + ry.toFixed(2) + "deg)";
      var dz = size / 2 + apart * size * 0.24;
      faces.forEach(function (f, i) {
        f.el.style.transform = f.tr + " translateZ(" + dz.toFixed(2) + "px)" + (apart > 0.001 ? " rotateZ(" + (apart * (i % 2 ? 5 : -5)).toFixed(2) + "deg)" : "");
        var g = seg(t, 4.0 + i * 0.05, 4.9 + i * 0.05);                     // блик
        f.glare.setAttribute("x", (-160 + g * 300).toFixed(1));
        f.glare.setAttribute("opacity", g > 0 && g < 1 ? 1 : 0);
      });
      shade(rx, ry);
      glow.style.opacity = Math.min(1.4, 0.75 + 0.25 * Math.sin(t / PERIOD * Math.PI * 6) + (spin > 0 && spin < 360 ? 0.2 : 0) + st.hover * 0.35 + st.burst * 0.5).toFixed(3);
      glow.style.transform = "translate(" + (st.lx * 14).toFixed(1) + "px," + (st.ly * 10).toFixed(1) + "px) scale(" + (1 + st.hover * 0.08 + st.burst * 0.15).toFixed(3) + ")";
      shadow.style.transform = "scale(" + (1 - float * 4 + apart * 0.25).toFixed(3) + ")";
      shadow.style.opacity = (0.9 - float * 6).toFixed(3);
    }

    var reduce = host.hasAttribute("data-static") ||
      (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    var visible = true, raf = 0, prev = 0, autoT = 0;
    function step(now) {
      raf = 0;
      if (!visible || document.hidden) { prev = 0; return; }
      var dt = prev ? Math.min((now - prev) / 1000, 0.05) : 0.016;
      prev = now;
      var k = 1 - Math.pow(0.001, dt);                                      // сглаживание, не зависящее от FPS
      I.lx += (I.tlx - I.lx) * k * 0.9; I.ly += (I.tly - I.ly) * k * 0.9;
      I.hover += (I.thover - I.hover) * k;
      I.burst *= Math.pow(0.02, dt);
      var busy = I.drag || I.hover > 0.05 || Math.abs(I.vyaw) > 30 || I.burst > 0.05;
      I.idle = busy ? 0 : I.idle + dt;
      var targetAuto = I.idle > 1.2 ? 1 : 0;                                // пока с кубом играют, цикл ждёт
      I.auto += (targetAuto - I.auto) * k * 0.6;
      if (!I.drag) {
        I.yaw += I.vyaw * dt; I.pitch += I.vpitch * dt;                     // инерция
        I.vyaw *= Math.pow(0.12, dt); I.vpitch *= Math.pow(0.05, dt);
        if (Math.abs(I.vyaw) < 40) {                                        // мягко домой: ближайший целый оборот
          var home = Math.round(I.yaw / 360) * 360;
          I.yaw += (home - I.yaw) * k * 0.5;
          I.pitch += (0 - I.pitch) * k * 0.5;
        }
      }
      if (I.auto > 0.02) autoT += dt * I.auto;
      render(autoT, I);
      raf = requestAnimationFrame(step);
    }
    function kick() { if (!raf && !reduce) raf = requestAnimationFrame(step); }
    render(0);
    if (!reduce) {
      if ("IntersectionObserver" in window) {
        new IntersectionObserver(function (es) { visible = es[0].isIntersecting; kick(); }).observe(host);
      }
      document.addEventListener("visibilitychange", kick);
      kick();
    }
    host.__cube = { render: render, state: I };
  }

  function init() { document.querySelectorAll("[data-cube-loop]").forEach(mount); }
  window.Cube2MP = { mount: mount, init: init };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
