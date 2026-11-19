(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const thread = $("#thread"), input = $("#input"), sendBtn = $("#sendBtn"), stopBtn = $("#stopBtn");
  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  };

  // ------------------------------------------------------------------ i18n
  const I18N = {
    en: { brand: "Property Concierge", newChat: "New conversation", knowledge: "Knowledge base", projects: "DarGlobal projects",
      listings: "Wasalt listings", pages: "Pages indexed", chunks: "Search chunks", engine: "Engine",
      retrieval: "Hybrid retrieval · BM25 + n-grams + semantic", filters: "Structured listing filters (EN/AR)",
      eyebrow: "DarGlobal developments · Wasalt marketplace", hero: "How can I help you find your next property?",
      lede: "Ask about DarGlobal's branded residences across the Gulf and Europe, or search Wasalt listings in Saudi Arabia by city, budget and bedrooms. Facts are cited to their source pages.",
      placeholder: "Ask about a project, a city, a budget…", disclaimer: "Answers are generated from publicly available website data and may be out of date. Not financial advice.",
      thinking: "Searching DarGlobal & Wasalt data", sources: "Sources", copy: "Copy", copied: "Copied", cached: "cached",
      matches: (n) => `${n} matching listing${n === 1 ? "" : "s"}`, relaxed: "closest matches (filters relaxed)", median: "median",
      view: "View on Wasalt", bed: "bd", bath: "ba", sqm: "sqm", onRequest: "Price on request", stopped: "Stopped.",
      scraped: (d) => `Data scraped ${d}.`, seed: "Showing a bundled seed snapshot — run the scraper for full data.",
      noKey: "No API key configured — retrieval-only mode", lang: "العربية", rateLimited: "You're sending messages quickly — please wait a moment.",
      netErr: "Connection problem — please try again.", refreshing: "Live refresh in progress…", refreshed: (d) => `Live data refreshed ${d}.` },
    ar: { brand: "مستشار العقارات", newChat: "محادثة جديدة", knowledge: "قاعدة المعرفة", projects: "مشاريع دار جلوبال",
      listings: "عروض وصلت", pages: "صفحات مفهرسة", chunks: "مقاطع البحث", engine: "المحرك",
      retrieval: "استرجاع هجين · BM25 + مقاطع حرفية + دلالي", filters: "فلاتر عقارية منظمة (عربي/إنجليزي)",
      eyebrow: "مشاريع دار جلوبال · سوق وصلت", hero: "كيف أساعدك في العثور على عقارك القادم؟",
      lede: "اسأل عن مساكن دار جلوبال الفاخرة في الخليج وأوروبا، أو ابحث في عروض وصلت في السعودية حسب المدينة والميزانية وعدد الغرف. المعلومات موثقة بروابط مصادرها.",
      placeholder: "اسأل عن مشروع أو مدينة أو ميزانية…", disclaimer: "الإجابات مولدة من بيانات المواقع العامة وقد لا تكون محدثة. ليست نصيحة مالية.",
      thinking: "جارٍ البحث في بيانات دار جلوبال ووصلت", sources: "المصادر", copy: "نسخ", copied: "تم النسخ", cached: "من الذاكرة",
      matches: (n) => `${n} عرض مطابق`, relaxed: "أقرب النتائج (تم تخفيف الفلاتر)", median: "الوسيط",
      view: "عرض على وصلت", bed: "غرف", bath: "حمام", sqm: "م²", onRequest: "السعر عند الطلب", stopped: "تم الإيقاف.",
      scraped: (d) => `تم جمع البيانات ${d}.`, seed: "يتم عرض بيانات تجريبية مضمنة — شغّل أداة الجمع للحصول على البيانات الكاملة.",
      noKey: "لا يوجد مفتاح API — وضع الاسترجاع فقط", lang: "English", rateLimited: "ترسل الرسائل بسرعة — يرجى الانتظار قليلاً.",
      netErr: "مشكلة في الاتصال — حاول مرة أخرى.", refreshing: "جارٍ تحديث البيانات…", refreshed: (d) => `تم تحديث البيانات ${d}.` },
  };
  let lang = store.get("lang", "en");
  const t = (k, ...a) => { const v = I18N[lang][k] ?? I18N.en[k]; return typeof v === "function" ? v(...a) : v; };
  function applyLang() {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
    document.querySelectorAll("[data-i18n-ph]").forEach((el) => (el.placeholder = t(el.dataset.i18nPh)));
    $("#langToggle").textContent = t("lang");
    renderHealth();
  }

  // ----------------------------------------------------------------- theme
  let theme = store.get("theme", matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const applyTheme = () => { document.documentElement.dataset.theme = theme; $("#themeToggle").textContent = theme === "dark" ? "☀" : "☾"; };

  // -------------------------------------------------------------- markdown
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
    s = s.replace(/(^|\s)_([^_\n]+)_(?=\s|$|[.,;:!?)])/g, "$1<em>$2</em>");
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    s = s.replace(/\[(L?)(\d{1,2})\](?!\()/g, (m, l, n) => `<a href="#" class="cite${l ? " l" : ""}" data-ref="${l}${n}">${l}${n}</a>`);
    s = s.replace(/\[((?:L?\d{1,2}\s*,\s*)+L?\d{1,2})\]/g, (m, list) =>
      list.split(/\s*,\s*/).map((r) => { const l = r.startsWith("L"); return `<a href="#" class="cite${l ? " l" : ""}" data-ref="${r}">${r}</a>`; }).join(""));
    return s;
  }
  function md(src) {
    const lines = src.replace(/\r/g, "").split("\n");
    let html = "", i = 0;
    const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
    const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    while (i < lines.length) {
      const line = lines[i];
      if (/^```/.test(line)) {
        const buf = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
        i++; html += `<pre><code>${esc(buf.join("\n"))}</code></pre>`; continue;
      }
      if (/^\s*$/.test(line)) { i++; continue; }
      let m;
      if ((m = line.match(/^(#{1,4})\s+(.*)/))) { html += `<h${m[1].length + 1}>${inline(m[2])}</h${m[1].length + 1}>`; i++; continue; }
      if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        const head = cells(line); i += 2;
        let rows = "";
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows += "<tr>" + cells(lines[i]).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>"; i++; }
        html += `<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`; continue;
      }
      if (/^\s*>/.test(line)) {
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
        html += `<blockquote>${inline(buf.join(" "))}</blockquote>`; continue;
      }
      if (/^\s*([-*•]|\d+[.)])\s+/.test(line)) {
        const ordered = /^\s*\d+[.)]/.test(line); const items = [];
        while (i < lines.length && /^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) {
          let item = lines[i++].replace(/^\s*([-*•]|\d+[.)])\s+/, "");
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) item += " " + lines[i++].trim();
          items.push(`<li>${inline(item)}</li>`);
        }
        html += ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`; continue;
      }
      const buf = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|\s*>|\s*([-*•]|\d+[.)])\s+)/.test(lines[i])) {
        if (lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) break;
        buf.push(lines[i++]);
      }
      html += `<p>${buf.map(inline).join("<br>")}</p>`;
    }
    return html;
  }

  // ------------------------------------------------------------- formatting
  const nf = () => new Intl.NumberFormat(lang === "ar" ? "ar-SA" : "en-US", { maximumFractionDigits: 0 });
  function price(v, period) {
    if (v == null) return t("onRequest");
    const s = v >= 1e6 ? `${(v / 1e6).toLocaleString(lang === "ar" ? "ar-SA" : "en-US", { maximumFractionDigits: 2 })}M` : nf().format(v);
    return `SAR ${s}` + (period ? ` <small>/ ${esc(period)}</small>` : "");
  }

  // ----------------------------------------------------------------- health
  let health = null;
  async function loadHealth() {
    try { health = await (await fetch("/api/health")).json(); renderHealth(); } catch {}
  }
  function renderHealth() {
    if (!health) return;
    const d = health.data;
    $("#statProjects").textContent = nf().format(d.projects);
    $("#statListings").textContent = nf().format(d.listings);
    $("#statPages").textContent = nf().format(d.pages);
    $("#statChunks").textContent = nf().format(d.chunks);
    const r = health.refresh || {};
    let note = "";
    if (r.running) note = t("refreshing");
    else if (r.last_status === "ok" && r.last_run) note = t("refreshed", new Date(r.last_run).toLocaleString());
    else if (d.scraped_at) note = t("scraped", new Date(d.scraped_at).toLocaleDateString());
    if (Object.values(d.data_source || {}).includes("seed")) note += " " + t("seed");
    $("#dataNote").textContent = note.trim();
    const dot = $("#llmDot");
    if (health.llm.configured) {
      dot.className = "dot ok";
      $("#llmModel").textContent = `${health.llm.candidates[0] || "openrouter/free"} (+${Math.max(0, health.llm.candidates.length - 1)} fallbacks)`;
    } else { dot.className = "dot warn"; $("#llmModel").textContent = t("noKey"); }
  }

  // ----------------------------------------------------------- suggestions
  const SUGG_ICONS = ["◆", "◈", "✦", "⌂", "≈", "ع"];
  async function loadSuggestions() {
    let list = [];
    try { list = (await (await fetch(`/api/suggestions?lang=${lang}`)).json()).suggestions; } catch {}
    $("#suggestions").innerHTML = list.map((s, i) =>
      `<button class="sugg ${/wasalt|villa|price|listing|rent|sale/i.test(s) ? "w" : ""}" data-q="${esc(s)}"><span class="sugg-ic">${SUGG_ICONS[i % 6]}</span><span>${esc(s)}</span></button>`).join("");
  }

  // ------------------------------------------------------------------- chat
  let history = [];
  let controller = null;

  function scrollDown(force) {
    const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160;
    if (force || nearBottom) thread.scrollTop = thread.scrollHeight;
  }
  function addUser(text) {
    $("#welcome")?.remove();
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="bubble-user"></div>`;
    el.firstChild.textContent = text;
    thread.appendChild(el); scrollDown(true);
  }
  function addBot() {
    const el = document.createElement("div");
    el.className = "msg";
    el.innerHTML = `<div class="avatar" aria-hidden="true">×</div><div class="bot">
      <div class="thinking"><i></i><i></i><i></i><span>${esc(t("thinking"))}</span></div>
      <div class="answer"></div><div class="listings" hidden></div><details class="sources" hidden></details><div class="meta" hidden></div></div>`;
    thread.appendChild(el); scrollDown(true);
    return el;
  }

  function renderListings(box, data) {
    if (!data || !data.items || !data.items.length) return;
    const f = Object.entries(data.filters || {}).map(([k, v]) => `<span class="pill">${esc(k.replace(/_/g, " "))}: ${esc(Array.isArray(v) ? v.join(", ") : typeof v === "number" && v > 9999 ? nf().format(v) : v)}</span>`).join("");
    const stats = data.stats && data.stats.median_price ? `<span class="pill">${t("median")} ${price(data.stats.median_price)}</span>` : "";
    const relaxed = data.relaxed && data.relaxed.length ? `<span class="pill warn">${esc(t("relaxed"))}</span>` : "";
    box.innerHTML = `<div class="listings-head"><strong>${esc(t("matches", data.total))}</strong>${relaxed}${stats}${f}</div>
      <div class="cards">${data.items.map((l) => {
        const area = l.built_area_sqm || l.land_area_sqm;
        return `<a class="card" id="L${l.n}" href="${esc(l.url)}" target="_blank" rel="noopener" data-ref="L${l.n}">
          <div class="card-top"><span class="card-tag">L${l.n} · ${esc((l.property_type || "Property") + " · " + (l.purpose || ""))}</span><span>${esc(l.building_year || "")}</span></div>
          <div class="card-price">${price(l.price_sar, l.rent_period)}</div>
          <div class="card-loc">${esc([l.district, l.city].filter(Boolean).join(", "))}</div>
          <div class="card-facts">${l.bedrooms != null ? `<span>${l.bedrooms} ${t("bed")}</span>` : ""}${l.bathrooms != null ? `<span>${l.bathrooms} ${t("bath")}</span>` : ""}${area ? `<span>${nf().format(area)} ${t("sqm")}</span>` : ""}</div>
          <div class="card-title">${esc(l.title || "")}</div></a>`;
      }).join("")}</div>`;
    box.hidden = false;
  }
  function renderSources(box, sources) {
    if (!sources || !sources.length) return;
    box.innerHTML = `<summary>${esc(t("sources"))} (${sources.length})</summary><div class="src-list">${sources.map((s) =>
      `<a class="src" data-ref="${s.n}" href="${esc(s.url)}" target="_blank" rel="noopener"><span class="src-n">${s.n}</span><span><span class="site ${esc(s.site)}">${esc(s.site === "wasalt" ? "Wasalt" : "DarGlobal")}</span><span class="src-t">${esc(s.title)}</span><div class="src-s">${esc(s.snippet)}</div></span></a>`).join("")}</div>`;
    box.hidden = false;
  }

  async function send(text) {
    text = text.trim();
    if (!text || controller) return;
    input.value = ""; autosize();
    addUser(text);
    const el = addBot();
    const ansEl = el.querySelector(".answer"), thinking = el.querySelector(".thinking");
    let answer = "", meta = {}, raf = 0, sources = [];
    const paint = () => { raf = 0; ansEl.innerHTML = md(answer); ansEl.classList.add("cursor"); scrollDown(); };

    controller = new AbortController();
    setBusy(true);
    try {
      const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history: history.slice(-12) }), signal: controller.signal });
      if (res.status === 429) throw new Error(t("rateLimited"));
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
          const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("\n");
          if (!data) continue;
          let ev; try { ev = JSON.parse(data); } catch { continue; }
          if (ev.type === "meta") {
            renderListings(el.querySelector(".listings"), ev.listings);
            sources = ev.sources || [];  // rendered at the end: only the ones the answer cites
          } else if (ev.type === "delta") {
            thinking.remove?.(); answer += ev.text; if (!raf) raf = requestAnimationFrame(paint);
          } else if (ev.type === "done") { meta = ev; }
          else if (ev.type === "error") { throw new Error(ev.message); }
        }
      }
    } catch (e) {
      thinking.remove?.();
      if (e.name === "AbortError") answer += (answer ? "\n\n" : "") + `_${t("stopped")}_`;
      else ansEl.insertAdjacentHTML("afterend", `<p class="error">${esc(e.message === "Failed to fetch" ? t("netErr") : e.message)}</p>`);
    } finally {
      cancelAnimationFrame(raf);
      ansEl.innerHTML = md(answer); ansEl.classList.remove("cursor");
      // show only sources the answer actually cites ([n]); small talk / general answers cite none
      const cited = new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
      renderSources(el.querySelector(".sources"), sources.filter((s) => cited.has(s.n)));
      if (answer) {
        history.push({ role: "user", content: text }, { role: "assistant", content: answer.slice(0, 4000) });
        const m = el.querySelector(".meta");
        const secs = meta.latency_ms ? `${(meta.latency_ms / 1000).toFixed(1)}s` : "";
        m.innerHTML = `${meta.model ? `<span class="badge ${meta.degraded ? "warn" : ""}">${esc(meta.model)}</span>` : ""}${secs ? `<span>${secs}</span>` : ""}${meta.cached ? `<span class="badge">${t("cached")}</span>` : ""}<button type="button" class="copy">${t("copy")}</button>`;
        m.hidden = false;
        m.querySelector(".copy").onclick = (ev) => { navigator.clipboard?.writeText(answer); ev.target.textContent = t("copied"); setTimeout(() => (ev.target.textContent = t("copy")), 1200); };
      }
      controller = null; setBusy(false); scrollDown();
    }
  }

  function setBusy(b) { stopBtn.hidden = !b; sendBtn.hidden = b; input.disabled = false; if (!b) input.focus(); }
  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 180) + "px"; sendBtn.disabled = !input.value.trim(); }
  function newChat() {
    controller?.abort(); history = [];
    thread.innerHTML = `<div class="welcome" id="welcome"><p class="eyebrow">${esc(t("eyebrow"))}</p><h1>${esc(t("hero"))}</h1><p class="lede">${esc(t("lede"))}</p><div class="suggestions" id="suggestions"></div></div>`;
    loadSuggestions(); closeMenu(); input.focus();
  }
  const closeMenu = () => { $("#sidebar").classList.remove("open"); $("#scrim").classList.remove("show"); };

  // ----------------------------------------------------------------- events
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(input.value); } });
  stopBtn.addEventListener("click", () => controller?.abort());
  thread.addEventListener("click", (e) => {
    const s = e.target.closest(".sugg"); if (s) return send(s.dataset.q);
    const c = e.target.closest(".cite"); if (!c) return;
    e.preventDefault();
    const bot = c.closest(".bot"), ref = c.dataset.ref;
    const target = ref.startsWith("L") ? bot.querySelector(`.card[data-ref="${ref}"]`) : bot.querySelector(`.src[data-ref="${ref}"]`);
    if (!target) return;
    if (!ref.startsWith("L")) bot.querySelector(".sources").open = true;
    target.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
    target.classList.add("flash"); setTimeout(() => target.classList.remove("flash"), 1200);
  });
  $("#newChat").onclick = newChat; $("#newChatTop").onclick = newChat;
  $("#langToggle").onclick = () => { lang = lang === "en" ? "ar" : "en"; store.set("lang", lang); applyLang(); if ($("#welcome")) newChat(); };
  $("#themeToggle").onclick = () => { theme = theme === "dark" ? "light" : "dark"; store.set("theme", theme); applyTheme(); };
  $("#menuBtn").onclick = () => { $("#sidebar").classList.add("open"); $("#scrim").classList.add("show"); };
  $("#scrim").onclick = closeMenu;

  applyTheme(); applyLang(); loadSuggestions(); loadHealth(); autosize();
  setInterval(loadHealth, 60000);
})();
