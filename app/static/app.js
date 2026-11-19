(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const thread = $("#thread"), input = $("#input"), sendBtn = $("#sendBtn"), stopBtn = $("#stopBtn");
  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  };

  // ----------------------------------------------------------------- icons
  const P = (d) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
  const ICONS = {
    plus: P('<path d="M12 5v14M5 12h14"/>'),
    send: P('<path d="M5 12h14M13 6l6 6-6 6"/>'),
    stop: '<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/></svg>',
    sun: P('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'),
    moon: P('<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>'),
    copy: P('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>'),
    check: P('<path d="M5 12.5l4.5 4.5L19 7"/>'),
    retry: P('<path d="M20 12a8 8 0 1 1-2.3-5.6"/><path d="M20 4v5h-5"/>'),
    globe: P('<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.7 3.8 5.7 3.8 9s-1.3 6.3-3.8 9c-2.5-2.7-3.8-5.7-3.8-9S9.5 5.7 12 3z"/>'),
    tower: P('<path d="M7 21V5l5-2 5 2v16M4 21h16M10 8h1M13 8h1M10 12h1M13 12h1M10 16h1M13 16h1"/>'),
    home: P('<path d="M4 11l8-7 8 7"/><path d="M6 9.5V20h12V9.5M10 20v-5h4v5"/>'),
    chart: P('<path d="M4 20h16M7 16v-4M12 16V8M17 16v-7"/>'),
  };
  const icon = (n) => ICONS[n] || "";
  document.querySelectorAll("[data-icon]").forEach((el) => (el.innerHTML = icon(el.dataset.icon)));

  // ------------------------------------------------------------------ i18n
  const I18N = {
    en: {
      tagline: "Property assistant · DarGlobal & Wasalt", newChat: "New chat",
      hello: "Hi, I'm Haya", lede: "Ask me about DarGlobal's branded residences, or find homes for sale and rent on Wasalt across Saudi Arabia, in English or Arabic.",
      placeholder: "Ask Haya about a project, a city or a budget…",
      disclaimer: "Haya can make mistakes. Listing prices come from Wasalt and may have changed. Not financial advice.",
      thinking: "Looking through DarGlobal & Wasalt…", copy: "Copy", copied: "Copied", retry: "Regenerate",
      listings: (n) => `${n.toLocaleString("en-US")} matching listing${n === 1 ? "" : "s"}`, relaxed: "Closest matches", median: "Median",
      bed: (n) => `${n} bed${n === 1 ? "" : "s"}`, bath: (n) => `${n} bath${n === 1 ? "" : "s"}`, sqm: "m²", onRequest: "Price on request",
      sale: "For sale", rent: "For rent", under: "Under", over: "Over", min: "Min", stopped: "Stopped.",
      limited: "Limited mode - AI summary unavailable, showing results directly",
      fresh: (d) => `Based on public pages of darglobal.co.uk and wasalt.sa · updated ${d}`,
      rateLimited: "You're sending messages quickly - please wait a moment.", netErr: "Connection problem - please try again.",
      starters: [
        ["tower", "DarGlobal in Saudi Arabia", "Which DarGlobal projects are in Saudi Arabia?", "Which DarGlobal projects are in Saudi Arabia?"],
        ["globe", "Trump Tower Jeddah", "Completion date, floors and unit types", "Tell me about Trump Tower Jeddah"],
        ["home", "Villas in Riyadh under 2M", "Live Wasalt listings with photos", "Show me villas for sale in Riyadh under 2 million SAR", "w"],
        ["chart", "Riyadh price check", "Median villa price on Wasalt", "What is the median villa price in Riyadh?", "w"],
      ],
    },
    ar: {
      tagline: "مساعدتك العقارية · دار جلوبال ووصلت", newChat: "محادثة جديدة",
      hello: "مرحباً، أنا هيا", lede: "اسألني عن مساكن دار جلوبال الفاخرة، أو ابحث عن عقارات للبيع والإيجار على وصلت في أنحاء السعودية - بالعربية أو الإنجليزية.",
      placeholder: "اسأل هيا عن مشروع أو مدينة أو ميزانية…",
      disclaimer: "قد تخطئ هيا أحياناً. أسعار العروض من وصلت وقد تكون تغيّرت. ليست نصيحة مالية.",
      thinking: "أبحث في دار جلوبال ووصلت…", copy: "نسخ", copied: "تم النسخ", retry: "إعادة التوليد",
      listings: (n) => `${n.toLocaleString("ar-SA")} عرض مطابق`, relaxed: "أقرب النتائج", median: "الوسيط",
      bed: (n) => `${n} غرف`, bath: (n) => `${n} حمام`, sqm: "م²", onRequest: "السعر عند الطلب",
      sale: "للبيع", rent: "للإيجار", under: "أقل من", over: "أكثر من", min: "من", stopped: "تم الإيقاف.",
      limited: "وضع محدود - الملخص الذكي غير متاح، تُعرض النتائج مباشرة",
      fresh: (d) => `من الصفحات العامة لموقعي دار جلوبال ووصلت · آخر تحديث ${d}`,
      rateLimited: "ترسل الرسائل بسرعة - يرجى الانتظار قليلاً.", netErr: "مشكلة في الاتصال - حاول مرة أخرى.",
      starters: [
        ["tower", "دار جلوبال في السعودية", "المشاريع في الرياض وجدة ووادي صفار", "ما هي مشاريع دار جلوبال في السعودية؟"],
        ["globe", "برج ترامب جدة", "موعد التسليم وعدد الطوابق والوحدات", "أخبرني عن برج ترامب جدة"],
        ["home", "فلل في الرياض بأقل من 2 مليون", "عروض وصلت مع الصور", "أرني فلل للبيع في الرياض بأقل من 2 مليون ريال", "w"],
        ["chart", "أسعار الرياض", "متوسط سعر الفلل على وصلت", "ما متوسط سعر الفلل في الرياض؟", "w"],
      ],
    },
  };
  let lang = store.get("lang", "en");
  const t = (k, ...a) => { const v = I18N[lang][k] ?? I18N.en[k]; return typeof v === "function" ? v(...a) : v; };
  function applyLang() {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
    document.querySelectorAll("[data-i18n-ph]").forEach((el) => (el.placeholder = t(el.dataset.i18nPh)));
    $("#langToggle").textContent = lang === "ar" ? "EN" : "ع";
  }

  // ----------------------------------------------------------------- theme
  let theme = store.get("theme", matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const applyTheme = () => { document.documentElement.dataset.theme = theme; $("#themeToggle").innerHTML = icon(theme === "dark" ? "sun" : "moon"); };

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
        html += `<div class="tbl"><table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`; continue;
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
  const loc = () => (lang === "ar" ? "ar-SA" : "en-US");
  const nf = (v) => new Intl.NumberFormat(loc(), { maximumFractionDigits: 0 }).format(v);
  const money = (v) => (v >= 1e6 ? `${(v / 1e6).toLocaleString(loc(), { maximumFractionDigits: 2 })}M` : v >= 1e3 ? `${Math.round(v / 1e3).toLocaleString(loc())}K` : nf(v));
  function price(v, period) {
    if (v == null) return t("onRequest");
    const p = { yearly: "/yr", monthly: "/mo", weekly: "/wk", daily: "/day" }[period] || (period ? `/${period}` : "");
    return `SAR ${money(v)}` + (p ? ` <small>${esc(p)}</small>` : "");
  }
  function filterPills(f) {
    const out = [];
    if (f.property_type) out.push(f.property_type);
    if (f.purpose) out.push(t(f.purpose) || f.purpose);
    if (f.district) out.push(f.district);
    if (f.city) out.push(f.city);
    if (f.bedrooms_exact != null) out.push(t("bed", f.bedrooms_exact));
    if (f.bedrooms_min != null) out.push(`${f.bedrooms_min}+ ${lang === "ar" ? "غرف" : "beds"}`);
    if (f.price_max) out.push(`${t("under")} SAR ${money(f.price_max)}`);
    if (f.price_min) out.push(`${t("over")} SAR ${money(f.price_min)}`);
    if (f.area_min) out.push(`${t("min")} ${nf(f.area_min)} ${t("sqm")}`);
    return out.map((x) => `<span class="pill">${esc(x)}</span>`).join("");
  }

  // ----------------------------------------------------------------- welcome
  let freshDate = "";
  async function loadFreshness() {
    try {
      const h = await (await fetch("/api/health")).json();
      const d = h.refresh?.last_status === "ok" && h.refresh?.last_run ? h.refresh.last_run : h.data?.scraped_at;
      if (d) { freshDate = d; const el = $("#fresh"); if (el) el.textContent = t("fresh", fmtDate(d)); }
    } catch {}
  }
  const fmtDate = (d) => new Date(d).toLocaleDateString(loc(), { day: "numeric", month: "short", year: "numeric" });
  function renderWelcome() {
    thread.innerHTML = `<section class="welcome" id="welcome">
      <div class="logo" aria-hidden="true">H</div>
      <h1>${esc(t("hello"))}</h1>
      <p class="lede">${esc(t("lede"))}</p>
      <div class="starters">${t("starters").map(([ic, title, desc, q, w]) =>
        `<button type="button" class="starter ${w || ""}" data-q="${esc(q)}"><span class="sic">${icon(ic)}</span><span><b>${esc(title)}</b><span class="d">${esc(desc)}</span></span></button>`).join("")}</div>
      <p class="fresh" id="fresh">${freshDate ? esc(t("fresh", fmtDate(freshDate))) : ""}</p>
    </section>`;
  }

  // ------------------------------------------------------------------- chat
  let history = [], controller = null;

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
    el.className = "msg bot-msg";
    el.innerHTML = `<div class="avatar" aria-hidden="true">H</div><div class="bot">
      <div class="thinking"><i></i><i></i><i></i><span>${esc(t("thinking"))}</span></div>
      <div class="answer"></div><div class="listings" hidden></div><div class="sources" hidden></div><div class="actions" hidden></div></div>`;
    thread.appendChild(el); scrollDown(true);
    return el;
  }

  function renderListings(box, data) {
    if (!data || !data.items || !data.items.length) return;
    const median = data.stats?.median_price ? `<span class="pill">${t("median")} SAR ${money(data.stats.median_price)}</span>` : "";
    const relaxed = data.relaxed?.length ? `<span class="pill warn">${esc(t("relaxed"))}</span>` : "";
    box.innerHTML = `<div class="listings-head"><strong>${esc(t("listings", data.total))}</strong>${filterPills(data.filters || {})}${median}${relaxed}</div>
      <div class="cards">${data.items.map((l) => {
        const area = l.built_area_sqm || l.land_area_sqm;
        const img = /^https:\/\//.test(l.image || "") ? `<img src="${esc(l.image)}" alt="" loading="lazy" onerror="this.remove()">` : "";
        const tag = [l.property_type, l.purpose && t(l.purpose)].filter(Boolean).join(" · ");
        return `<a class="card" href="${esc(l.url)}" target="_blank" rel="noopener" data-ref="L${l.n}">
          <div class="card-img">${icon("home")}${img}<span class="card-tag">L${l.n}${tag ? " · " + esc(tag) : ""}</span></div>
          <div class="card-body"><div class="card-price">${price(l.price_sar, l.rent_period)}</div>
          <div class="card-loc">${esc([l.district, l.city].filter(Boolean).join(", "))}</div>
          <div class="card-facts">${l.bedrooms != null ? `<span>${esc(t("bed", l.bedrooms))}</span>` : ""}${l.bathrooms != null ? `<span>${esc(t("bath", l.bathrooms))}</span>` : ""}${area ? `<span>${nf(area)} ${t("sqm")}</span>` : ""}${l.building_year ? `<span>${esc(l.building_year)}</span>` : ""}</div></div></a>`;
      }).join("")}</div>`;
    box.hidden = false;
  }
  function renderSources(box, sources) {
    if (!sources.length) return;
    // one chip per page: [1] and [2] are often two excerpts of the same URL
    const byUrl = new Map();
    for (const s of sources) {
      const g = byUrl.get(s.url) || { ...s, ns: [] }; g.ns.push(s.n); byUrl.set(s.url, g);
    }
    box.innerHTML = [...byUrl.values()].map((s) => {
      const site = s.site === "wasalt" ? "Wasalt" : "DarGlobal";
      return `<a class="src ${esc(s.site)}" data-refs="${s.ns.join(" ")}" href="${esc(s.url)}" target="_blank" rel="noopener" title="${esc(s.snippet || "")}">
        <span class="src-n">${s.ns.join("·")}</span><span class="src-site">${site}</span><span class="src-t">${esc(s.title)}</span></a>`;
    }).join("");
    box.hidden = false;
  }
  function renderActions(el, answer, question, degraded) {
    const box = $(".actions", el);
    box.innerHTML = `<button type="button" class="act copy" title="${esc(t("copy"))}" aria-label="${esc(t("copy"))}">${icon("copy")}</button>
      <button type="button" class="act retry" title="${esc(t("retry"))}" aria-label="${esc(t("retry"))}">${icon("retry")}</button>
      ${degraded ? `<span class="note">${esc(t("limited"))}</span>` : ""}`;
    box.hidden = false;
    $(".copy", box).onclick = (ev) => {
      navigator.clipboard?.writeText(answer.replace(/\[(L?\d{1,2})\]/g, ""));
      const b = ev.currentTarget; b.innerHTML = icon("check"); setTimeout(() => (b.innerHTML = icon("copy")), 1200);
    };
    $(".retry", box).onclick = () => regenerate(el, question);
  }
  function regenerate(el, question) {
    if (controller) return;
    // drop this exchange from history (and anything after it), then ask again
    const idx = history.findLastIndex((m) => m.role === "user" && m.content === question);
    if (idx >= 0) history = history.slice(0, idx);
    let next = el.nextElementSibling;
    while (next) { const n = next.nextElementSibling; next.remove(); next = n; }
    el.remove();
    send(question, { regenerate: true });
  }

  async function send(text, opts = {}) {
    text = text.trim();
    if (!text || controller) return;
    if (!opts.regenerate) { input.value = ""; autosize(); addUser(text); }
    const el = addBot();
    const ansEl = $(".answer", el), thinking = $(".thinking", el);
    let answer = "", meta = {}, raf = 0, sources = [];
    const paint = () => { raf = 0; ansEl.innerHTML = md(answer); ansEl.classList.add("cursor"); scrollDown(); };

    controller = new AbortController();
    setBusy(true);
    try {
      const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history: history.slice(-12), regenerate: !!opts.regenerate }), signal: controller.signal });
      if (res.status === 429) throw new Error(t("rateLimited"));
      if (!res.ok || !res.body) throw new Error(t("netErr"));
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
          if (ev.type === "meta") { renderListings($(".listings", el), ev.listings); sources = ev.sources || []; }
          else if (ev.type === "delta") { thinking.remove(); answer += ev.text; if (!raf) raf = requestAnimationFrame(paint); }
          else if (ev.type === "done") meta = ev;
          else if (ev.type === "error") throw new Error(t("netErr"));
        }
      }
    } catch (e) {
      thinking.remove();
      if (e.name === "AbortError") answer += (answer ? "\n\n" : "") + `_${t("stopped")}_`;
      else ansEl.insertAdjacentHTML("afterend", `<p class="error">${esc(e.message === "Failed to fetch" ? t("netErr") : e.message)}</p>`);
    } finally {
      cancelAnimationFrame(raf);
      ansEl.innerHTML = md(answer); ansEl.classList.remove("cursor");
      // only the sources the answer actually cites; small talk / general answers cite none
      const cited = new Set([...answer.matchAll(/\[(\d{1,2})\]/g)].map((m) => Number(m[1])));
      renderSources($(".sources", el), sources.filter((s) => cited.has(s.n)));
      if (answer) {
        history.push({ role: "user", content: text }, { role: "assistant", content: answer.slice(0, 4000) });
        renderActions(el, answer, text, meta.degraded);
      }
      controller = null; setBusy(false); scrollDown();
    }
  }

  function setBusy(b) { stopBtn.hidden = !b; sendBtn.hidden = b; if (!b) { autosize(); input.focus(); } }
  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 180) + "px"; sendBtn.disabled = !input.value.trim(); }
  function newChat() { controller?.abort(); history = []; renderWelcome(); input.value = ""; autosize(); input.focus(); }

  // ----------------------------------------------------------------- events
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(input.value); } });
  stopBtn.addEventListener("click", () => controller?.abort());
  thread.addEventListener("click", (e) => {
    const s = e.target.closest(".starter"); if (s) return send(s.dataset.q);
    const c = e.target.closest(".cite"); if (!c) return;
    e.preventDefault();
    const bot = c.closest(".bot"), ref = c.dataset.ref;
    const target = $(ref.startsWith("L") ? `.card[data-ref="${ref}"]` : `.src[data-refs~="${ref}"]`, bot);
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
    target.classList.add("flash"); setTimeout(() => target.classList.remove("flash"), 1200);
  });
  $("#newChat").onclick = newChat;
  $(".brand").onclick = (e) => { e.preventDefault(); newChat(); };
  $("#langToggle").onclick = () => { lang = lang === "en" ? "ar" : "en"; store.set("lang", lang); applyLang(); if ($("#welcome")) renderWelcome(); };
  $("#themeToggle").onclick = () => { theme = theme === "dark" ? "light" : "dark"; store.set("theme", theme); applyTheme(); };

  applyTheme(); applyLang(); renderWelcome(); loadFreshness(); autosize();
})();
