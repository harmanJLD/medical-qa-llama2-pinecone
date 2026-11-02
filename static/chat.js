const log = document.getElementById("log");
const form = document.getElementById("ask-form");
const box = document.getElementById("question");
const send = document.getElementById("send");
const docsList = document.getElementById("docs");
const docCount = document.getElementById("doc-count");

// Matches [p. 7], [page 7], [p. 7.0] and the long form [file.pdf, p. 7].
const CITE_SPLIT = /(\[(?:[^\[\]]*?,\s*)?p(?:age|\.)?\s*\d+(?:\.0)?\])/i;
const CITE_MATCH = /^\[(?:[^\[\]]*?,\s*)?p(?:age|\.)?\s*(\d+)(?:\.0)?\]$/i;

let titles = {};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToEnd() {
  log.scrollTop = log.scrollHeight;
}

function titleFor(name) {
  return titles[name] || name.replace(/\.pdf$/i, "").replace(/[-_]/g, " ");
}

// Answer text with [p. N] turned into highlighter marks that jump to their source card.
function renderAnswer(container, text, sourcesBox, pages) {
  container.replaceChildren();
  text.split(CITE_SPLIT).forEach((part) => {
    const m = part.match(CITE_MATCH);
    if (!m) return container.appendChild(document.createTextNode(part));
    const page = Number(m[1]);
    const known = pages.includes(page);
    const mark = el("a", known ? "cite" : "cite bad", `p. ${page}`);
    mark.href = "#";
    mark.title = known ? `Show the passage from page ${page}` : `Page ${page} was not among the retrieved passages`;
    mark.addEventListener("click", (e) => {
      e.preventDefault();
      const card = sourcesBox.querySelector(`[data-page="${page}"]`);
      if (!card) return;
      card.querySelector("details")?.setAttribute("open", "");
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 1400);
    });
    container.appendChild(mark);
  });
}

function renderSources(boxEl, sources) {
  boxEl.replaceChildren();
  if (!sources || !sources.length) return;
  boxEl.appendChild(el("p", "sources-label", sources.length === 1 ? "Source" : `Sources (${sources.length})`));
  sources.forEach((s) => {
    const card = el("article", "source");
    card.dataset.page = s.page;

    const tab = el("div", "tab");
    tab.append(el("small", "", "page"), el("b", "", String(s.page)));

    const body = el("div", "source-body");
    body.appendChild(el("p", "source-title", titleFor(s.source)));
    if (s.relevance !== null && s.relevance !== undefined) {
      const meta = el("div", "source-meta");
      const meter = el("div", "meter");
      const fill = el("span");
      fill.style.width = `${Math.max(6, Math.round(Math.min(1, s.relevance) * 100))}%`;
      meter.appendChild(fill);
      meta.append(meter, el("span", "", `match ${Math.round(Math.min(1, s.relevance) * 100)}%`));
      body.appendChild(meta);
    }
    const details = el("details");
    details.append(el("summary", "", "Read the passage"), el("blockquote", "", `${s.excerpt}…`));
    body.appendChild(details);

    card.append(tab, body);
    boxEl.appendChild(card);
  });
  highlightLibrary(sources.map((s) => s.source));
}

function highlightLibrary(names) {
  docsList.querySelectorAll("li").forEach((li) => li.classList.toggle("active", names.includes(li.dataset.name)));
}

function startTimer(label) {
  const status = el("div", "status");
  const text = el("span", "", "Searching the library…");
  const timer = el("span", "timer", "");
  status.append(el("span", "pulse"), text, timer);
  const started = performance.now();
  const id = setInterval(() => {
    timer.textContent = `${Math.round((performance.now() - started) / 1000)} s`;
  }, 500);
  return {
    node: status,
    set(msg) { text.textContent = msg; },
    stop() { clearInterval(id); status.remove(); },
  };
}

function seconds(v) {
  return v === null || v === undefined ? null : `${Number(v).toFixed(1)} s`;
}

async function ask(question) {
  document.getElementById("welcome")?.remove();
  send.disabled = true;

  const ex = el("section", "exchange");
  const q = el("h2", "question", question);
  const status = startTimer();
  const answer = el("p", "answer");
  const sourcesBox = el("div", "sources");
  const timing = el("p", "timing");
  ex.append(q, status.node, answer, sourcesBox, timing);
  log.appendChild(ex);
  scrollToEnd();

  let text = "";
  let pages = [];
  try {
    const res = await fetch("/api/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `The server returned an error (${res.status}).`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        const event = JSON.parse(part.slice(6));
        if (event.type === "sources") {
          pages = event.sources.map((s) => s.page);
          renderSources(sourcesBox, event.sources);
          status.set(`Found ${event.sources.length} relevant passages. Llama 2 is reading them…`);
        } else if (event.type === "token") {
          if (!text) {
            status.stop();
            answer.classList.add("typing");
          }
          text += event.text;
          renderAnswer(answer, text, sourcesBox, pages);
        } else if (event.type === "error") {
          throw new Error(event.message);
        } else if (event.type === "done") {
          status.stop();
          finish(ex, answer, sourcesBox, timing, event);
        }
      }
      scrollToEnd();
    }
  } catch (err) {
    status.stop();
    answer.classList.remove("typing");
    answer.className = "answer error";
    answer.textContent = /fetch|network/i.test(err.message)
      ? "The app stopped responding. Check the terminal running app.py, then ask again."
      : err.message;
  } finally {
    send.disabled = false;
    box.focus();
  }
}

function finish(ex, answer, sourcesBox, timing, event) {
  answer.classList.remove("typing");
  if (event.abstained) {
    ex.insertBefore(el("p", "note", "Not covered by the library"), answer);
    answer.textContent = "The guides in the library don't answer this. Try rephrasing, or add a PDF that covers it.";
    sourcesBox.remove();
    highlightLibrary([]);
  } else {
    const pages = event.sources.map((s) => s.page);
    renderAnswer(answer, event.answer, sourcesBox, pages);
    renderSources(sourcesBox, event.sources);
    if (event.uncited_pages && event.uncited_pages.length) {
      ex.insertBefore(el("p", "note", `Cites a page that wasn't retrieved (p. ${event.uncited_pages.join(", p. ")}), so check it`), answer);
    } else if (!event.cited_pages || !event.cited_pages.length) {
      ex.insertBefore(el("p", "note", "No page citations in this answer, so check the sources below"), answer);
    }
  }
  const t = event.timings || {};
  const parts = [];
  if (t.retrieval_s !== undefined) parts.push(`Found sources in ${seconds(t.retrieval_s)}`);
  if (t.generation_s) parts.push(`answered in ${seconds(t.generation_s)}`);
  timing.textContent = parts.length ? `${parts.join(", ")}.` : "";
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = box.value.trim();
  if (!q || send.disabled) return;
  box.value = "";
  autosize();
  ask(q);
});
box.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
function autosize() {
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 160)}px`;
}
box.addEventListener("input", autosize);
document.querySelectorAll(".starter").forEach((b) =>
  b.addEventListener("click", () => {
    if (!send.disabled) ask(b.textContent);
  })
);

// Library and upload
const fileInput = document.getElementById("file");
const fileLabel = document.getElementById("file-label");
const uploadForm = document.getElementById("upload-form");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");

async function loadDocs() {
  try {
    const data = await (await fetch("/api/documents")).json();
    docsList.replaceChildren();
    titles = {};
    data.documents.forEach((d) => (titles[d.name] = d.title));
    const n = data.documents.length;
    docCount.textContent = n ? `${n} ${n === 1 ? "guide" : "guides"}` : "";
    if (!data.documents.length) {
      docsList.appendChild(el("li", "quiet", "No documents yet. Add a PDF below."));
      return;
    }
    data.documents
      .slice()
      .sort((a, b) => a.title.localeCompare(b.title))
      .forEach((d) => {
        const li = el("li", "", d.title);
        li.dataset.name = d.name;
        li.title = d.name;
        docsList.appendChild(li);
      });
  } catch {
    docsList.replaceChildren(el("li", "quiet", "Couldn't load the library."));
  }
}

fileInput.addEventListener("change", () => {
  fileLabel.textContent = fileInput.files[0] ? fileInput.files[0].name : "Add a PDF to the library";
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files[0]) {
    uploadStatus.textContent = "Choose a PDF first.";
    return;
  }
  const body = new FormData();
  body.append("file", fileInput.files[0]);
  uploadBtn.disabled = true;
  uploadStatus.textContent = "Indexing: splitting, embedding, and saving to Pinecone…";
  try {
    const res = await fetch("/api/upload", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    uploadStatus.textContent = `Indexed ${data.name}: ${data.pages} pages, ${data.chunks} passages.`;
    fileInput.value = "";
    fileLabel.textContent = "Add a PDF to the library";
    loadDocs();
  } catch (err) {
    uploadStatus.textContent = err.message || "Indexing failed.";
  } finally {
    uploadBtn.disabled = false;
  }
});

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    document.getElementById("fetch-k").textContent = h.fetch_k;
    document.getElementById("top-k").textContent = h.top_k;
  })
  .catch(() => {});
loadDocs();
