const log = document.getElementById("log");
const form = document.getElementById("ask-form");
const box = document.getElementById("question");
const send = document.getElementById("send");

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scroll() {
  log.scrollTop = log.scrollHeight;
}

// Render answer text, turning [p. 7] into clickable citations that highlight the matching source.
function renderAnswer(container, text, sourcesBox, retrievedPages) {
  container.replaceChildren();
  text.split(/(\[p(?:age|\.)?\s*\d+\])/i).forEach((part) => {
    const m = part.match(/^\[p(?:age|\.)?\s*(\d+)\]$/i);
    if (!m) return container.appendChild(document.createTextNode(part));
    const page = Number(m[1]);
    const ok = retrievedPages.includes(page);
    const a = el("a", ok ? "cite" : "cite bad", part);
    a.title = ok ? `Show page ${page}` : "This page was not among the retrieved sources";
    a.href = "#";
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const card = sourcesBox.querySelector(`[data-page="${page}"]`);
      if (!card) return;
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 1200);
    });
    container.appendChild(a);
  });
}

function renderSources(box, sources) {
  box.replaceChildren();
  if (!sources.length) return;
  sources.forEach((s) => {
    const card = el("div", "source");
    card.dataset.page = s.page;
    const head = el("div", "source-head");
    head.appendChild(el("b", "", `${s.source}, p. ${s.page}`));
    if (s.relevance !== null && s.relevance !== undefined) {
      const bar = el("div", "bar");
      bar.title = `Relevance ${s.relevance}`;
      const fill = el("span");
      fill.style.width = `${Math.max(4, Math.round(s.relevance * 100))}%`;
      bar.appendChild(fill);
      head.appendChild(bar);
    }
    card.appendChild(head);
    const details = el("details");
    details.appendChild(el("summary", "", "Excerpt"));
    details.appendChild(el("p", "", s.excerpt + "..."));
    card.appendChild(details);
    box.appendChild(card);
  });
}

function addUser(text) {
  const div = el("div", "msg user");
  div.appendChild(el("p", "", text));
  log.appendChild(div);
  scroll();
}

async function ask(question) {
  addUser(question);
  send.disabled = true;

  const msg = el("div", "msg bot");
  const status = el("div", "status");
  status.append(el("span", "dot"), el("span", "", "Searching documents..."));
  const answer = el("p", "cursor");
  const sourcesBox = el("div", "sources");
  const meta = el("div", "meta");
  msg.append(status, answer, sourcesBox, meta);
  log.appendChild(msg);
  scroll();

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
      throw new Error(data.error || `Server error ${res.status}`);
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
          status.lastChild.textContent = "Generating answer...";
        } else if (event.type === "token") {
          text += event.text;
          renderAnswer(answer, text, sourcesBox, pages);
        } else if (event.type === "error") {
          throw new Error(event.message);
        } else if (event.type === "done") {
          finish(msg, status, answer, sourcesBox, meta, event);
        }
      }
      scroll();
    }
  } catch (err) {
    status.remove();
    answer.classList.remove("cursor");
    msg.classList.add("error");
    answer.textContent = err.message.includes("fetch") ? "Could not reach the server. Check the terminal running app.py." : err.message;
  } finally {
    send.disabled = false;
    box.focus();
  }
}

function finish(msg, status, answer, sourcesBox, meta, event) {
  status.remove();
  answer.classList.remove("cursor");
  if (event.abstained) {
    msg.insertBefore(el("span", "badge", "Not found in the documents"), answer);
    answer.textContent = event.answer;
    sourcesBox.remove();
  } else {
    const pages = event.sources.map((s) => s.page);
    renderAnswer(answer, event.answer, sourcesBox, pages);
    renderSources(sourcesBox, event.sources);
    if (event.uncited_pages && event.uncited_pages.length) {
      msg.insertBefore(el("span", "badge", `Cites pages not retrieved: ${event.uncited_pages.join(", ")}`), answer);
    } else if (!event.cited_pages || !event.cited_pages.length) {
      msg.insertBefore(el("span", "badge", "No page citations"), answer);
    }
  }
  const t = event.timings || {};
  const parts = [];
  if (t.retrieval_s !== undefined) parts.push(`retrieval ${t.retrieval_s}s`);
  if (t.first_token_s) parts.push(`first token ${t.first_token_s}s`);
  if (t.generation_s) parts.push(`total ${t.generation_s}s`);
  meta.textContent = parts.join(" · ");
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = box.value.trim();
  if (!q || send.disabled) return;
  box.value = "";
  ask(q);
});
box.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    if (!send.disabled) ask(chip.textContent);
  })
);

// Documents panel and upload
const docsList = document.getElementById("docs");
const fileInput = document.getElementById("file");
const fileLabel = document.getElementById("file-label");
const uploadForm = document.getElementById("upload-form");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");

async function loadDocs() {
  try {
    const data = await (await fetch("/api/documents")).json();
    docsList.replaceChildren();
    if (!data.documents.length) docsList.appendChild(el("li", "muted", "No PDFs yet"));
    data.documents.forEach((d) => {
      const li = el("li", "", d.name + " ");
      li.appendChild(el("span", "", `${d.size_kb} KB`));
      docsList.appendChild(li);
    });
  } catch {
    docsList.replaceChildren(el("li", "muted", "Could not load documents"));
  }
}

fileInput.addEventListener("change", () => {
  fileLabel.textContent = fileInput.files[0] ? fileInput.files[0].name : "Choose a PDF";
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files[0]) return (uploadStatus.textContent = "Choose a PDF first.");
  const body = new FormData();
  body.append("file", fileInput.files[0]);
  uploadBtn.disabled = true;
  uploadStatus.textContent = "Chunking, embedding, and upserting to Pinecone...";
  try {
    const res = await fetch("/api/upload", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    uploadStatus.textContent = `Indexed ${data.name}: ${data.pages} pages, ${data.chunks} chunks.`;
    fileInput.value = "";
    fileLabel.textContent = "Choose a PDF";
    loadDocs();
  } catch (err) {
    uploadStatus.textContent = err.message || "Upload failed.";
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
