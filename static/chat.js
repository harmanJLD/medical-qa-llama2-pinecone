const chat = document.getElementById("chat");
const form = document.getElementById("ask-form");
const box = document.getElementById("question");
const send = document.getElementById("send");

function bubble(cls, text) {
  const div = document.createElement("div");
  div.className = `bubble ${cls}`;
  div.textContent = text;
  chat.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
  return div;
}

function addSources(div, sources) {
  if (!sources || !sources.length) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `Sources (${sources.length})`;
  details.appendChild(summary);
  const list = document.createElement("ul");
  sources.forEach((s) => {
    const li = document.createElement("li");
    const label = document.createElement("b");
    label.textContent = `${s.source}, page ${s.page}: `;
    li.appendChild(label);
    li.appendChild(document.createTextNode(`${s.excerpt}...`));
    list.appendChild(li);
  });
  details.appendChild(list);
  div.appendChild(details);
}

async function submit() {
  const question = box.value.trim();
  if (!question) return;
  bubble("me", question);
  box.value = "";
  send.disabled = true;
  const pending = bubble("bot pending", "Thinking...");
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    pending.remove();
    if (!res.ok) {
      bubble("bot error", data.error || "Something went wrong.");
    } else {
      addSources(bubble("bot", data.answer), data.sources);
    }
  } catch (err) {
    pending.remove();
    bubble("bot error", "Could not reach the server.");
  } finally {
    send.disabled = false;
    box.focus();
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
box.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
});
