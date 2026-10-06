const form = document.getElementById("search-form");
const input = document.getElementById("query");
const log = document.getElementById("log");
const button = form.querySelector("button");

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  log.textContent = "";
  button.disabled = true;

  const source = new EventSource(`/search?q=${encodeURIComponent(query)}`);

  source.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "token") {
      log.append(data.text);
    } else if (data.type === "tool_call") {
      const span = document.createElement("span");
      span.className = "tool-call";
      span.textContent = `→ searching your library for: ${data.payload?.query ?? ""}`;
      log.appendChild(span);
    } else if (data.type === "error") {
      const span = document.createElement("span");
      span.className = "error";
      span.textContent = data.text;
      log.appendChild(span);
      source.close();
      button.disabled = false;
    } else if (data.type === "done") {
      if (data.text) log.append(data.text);
      source.close();
      button.disabled = false;
    }
  };

  source.onerror = () => {
    source.close();
    button.disabled = false;
  };
});
