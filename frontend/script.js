const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
let threadId = sessionStorage.getItem("rag_thread_id") || crypto.randomUUID();
sessionStorage.setItem("rag_thread_id", threadId);

// =========================================
// THEME LOGIC
// =========================================
function initTheme() {
    const savedTheme = localStorage.getItem("theme") || "light";
    document.documentElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    updateThemeIcon(next);
}

function updateThemeIcon(theme) {
    const btn = document.getElementById("themeToggleBtn");
    if (!btn) return;
    // Simple Moon/Sun icons
    btn.innerHTML = theme === "light"
        ? `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>` // Sun
        : `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`; // Moon
}

// =========================================
// CHAT & UPLOAD LOGIC
// =========================================

// Handle Enter key to send
function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        ask();
    }
}

async function upload() {
    const fileInput = document.getElementById("files");
    const files = fileInput.files;
    const statusDiv = document.getElementById("uploadStatus");
    const progressContainer = document.querySelector(".progress-container");
    const progressBar = document.querySelector(".progress-bar");

    if (!files.length) {
        alert("Please select at least one file.");
        return;
    }

    // Validation
    for (let f of files) {
        if (f.size > MAX_FILE_SIZE) {
            alert(`File "${f.name}" is too large. Max allowed is 50MB.`);
            return;
        }
    }

    // Reset UI
    if (statusDiv) statusDiv.innerText = "Uploading...";
    if (progressContainer) {
        progressContainer.style.display = "block";
        progressBar.style.width = "0%";
    }

    const fd = new FormData();
    for (let f of files) fd.append("files", f);

    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            if (progressBar) progressBar.style.width = percent + "%";
        }
    });

    xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            try {
                const data = JSON.parse(xhr.responseText);
                if (statusDiv) {
                    statusDiv.innerText = data.message || "Upload complete!";
                    statusDiv.style.color = "var(--success, green)";
                }
                if (progressBar) progressBar.style.width = "100%";
                fileInput.value = "";

                // Add a system message to chat
                addMessage("ai", `✅ **System:** ${data.message || "Files processed successfully."}`);
            } catch (e) {
                if (statusDiv) statusDiv.innerText = "Error parsing server response.";
            }
        } else {
            if (statusDiv) statusDiv.innerText = "Upload failed.";
        }
    });

    xhr.addEventListener("error", () => {
        if (statusDiv) statusDiv.innerText = "Network error.";
    });

    xhr.open("POST", "/upload");
    xhr.send(fd);
}

async function ask() {
    const input = document.getElementById("question");
    const q = input.value.trim();
    if (!q) return;

    // Clear input
    input.value = "";

    // Add User Message
    addMessage("user", q);

    // Show thinking bubble
    const thinkingId = addMessage("ai", '<div class="typing-indicator">Thinking...</div>');

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: q, thread_id: threadId })
        });

        const data = await res.json();

        // Remove thinking bubble and add real answer
        removeMessage(thinkingId);

        let html = formatAnswer(data);
        addMessage("ai", html);

        saveToHistory(q, html);

    } catch (e) {
        removeMessage(thinkingId);
        addMessage("ai", "❌ **Error:** Could not connect to the server.");
    }
}

function formatAnswer(data) {
    let html = data.answer
        .replace(/\n/g, "<br>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>"); // Basic Markdown bold

    if (data.confidence > 0) {
        let label = "Low";
        if (data.confidence >= 0.7) label = "High";
        else if (data.confidence >= 0.5) label = "Medium";
        html += `<br><div class="confidence-badge">Confidence: ${label} (${Math.round(data.confidence * 100)}%)</div>`;
    }

    if (data.citations?.length) {
        html += "<div class='citation-box'><strong>Sources:</strong><ul>";
        data.citations.forEach(c => {
            html += `<li>${c.source} (Page ${c.page})</li>`;
        });
        html += "</ul></div>";
    }
    return html;
}

function summarize() {
    const input = document.getElementById("question");
    input.value = "Summarize the uploaded documents";
    ask();
}

// =========================================
// UI HELPERS
// =========================================
function addMessage(type, html) {
    const chatContainer = document.getElementById("chatContainer");
    const div = document.createElement("div");
    div.className = `message ${type}-message`;
    div.id = "msg-" + Date.now();

    div.innerHTML = `
        <div class="bubble ${type}-bubble glass">
            ${html}
        </div>
    `;

    chatContainer.appendChild(div);
    scrollToBottom();
    return div.id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    const container = document.getElementById("mainContent"); // The scrollable area
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
}

// =========================================
// HISTORY
// =========================================
function loadHistory() {
    const list = document.getElementById("historyList");
    if (!list) return;
    list.innerHTML = "";
    const history = JSON.parse(localStorage.getItem("rag_history") || "[]");

    history.forEach((item, index) => {
        const div = document.createElement("div");
        div.className = "history-item";
        div.innerText = item.query;
        div.onclick = () => loadSession(item);
        list.appendChild(div);
    });
}

function saveToHistory(query, answerHtml) {
    const history = JSON.parse(localStorage.getItem("rag_history") || "[]");
    history.unshift({ query, answerHtml, timestamp: Date.now() });
    if (history.length > 50) history.pop();
    localStorage.setItem("rag_history", JSON.stringify(history));
    loadHistory();
}

function loadSession(item) {
    // Clear chat and load just this exchange? Or append?
    // Let's clear for "session-like" feel
    document.getElementById("chatContainer").innerHTML = "";
    addMessage("user", item.query);
    addMessage("ai", item.answerHtml);
}

function newChat() {
    document.getElementById("chatContainer").innerHTML = "";
    addMessage("ai", "Hello! I am NexusGraph AI. Upload a document or ask me anything.");
    threadId = crypto.randomUUID();
    sessionStorage.setItem("rag_thread_id", threadId);
}

function clearHistory() {
    if (confirm("Delete all history?")) {
        localStorage.removeItem("rag_history");
        loadHistory();
        newChat();
    }
}

// Init
window.addEventListener("DOMContentLoaded", () => {
    initTheme();
    loadHistory();
    newChat(); // Show welcome message
});
