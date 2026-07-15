const chatWindow = document.getElementById('chat-window');
const userInput  = document.getElementById('user-input');
const sendBtn    = document.getElementById('send-btn');
const statusDot  = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

const API_URL = '/api/chat';
const MAX_HISTORY = 20;

let history = [];
let isSending = false;

function appendMessage(role, text) {
  const isUser = role === 'user';
  const div = document.createElement('div');
  div.className = `message max-w-[72%] text-[13px] leading-relaxed ${
    isUser
      ? 'self-end bg-[#111111] border border-[#222222] px-3.5 py-2.5'
      : 'self-start border-l-2 border-[#f5c400] px-3.5 py-2.5 text-[#c8c8c8]'
  }`;

  const label = document.createElement('div');
  label.className = `text-[10px] tracking-[0.2em] mb-1.5 ${
    isUser ? 'text-[#555555]' : 'text-[#f5c400]'
  }`;
  label.textContent = isUser ? 'YOU' : 'TEO';

  const body = document.createElement('div');
  body.textContent = text;

  div.appendChild(label);
  div.appendChild(body);
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}

function showThinking() {
  const div = document.createElement('div');
  div.id = 'thinking';
  div.className = 'thinking self-start border-l-2 border-[#f5c400] px-3.5 py-2.5 text-[13px] text-[#555555]';
  div.textContent = 'TEO IS THINKING...';
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeThinking() {
  document.getElementById('thinking')?.remove();
}

function setStatus(online, label) {
  statusDot.className = `status-dot ${online ? 'online' : 'offline'}`;
  statusText.textContent = label;
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('unhealthy');
    const data = await res.json();
    const label = data.model_loaded ? 'MODEL LIVE' : 'PLACEHOLDER';
    setStatus(true, label);
  } catch {
    setStatus(false, 'OFFLINE');
  }
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isSending) return;

  isSending = true;
  userInput.value = '';
  sendBtn.disabled = true;

  appendMessage('user', text);
  history.push({ role: 'user', content: text });
  if (history.length > MAX_HISTORY) history = history.slice(-MAX_HISTORY);

  showThinking();

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history })
    });

    removeThinking();

    if (!res.ok) {
      appendMessage('teo', `[server error ${res.status} — is uvicorn running?]`);
      isSending = false;
      sendBtn.disabled = false;
      userInput.focus();
      return;
    }

    const data = await res.json();
    appendMessage('teo', data.response);
    history.push({ role: 'teo', content: data.response });
    if (history.length > MAX_HISTORY) history = history.slice(-MAX_HISTORY);
    checkHealth();

  } catch {
    removeThinking();
    appendMessage('teo', '[connection failed — start backend: uvicorn main:app --reload]');
    setStatus(false, 'OFFLINE');
  }

  isSending = false;
  sendBtn.disabled = false;
  userInput.focus();
}

sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

appendMessage('teo', 'Accepted. DEPLOY. DEFEND. EVOLVE. — Talk to Teo.');
checkHealth();
setInterval(checkHealth, 30000);
