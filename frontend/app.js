/* ═══════════════════════════════════════════════
   LEX AI — app.js  |  Frontend Logic
   Connects to FastAPI backend at localhost:8000
═══════════════════════════════════════════════ */

// ── CONFIG ──────────────────────────────────────────────────────────────────
let API_URL        = localStorage.getItem('lexai_api_url') || 'http://localhost:8000';
let SESSION_ID     = generateUUID();
let contextText    = null;
let contractId     = null;
let savedItems     = JSON.parse(localStorage.getItem('lexai_saved') || '[]');
let currentTab     = 'chat';
let isSending      = false;

// ── INIT ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkAPIStatus();
  loadSettings();
  renderJudgments();
  renderSavedItems();
  selectTemplate('nda');
  setInterval(checkAPIStatus, 30000);

  // restore display name
  const name = localStorage.getItem('lexai_display_name');
  if (name) {
    document.querySelector('.user-name').textContent = name;
    document.querySelector('.user-avatar').textContent = name.charAt(0).toUpperCase();
  }
});

// ── UUID ─────────────────────────────────────────────────────────────────────
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : r & 0x3 | 0x8).toString(16);
  });
}

// ── API STATUS ────────────────────────────────────────────────────────────────
async function checkAPIStatus() {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  dot.className = 'status-dot checking';
  text.textContent = 'Connecting…';
  try {
    const r = await fetch(`${API_URL}/`, { signal: AbortSignal.timeout(4000) });
    if (r.ok) {
      dot.className = 'status-dot online';
      text.textContent = 'API Online';
    } else {
      throw new Error();
    }
  } catch {
    dot.className = 'status-dot offline';
    text.textContent = 'API Offline';
  }
}

// ── TAB SWITCHING ─────────────────────────────────────────────────────────────
const TAB_MAP = {
  chat:      { panel: 'tabChat',      nav: 'navChat',      title: 'AI Legal Assistant' },
  analyze:   { panel: 'tabAnalyze',   nav: 'navAnalyze',   title: 'Contract Analysis' },
  landmarks: { panel: 'tabLandmarks', nav: 'navLandmarks',  title: 'Landmark Judgments' },
  draft:     { panel: 'tabDraft',     nav: 'navDraft',      title: 'Draft Generator' },
  saved:     { panel: 'tabSaved',     nav: 'navSaved',      title: 'Saved Items' },
};

function switchTab(tab) {
  if (!TAB_MAP[tab]) return;
  currentTab = tab;
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(TAB_MAP[tab].panel).classList.add('active');
  document.getElementById(TAB_MAP[tab].nav).classList.add('active');
  document.getElementById('pageTitle').textContent = TAB_MAP[tab].title;
  if (tab === 'saved') renderSavedItems();
}

// ── FILE UPLOAD ───────────────────────────────────────────────────────────────
async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const label = document.getElementById('uploadLabel');
  label.textContent = `Uploading ${file.name}…`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const r = await fetch(`${API_URL}/contracts/upload`, { method: 'POST', body: formData });
    if (r.status === 201) {
      const data = await r.json();
      contractId  = data.contract_id;
      contextText = null;
      label.textContent = `✅ ${file.name}`;
      showContextBadge(`📄 ${file.name}`);
      showToast(`Document "${file.name}" uploaded!`, 'success');
    } else {
      label.textContent = 'Drop PDF / DOCX here';
      const err = await r.json().catch(() => ({}));
      showToast(err.detail || 'Upload failed', 'error');
    }
  } catch {
    label.textContent = 'Drop PDF / DOCX here';
    showToast('Cannot reach backend — check API URL in settings', 'error');
  }
}

// Drag‑drop
const uploadArea = document.getElementById('uploadArea');
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.style.borderColor = 'var(--accent)'; });
uploadArea.addEventListener('dragleave', () => uploadArea.style.borderColor = '');
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.style.borderColor = '';
  const file = e.dataTransfer.files[0];
  if (file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    document.getElementById('fileInput').files = dt.files;
    handleFileUpload({ target: { files: [file] } });
  }
});

// ── TEXT CONTEXT ──────────────────────────────────────────────────────────────
function loadTextContext() {
  const text = document.getElementById('pasteArea').value.trim();
  if (!text) { showToast('Please paste some contract text first', 'error'); return; }
  contextText = text;
  contractId  = null;
  showContextBadge('📝 Text loaded');
  showToast('Contract text loaded into memory!', 'success');
}

function showContextBadge(label) {
  const badge = document.getElementById('contextBadge');
  document.getElementById('contextLabel').textContent = label;
  badge.style.display = 'flex';
}

// ── CHAT ──────────────────────────────────────────────────────────────────────
function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function quickPrompt(text) {
  document.getElementById('chatInput').value = text;
  sendMessage();
}

async function sendMessage() {
  if (isSending) return;
  const input = document.getElementById('chatInput');
  const msg   = input.value.trim();
  if (!msg) return;

  isSending = true;
  const btn = document.getElementById('sendBtn');
  btn.disabled = true;

  // Remove welcome screen if present
  const welcome = document.querySelector('.welcome-screen');
  if (welcome) welcome.remove();

  appendMessage('user', msg);
  input.value = '';
  input.style.height = 'auto';

  const typingId = appendTyping();

  const payload = {
    message: msg,
    session_id: SESSION_ID,
    extra_context: {},
  };
  if (contractId) payload.contract_id = contractId;
  if (contextText) {
    payload.extra_context.text = contextText;
    contextText = null; // only send once
  }

  try {
    const r = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(120000),
    });

    removeTyping(typingId);

    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      appendMessage('assistant', `❌ **Error:** ${err.detail || 'Request failed'}`, 'GENERAL');
    } else {
      const data = await r.json();
      const meta   = data.meta || {};
      const status = meta.status || 'completed';
      const intent = data.intent || 'GENERAL';

      if (status === 'processing' && meta.task_id) {
        await pollTask(meta.task_id, intent, meta);
      } else {
        const reply = buildDisplayMessage(data, intent);
        appendMessage('assistant', reply, intent, meta);
      }
    }
  } catch (err) {
    removeTyping(typingId);
    appendMessage('assistant',
      '❌ **Connection failed.** Make sure the API server is running at ' + API_URL,
      'GENERAL');
  }

  isSending = false;
  btn.disabled = false;
  input.focus();
}

async function pollTask(taskId, intent, originalMeta) {
  const typingId = appendTyping();
  const MAX_POLLS = 300;
  for (let i = 0; i < MAX_POLLS; i++) {
    await sleep(1500);
    try {
      const r = await fetch(`${API_URL}/chat/result/${taskId}`, { signal: AbortSignal.timeout(5000) });
      if (!r.ok) continue;
      const data = await r.json();
      if (data.status === 'completed') {
        removeTyping(typingId);
        const res  = data.result || data;
        const meta = { ...originalMeta, ...(res.meta || {}), status: 'completed', task_id: taskId };
        appendMessage('assistant', buildDisplayMessage(res, intent), intent, meta);
        return;
      }
      if (data.status === 'failed') {
        removeTyping(typingId);
        appendMessage('assistant', `❌ **Task failed:** ${data.error || 'Unknown error'}`, 'GENERAL');
        return;
      }
    } catch { /* retry */ }
  }
  removeTyping(typingId);
  appendMessage('assistant', '⚠️ Request timed out. Please try again.', 'GENERAL');
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function buildDisplayMessage(result, intent = '') {
  const data = result?.data || {};
  const baseText = result?.message || result?.response || '';

  if (intent === 'SUMMARY' || data.summary || data.key_points || data.explanation) {
    const lines = [];
    if (typeof data.summary === 'string' && data.summary.trim()) {
      lines.push(data.summary.trim());
    }

    if (Array.isArray(data.key_points) && data.key_points.length > 0) {
      lines.push('');
      lines.push('**Key points**');
      for (const point of data.key_points) {
        lines.push(`- ${point}`);
      }
    }

    if (typeof data.explanation === 'string' && data.explanation.trim()) {
      lines.push('');
      lines.push('**Plain-English explanation**');
      lines.push(data.explanation.trim());
    }

    if (lines.length > 0) {
      return lines.join('\n');
    }
  }

  if (intent === 'CLAUSE_MAP' || data.clause_groups || data.unique_clause_types !== undefined) {
    const lines = [];

    if (baseText) {
      lines.push(baseText.trim());
      lines.push('');
    }

    const total = Number.isFinite(data.total_paragraphs) ? data.total_paragraphs : 0;
    const unique = Number.isFinite(data.unique_clause_types) ? data.unique_clause_types : 0;
    lines.push(`**Coverage:** ${unique} clause types across ${total} paragraphs.`);

    const groups = data.clause_groups && typeof data.clause_groups === 'object' ? data.clause_groups : {};
    const entries = Object.entries(groups);

    if (entries.length > 0) {
      lines.push('');
      lines.push('**Clause groups**');
      for (const [, group] of entries.slice(0, 10)) {
        const label = group?.label || 'Unknown';
        const count = group?.count ?? 0;
        const avg = typeof group?.avg_confidence === 'number' ? group.avg_confidence : null;
        const avgText = avg !== null ? ` (avg conf ${Math.round(avg * 100)}%)` : '';
        lines.push(`- ${label}: ${count}${avgText}`);
      }
    }

    if (data.error) {
      lines.push('');
      lines.push(`⚠️ ${data.error}`);
    }

    return lines.join('\n');
  }

  return baseText || 'Analysis complete.';
}

function appendMessage(role, text, intent, meta) {
  const container = document.getElementById('chatMessages');
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'user' ? 'A' : '⚖';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  // Markdown-lite
  const formatted = markdownLite(text);
  bubble.innerHTML = formatted;

  // Meta row
  if (intent && role === 'assistant') {
    const metaRow = document.createElement('div');
    metaRow.className = 'msg-meta';
    const tag = document.createElement('span');
    tag.className = `intent-tag ${intent}`;
    tag.textContent = intent.replace('_', ' ');
    metaRow.appendChild(tag);
    if (meta?.task_id) {
      const tid = document.createElement('span');
      tid.textContent = `Task: ${meta.task_id.slice(0, 8)}…`;
      metaRow.appendChild(tid);
    }
    bubble.appendChild(metaRow);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  container.appendChild(wrap);
  container.scrollTop = container.scrollHeight;
  return wrap;
}

let typingCounter = 0;
function appendTyping() {
  const id = 'typing-' + (++typingCounter);
  const container = document.getElementById('chatMessages');
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.id = id;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = '⚖';

  const indicator = document.createElement('div');
  indicator.className = 'typing-indicator';
  indicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

  wrap.appendChild(avatar);
  wrap.appendChild(indicator);
  container.appendChild(wrap);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function clearChat() {
  SESSION_ID = generateUUID();
  contextText = null;
  contractId  = null;
  document.getElementById('contextBadge').style.display = 'none';
  document.getElementById('pasteArea').value = '';
  document.getElementById('uploadLabel').textContent = 'Drop PDF / DOCX here';

  const container = document.getElementById('chatMessages');
  container.innerHTML = `
    <div class="welcome-screen">
      <div class="welcome-icon">⚖️</div>
      <h2>Welcome to Lex AI</h2>
      <p>Your intelligent legal assistant. Upload a contract in the sidebar, then ask me to analyze risks, summarize key points, or map all clause types.</p>
      <div class="quick-actions">
        <button class="quick-btn" onclick="quickPrompt('Summarize this contract')">📄 Summarize</button>
        <button class="quick-btn" onclick="quickPrompt('Find all legal risks')">⚠️ Find Risks</button>
        <button class="quick-btn" onclick="quickPrompt('List all clause types')">🗂️ Map Clauses</button>
        <button class="quick-btn" onclick="quickPrompt('What are the termination terms?')">🔍 Review Terms</button>
      </div>
    </div>`;
  showToast('Chat cleared', 'success');
}

// ── MARKDOWN LITE ─────────────────────────────────────────────────────────────
function markdownLite(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code style="background:var(--bg-input);padding:1px 5px;border-radius:4px;font-size:12px;">$1</code>')
    .replace(/^#{1,3}\s+(.+)$/gm, '<strong>$1</strong>')
    .replace(/^[-•]\s+(.+)$/gm, '• $1')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^/, '<p>').replace(/$/, '</p>');
}

// ── ANALYZE TAB ───────────────────────────────────────────────────────────────
async function runAnalysis(type) {
  if (!contextText && !contractId) {
    showToast('Please upload a contract first', 'error');
    return;
  }

  const resultDiv  = document.getElementById('analysisResult');
  const contentDiv = document.getElementById('resultContent');
  const titleEl    = document.getElementById('resultTitle');

  resultDiv.style.display = 'block';
  contentDiv.innerHTML = '<div class="typing-indicator" style="margin:0"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';

  const msgMap = { risk: 'Find all legal risks', summary: 'Summarize this contract', clauses: 'List all clause types' };
  const titleMap = { risk: '⚠️ Risk Analysis', summary: '📄 Summary', clauses: '🗂️ Clause Map' };
  titleEl.textContent = titleMap[type] || 'Analysis';

  // Route via chat endpoint
  const payload = {
    message: msgMap[type],
    session_id: SESSION_ID,
    extra_context: {},
  };
  if (contractId) payload.contract_id = contractId;
  if (contextText) payload.extra_context.text = contextText;

  try {
    const r    = await fetch(`${API_URL}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await r.json();
    const meta = data.meta || {};

    let result = data;
    if (meta.status === 'processing' && meta.task_id) {
      contentDiv.innerHTML = '<em style="color:var(--text-muted)">⏳ Running analysis in background…</em>';
      for (let i = 0; i < 200; i++) {
        await sleep(1500);
        const pr = await fetch(`${API_URL}/chat/result/${meta.task_id}`);
        const pd = await pr.json();
        if (pd.status === 'completed') { result = pd.result || pd; break; }
        if (pd.status === 'failed')    { contentDiv.innerHTML = `<em style="color:var(--risk-high)">❌ ${pd.error}</em>`; return; }
      }
    }

    contentDiv.innerHTML = markdownLite(buildDisplayMessage(result, type === 'summary' ? 'SUMMARY' : ''));
    showToast(titleMap[type] + ' complete!', 'success');
  } catch {
    contentDiv.innerHTML = '<em style="color:var(--risk-high)">❌ Failed — check API connection</em>';
  }
}

// ── EXPORT PDF ────────────────────────────────────────────────────────────────
function exportPDF() {
  const messages = document.querySelectorAll('.msg-bubble');
  if (!messages.length) { showToast('Nothing to export', 'error'); return; }
  const content = Array.from(messages).map(m => m.innerText).join('\n\n---\n\n');
  const blob = new Blob([content], { type: 'text/plain' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'lexai-analysis.txt';
  a.click();
  showToast('Exported as text file', 'success');
}

// ── SAVED ITEMS ───────────────────────────────────────────────────────────────
function saveResultItem() {
  const title   = document.getElementById('resultTitle').textContent;
  const content = document.getElementById('resultContent').innerText;
  if (!content.trim()) { showToast('Nothing to save', 'error'); return; }
  const item = { id: generateUUID(), title, content, date: new Date().toLocaleDateString() };
  savedItems.push(item);
  localStorage.setItem('lexai_saved', JSON.stringify(savedItems));
  renderSavedItems();
  showToast('Saved!', 'success');
}

function renderSavedItems() {
  const list = document.getElementById('savedList');
  if (!savedItems.length) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📂</div>
        <p>No saved items yet</p>
        <span>Run an analysis and save the results to see them here</span>
      </div>`;
    return;
  }
  list.innerHTML = savedItems.map(item => `
    <div class="saved-item">
      <div>
        <div style="font-weight:600;font-size:14px;margin-bottom:4px">${escapeHtml(item.title)}</div>
        <div style="font-size:12px;color:var(--text-muted)">${item.date}</div>
        <div style="font-size:13px;color:var(--text-secondary);margin-top:8px">${escapeHtml(item.content.slice(0, 160))}${item.content.length > 160 ? '…' : ''}</div>
      </div>
      <button class="btn-ghost small" onclick="deleteSaved('${item.id}')">🗑</button>
    </div>`).join('');
}

function deleteSaved(id) {
  savedItems = savedItems.filter(i => i.id !== id);
  localStorage.setItem('lexai_saved', JSON.stringify(savedItems));
  renderSavedItems();
  showToast('Item removed', 'success');
}

// ── LANDMARK JUDGMENTS ────────────────────────────────────────────────────────
const JUDGMENTS = [
  { icon:'🏛️', color:'#d97706', tag:'Constitutional', tagColor:'rgba(217,119,6,0.15)', tagFg:'#d97706',
    title:'Bachan Singh v. State of Punjab', citation:'1980 SCC (2) 684',
    summary:'Upheld death penalty as constitutional but restricted it to "rarest of rare" cases.' },
  { icon:'📜', color:'#ef4444', tag:'Criminal Law', tagColor:'rgba(239,68,68,0.15)', tagFg:'#ef4444',
    title:'Section 498A: Cruelty — Landmark Interpretation', citation:'Indian Penal Code, 1860',
    summary:'Supreme Court guidelines on misuse and proper application of IPC Section 498A in matrimonial disputes.' },
  { icon:'⚖️', color:'#3b82f6', tag:'Constitutional', tagColor:'rgba(59,130,246,0.15)', tagFg:'#60a5fa',
    title:'Article 14: Equality Before the Law', citation:'Constitution of India, 1950',
    summary:'Right to equality — the state shall not deny equality before law or equal protection of laws.' },
  { icon:'🛡️', color:'#22c55e', tag:'Fundamental Rights', tagColor:'rgba(34,197,94,0.15)', tagFg:'#22c55e',
    title:'Article 21: Protection of Life and Liberty', citation:'Constitution of India, 1950',
    summary:'No person shall be deprived of life or personal liberty except according to procedure established by law.' },
  { icon:'💼', color:'#a855f7', tag:'Contract Law', tagColor:'rgba(168,85,247,0.15)', tagFg:'#a855f7',
    title:'Carlill v. Carbolic Smoke Ball Co.', citation:'[1893] 1 QB 256',
    summary:'Seminal case on unilateral contracts — an advertisement can constitute a binding offer.' },
  { icon:'🏗️', color:'#f59e0b', tag:'Property Law', tagColor:'rgba(245,158,11,0.15)', tagFg:'#f59e0b',
    title:'Transfer of Property Act — Section 53A', citation:'Transfer of Property Act, 1882',
    summary:'Part performance doctrine protects a transferee who acts on an unregistered agreement.' },
  { icon:'🔒', color:'#06b6d4', tag:'Data & Privacy', tagColor:'rgba(6,182,212,0.15)', tagFg:'#06b6d4',
    title:'K.S. Puttaswamy v. Union of India', citation:'(2017) 10 SCC 1',
    summary:'Landmark 9-judge bench ruling — Right to Privacy is a fundamental right under Article 21.' },
  { icon:'📊', color:'#ec4899', tag:'Company Law', tagColor:'rgba(236,72,153,0.15)', tagFg:'#ec4899',
    title:'Salomon v. Salomon & Co. Ltd.', citation:'[1897] AC 22',
    summary:'Corporate personality is distinct from its members — the veil of incorporation principle.' },
  { icon:'🤝', color:'#14b8a6', tag:'Labour Law', tagColor:'rgba(20,184,166,0.15)', tagFg:'#14b8a6',
    title:'Vishaka v. State of Rajasthan', citation:'(1997) 6 SCC 241',
    summary:'Guidelines for prevention of sexual harassment at workplace — precursor to POSH Act.' },
  { icon:'📋', color:'#8b5cf6', tag:'Tort Law', tagColor:'rgba(139,92,246,0.15)', tagFg:'#8b5cf6',
    title:'Donoghue v. Stevenson', citation:'[1932] AC 562',
    summary:'Established the "neighbour principle" and duty of care in modern tort law / negligence.' },
];

let filteredJudgments = [...JUDGMENTS];

function renderJudgments(list) {
  const grid = document.getElementById('judgmentsGrid');
  const src   = list || filteredJudgments;
  if (!src.length) { grid.innerHTML = '<div class="empty-state"><div class="empty-icon">🔍</div><p>No results found</p></div>'; return; }
  grid.innerHTML = src.map(j => `
    <div class="judgment-card" onclick="showJudgmentDetail('${escapeAttr(j.title)}')">
      <div class="judgment-icon" style="background:${j.tagColor}">
        <span style="font-size:18px">${j.icon}</span>
      </div>
      <div class="judgment-body">
        <h4>${escapeHtml(j.title)}</h4>
        <div class="citation">${escapeHtml(j.citation)}</div>
        <span class="tag" style="background:${j.tagColor};color:${j.tagFg}">${escapeHtml(j.tag)}</span>
        <div style="font-size:12px;color:var(--text-muted);margin-top:8px;line-height:1.5">${escapeHtml(j.summary)}</div>
      </div>
    </div>`).join('');
}

function filterJudgments() {
  const q = document.getElementById('judgmentSearch').value.toLowerCase();
  filteredJudgments = JUDGMENTS.filter(j =>
    j.title.toLowerCase().includes(q) ||
    j.citation.toLowerCase().includes(q) ||
    j.tag.toLowerCase().includes(q) ||
    j.summary.toLowerCase().includes(q)
  );
  renderJudgments();
}

function showJudgmentDetail(title) {
  const j = JUDGMENTS.find(x => x.title === title);
  if (!j) return;
  showToast(`Opening: ${j.title.slice(0, 40)}…`);
  // Future: open a detail modal
}

// ── DRAFT GENERATOR ───────────────────────────────────────────────────────────
const TEMPLATES = {
  nda: {
    label: 'Non-Disclosure Agreement (NDA)',
    fields: [
      { id: 'party1',    label: 'Disclosing Party (Name)', placeholder: 'ABC Corporation Ltd.' },
      { id: 'party2',    label: 'Receiving Party (Name)', placeholder: 'XYZ Consultants Pvt. Ltd.' },
      { id: 'purpose',   label: 'Purpose of Disclosure', placeholder: 'Evaluation of potential business partnership' },
      { id: 'duration',  label: 'Confidentiality Period', placeholder: '2 years' },
      { id: 'governing', label: 'Governing Law / Jurisdiction', placeholder: 'Mumbai, Maharashtra, India' },
      { id: 'date',      label: 'Agreement Date', placeholder: '16 April 2026', type: 'date' },
    ],
    generate: (f) => `NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is entered into as of ${f.date || '[Date]'} between:

DISCLOSING PARTY: ${f.party1 || '[Party 1]'}
RECEIVING PARTY:  ${f.party2 || '[Party 2]'}

1. PURPOSE
The parties wish to explore ${f.purpose || '[purpose]'} and may disclose confidential information to each other.

2. CONFIDENTIAL INFORMATION
"Confidential Information" means any non-public information disclosed by either party.

3. OBLIGATIONS
The Receiving Party agrees to:
(a) Keep all Confidential Information strictly confidential;
(b) Not disclose it to any third party without prior written consent;
(c) Use it solely for the Purpose stated above.

4. TERM
This Agreement shall remain in force for ${f.duration || '[duration]'} from the date of execution.

5. GOVERNING LAW
This Agreement shall be governed by the laws of ${f.governing || '[jurisdiction]'}.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

_______________________          _______________________
${f.party1 || '[Party 1]'}              ${f.party2 || '[Party 2]'}
Authorized Signatory              Authorized Signatory`,
  },
  service: {
    label: 'Service Agreement',
    fields: [
      { id: 'client',    label: 'Client Name', placeholder: 'ABC Company Ltd.' },
      { id: 'provider',  label: 'Service Provider Name', placeholder: 'Tech Solutions Pvt. Ltd.' },
      { id: 'services',  label: 'Description of Services', placeholder: 'Software development and maintenance' },
      { id: 'fees',      label: 'Service Fees', placeholder: '₹5,00,000 per month' },
      { id: 'term',      label: 'Contract Term', placeholder: '12 months' },
      { id: 'notice',    label: 'Termination Notice Period', placeholder: '30 days' },
    ],
    generate: (f) => `SERVICE AGREEMENT

This Service Agreement is entered into between:
CLIENT:   ${f.client || '[Client]'}
PROVIDER: ${f.provider || '[Provider]'}

1. SERVICES
Provider shall perform: ${f.services || '[services]'}

2. FEES
Client shall pay ${f.fees || '[fees]'} for the services rendered.

3. TERM
This Agreement shall commence on the date of signing and continue for ${f.term || '[term]'}.

4. TERMINATION
Either party may terminate this Agreement with ${f.notice || '30 days'} written notice.

5. LIMITATION OF LIABILITY
Provider's total liability shall not exceed the fees paid in the preceding 3 months.`,
  },
  employment: {
    label: 'Employment Contract',
    fields: [
      { id: 'employer',    label: 'Employer (Company)', placeholder: 'XYZ Corporation' },
      { id: 'employee',    label: 'Employee Name', placeholder: 'John Doe' },
      { id: 'designation', label: 'Designation / Role', placeholder: 'Senior Software Engineer' },
      { id: 'salary',      label: 'Annual Salary (CTC)', placeholder: '₹12,00,000 per annum' },
      { id: 'startDate',   label: 'Start Date', placeholder: '1 May 2026', type: 'date' },
      { id: 'probation',   label: 'Probation Period', placeholder: '6 months' },
    ],
    generate: (f) => `EMPLOYMENT CONTRACT

This Employment Agreement is entered into between:
EMPLOYER: ${f.employer || '[Employer]'}
EMPLOYEE: ${f.employee || '[Employee]'}

1. POSITION
Employee is hired as ${f.designation || '[Designation]'}, effective ${f.startDate || '[Start Date]'}.

2. COMPENSATION
Annual CTC: ${f.salary || '[Salary]'}. Payable monthly in arrears.

3. PROBATION PERIOD
Employee shall serve a probation period of ${f.probation || '6 months'}.

4. WORKING HOURS
Employee shall work 5 days a week, 8 hours per day.

5. CONFIDENTIALITY
Employee agrees to maintain strict confidentiality of all company information.`,
  },
  freelance: {
    label: 'Freelancer Agreement',
    fields: [
      { id: 'client',    label: 'Client Name', placeholder: 'XYZ Startup' },
      { id: 'freelancer',label: 'Freelancer Name', placeholder: 'Jane Doe' },
      { id: 'project',   label: 'Project Description', placeholder: 'Website redesign and development' },
      { id: 'rate',      label: 'Rate / Payment', placeholder: '₹80,000 fixed fee' },
      { id: 'deadline',  label: 'Project Deadline', placeholder: '60 days from signing' },
    ],
    generate: (f) => `FREELANCE AGREEMENT

CLIENT:     ${f.client || '[Client]'}
FREELANCER: ${f.freelancer || '[Freelancer]'}

1. PROJECT SCOPE
Freelancer agrees to complete: ${f.project || '[project description]'}

2. PAYMENT
Total payment: ${f.rate || '[amount]'}. 50% upfront, 50% on delivery.

3. TIMELINE
Project to be delivered within ${f.deadline || '[deadline]'}.

4. INTELLECTUAL PROPERTY
Upon final payment, all work product shall be the exclusive property of the Client.

5. INDEPENDENT CONTRACTOR
Freelancer is an independent contractor, not an employee of Client.`,
  },
  lease: {
    label: 'Lease Agreement',
    fields: [
      { id: 'lessor',    label: 'Lessor (Owner)', placeholder: 'Property Owner Name' },
      { id: 'lessee',    label: 'Lessee (Tenant)', placeholder: 'Tenant Name' },
      { id: 'property',  label: 'Property Address', placeholder: 'Flat No. 501, ABC Tower, Mumbai - 400001' },
      { id: 'rent',      label: 'Monthly Rent', placeholder: '₹25,000 per month' },
      { id: 'deposit',   label: 'Security Deposit', placeholder: '₹75,000 (3 months)' },
      { id: 'term',      label: 'Lease Term', placeholder: '11 months' },
    ],
    generate: (f) => `LEASE AGREEMENT

LESSOR: ${f.lessor || '[Owner]'}
LESSEE: ${f.lessee || '[Tenant]'}

1. PREMISES
Lessor hereby leases: ${f.property || '[Property Address]'}

2. TERM
The lease shall be for ${f.term || '[term]'} commencing on the date of signing.

3. RENT
Monthly rent: ${f.rent || '[rent]'}, payable by 5th of each month.

4. SECURITY DEPOSIT
Lessee shall pay a refundable security deposit of ${f.deposit || '[deposit]'}.

5. MAINTENANCE
Lessee shall maintain the property in good condition and bear minor repair costs.`,
  },
  partnership: {
    label: 'Partnership Deed',
    fields: [
      { id: 'firm',      label: 'Firm Name', placeholder: 'ABC & Associates' },
      { id: 'partner1',  label: 'Partner 1 Name', placeholder: 'Rahul Sharma' },
      { id: 'partner2',  label: 'Partner 2 Name', placeholder: 'Priya Mehta' },
      { id: 'business',  label: 'Nature of Business', placeholder: 'Legal consultancy services' },
      { id: 'capital',   label: 'Total Capital Contribution', placeholder: '₹10,00,000' },
      { id: 'profit',    label: 'Profit-Sharing Ratio', placeholder: '50:50' },
    ],
    generate: (f) => `PARTNERSHIP DEED

FIRM NAME: ${f.firm || '[Firm Name]'}
PARTNERS:  ${f.partner1 || '[Partner 1]'} and ${f.partner2 || '[Partner 2]'}

1. BUSINESS
The partners shall carry on the business of ${f.business || '[business]'}.

2. CAPITAL
Total capital: ${f.capital || '[capital]'}, contributed equally by both partners.

3. PROFIT & LOSS SHARING
Profits and losses shall be shared in the ratio of ${f.profit || '50:50'}.

4. MANAGEMENT
All major decisions require the consent of both partners.

5. DISSOLUTION
The partnership may be dissolved by mutual written consent of all partners.`,
  },
};

let currentTemplate = 'nda';

function selectTemplate(key) {
  currentTemplate = key;
  document.querySelectorAll('.template-item').forEach(el => el.classList.remove('active'));
  const btn = document.querySelector(`.template-item[onclick="selectTemplate('${key}')"]`);
  if (btn) btn.classList.add('active');

  const tmpl   = TEMPLATES[key];
  const fields = document.getElementById('draftFields');
  const output = document.getElementById('draftOutput');
  output.style.display = 'none';

  fields.innerHTML = tmpl.fields.map(f => `
    <div class="field-group">
      <label>${escapeHtml(f.label)}</label>
      <input type="${f.type || 'text'}" id="draft_${f.id}" placeholder="${escapeHtml(f.placeholder)}" class="setting-input" />
    </div>`).join('');
}

function generateDraft() {
  const tmpl  = TEMPLATES[currentTemplate];
  const values = {};
  tmpl.fields.forEach(f => {
    const el = document.getElementById('draft_' + f.id);
    values[f.id] = el ? el.value.trim() : '';
  });
  const text   = tmpl.generate(values);
  const output = document.getElementById('draftOutput');
  output.textContent = text;
  output.style.display = 'block';
  showToast('Draft generated!', 'success');
}

// ── SETTINGS ─────────────────────────────────────────────────────────────────
function openSettings() {
  document.getElementById('settingsModal').classList.add('open');
  document.getElementById('apiUrlInput').value     = API_URL;
  document.getElementById('displayNameInput').value = localStorage.getItem('lexai_display_name') || 'Adv. Aryan';
}

function closeSettings(event) {
  if (!event || event.target === document.getElementById('settingsModal')) {
    document.getElementById('settingsModal').classList.remove('open');
  }
}

function saveSettings() {
  const url  = document.getElementById('apiUrlInput').value.trim();
  const key  = document.getElementById('openaiKeyInput').value.trim();
  const name = document.getElementById('displayNameInput').value.trim();

  if (url) { API_URL = url; localStorage.setItem('lexai_api_url', url); }
  if (key) localStorage.setItem('lexai_openai_key', key);
  if (name) {
    localStorage.setItem('lexai_display_name', name);
    document.querySelector('.user-name').textContent    = name;
    document.querySelector('.user-avatar').textContent  = name.charAt(0).toUpperCase();
  }

  closeSettings();
  checkAPIStatus();
  showToast('Settings saved!', 'success');
}

function loadSettings() {
  const key = localStorage.getItem('lexai_openai_key');
  if (key) document.getElementById('openaiKeyInput').value = key;
}

// ── TOAST ─────────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type = '') {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className   = `toast ${type} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
}

// ── UTILS ─────────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escapeAttr(str) {
  return String(str).replace(/'/g, "\\'");
}
