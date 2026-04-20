const API_BASE_URL = 'http://localhost:8000';
const STORAGE_KEY = 'jarvis_conversations';

let currentConversationId = null;
let currentModel = null;
let currentBackend = 'deep_agent';
let currentBackendConfig = {};
let uploadedFileData = null;
let isStreaming = false;
let conversations = {}; // In-memory store of all conversations

// ── Generation / Ollama options ─────────────────────────────────────────────
// Mirrors what the user has set in the Generation panel.
// Values are sent with every /chat request and apply from the next turn onward.
let ollamaOptions = {
    reasoning:   false,   // send reasoning=False unless model supports it and toggle is on
    temperature: 0.8,     // Ollama default
    num_ctx:     10240,   // app default (Ollama default 2048 is too small for attachments)
    num_predict: -1,      // -1 = unlimited (Ollama default in v0.5+)
};
let modelSupportsThinking  = false;   // updated on every model change

// ── Model capabilities & context tracking ───────────────────────────────────
let modelCapabilities    = { tools: false, vision: false, thinking: false };
let modelNativeCtxLength = null;   // context_length from Ollama /api/show
let pdfVisionEnabled     = true;   // mirrors the PDF-vision toggle
let localConvMessages    = [];     // local mirror of sent/received messages for token counting
const TOKENS_PER_PDF_PAGE  = 1000; // estimated tokens per PDF page image at 150 DPI
const SYSTEM_PROMPT_TOKENS = 150;  // estimated system-prompt overhead

/** Read current form values → update ollamaOptions object. */
function syncGenOptions() {
    const temp    = parseFloat(document.getElementById('gen-temperature')?.value);
    const numCtx  = parseInt(document.getElementById('gen-num-ctx')?.value, 10);
    const numPred = parseInt(document.getElementById('gen-num-predict')?.value, 10);
    const thinkOn = document.getElementById('thinking-toggle')?.checked ?? false;

    ollamaOptions = {
        reasoning:   modelSupportsThinking && thinkOn ? true : false,
        temperature: isNaN(temp)    ? 0.8   : temp,
        num_ctx:     isNaN(numCtx)  ? 10240 : numCtx,
        num_predict: isNaN(numPred) ? -1    : numPred,
    };

    updateContextBar();  // num_ctx change affects effective total
}

/** Called when the thinking toggle changes. */
function onThinkingToggle() {
    syncGenOptions();
}

/**
 * Fetch model capabilities from /model-info and update:
 *   • capability tags (Tools / Vision / Thinking)
 *   • thinking badge + toggle
 *   • PDF-vision badge
 *   • num_ctx max clamp
 *   • context bar total
 *
 * Called every time the user selects a different model.
 */
async function checkModelCapabilities(modelId) {
    const thinkBadge  = document.getElementById('thinking-badge');
    const thinkToggle = document.getElementById('thinking-toggle');
    const visionBadge = document.getElementById('pdf-vision-badge');

    // Set loading state
    if (thinkBadge)  { thinkBadge.textContent  = '…'; thinkBadge.className  = 'thinking-badge checking'; }
    if (visionBadge) { visionBadge.textContent = '…'; visionBadge.className = 'thinking-badge checking'; }
    _setCapTagsLoading();

    try {
        const resp = await fetch(`${API_BASE_URL}/model-info/${encodeURIComponent(modelId)}`);
        const data = await resp.json();

        modelCapabilities = {
            tools:    (data.capabilities || []).includes('tools'),
            vision:   (data.capabilities || []).includes('vision'),
            thinking: (data.capabilities || []).includes('thinking'),
        };
        modelNativeCtxLength = data.context_length || null;
        modelSupportsThinking = modelCapabilities.thinking;
    } catch (_) {
        modelCapabilities    = { tools: false, vision: false, thinking: false };
        modelNativeCtxLength = null;
        modelSupportsThinking = false;
    }

    // ── Capability tags ────────────────────────────────────────────────
    _updateCapTags();

    // ── Thinking badge / toggle ────────────────────────────────────────
    if (thinkBadge && thinkToggle) {
        if (modelCapabilities.thinking) {
            thinkBadge.textContent = 'supported'; thinkBadge.className = 'thinking-badge yes';
            thinkToggle.disabled   = false;
        } else {
            thinkBadge.textContent = 'not supported'; thinkBadge.className = 'thinking-badge no';
            thinkToggle.checked    = false; thinkToggle.disabled = true;
        }
    }

    // ── PDF-vision badge ───────────────────────────────────────────────
    _updateVisionBadge();

    // ── Clamp num_ctx input to model native max ────────────────────────
    _clampNumCtxInput();

    // ── Context bar ────────────────────────────────────────────────────
    syncGenOptions();   // also calls updateContextBar()
}

function _setCapTagsLoading() {
    ['cap-tools', 'cap-vision', 'cap-thinking'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.className = 'cap-tag loading';
    });
}

function _updateCapTags() {
    const map = {
        'cap-tools':    modelCapabilities.tools,
        'cap-vision':   modelCapabilities.vision,
        'cap-thinking': modelCapabilities.thinking,
    };
    for (const [id, active] of Object.entries(map)) {
        const el = document.getElementById(id);
        if (el) el.className = 'cap-tag' + (active ? ' active' : ' inactive');
    }
}

function _updateVisionBadge() {
    const badge = document.getElementById('pdf-vision-badge');
    if (!badge) return;
    if (modelCapabilities.vision) {
        badge.textContent = 'supported'; badge.className = 'thinking-badge yes';
    } else if (pdfVisionEnabled) {
        // Toggle is on but model can't do vision → amber warning
        badge.textContent = 'not supported'; badge.className = 'thinking-badge warn';
    } else {
        badge.textContent = 'not supported'; badge.className = 'thinking-badge no';
    }
}

function _clampNumCtxInput() {
    if (!modelNativeCtxLength) return;
    const input = document.getElementById('gen-num-ctx');
    if (!input) return;
    input.max = modelNativeCtxLength;
    const cur = parseInt(input.value, 10);
    if (!isNaN(cur) && cur > modelNativeCtxLength) {
        input.value = modelNativeCtxLength;
    }
}

/** Called when the PDF-vision toggle changes. */
function onPdfVisionToggle() {
    pdfVisionEnabled = document.getElementById('pdf-vision-toggle')?.checked ?? true;
    _updateVisionBadge();
    updateContextBar();
}

// ── Trace sidebar state ─────────────────────────────────────────────────────
let traceEntries        = [];   // full history of all trace entries (accumulates)
let traceEventCount     = 0;
let _traceThinkId       = null; // ID of currently-open think bubble
let _traceAnswerId      = null; // ID of currently-open answer bubble

function _getTextLimit()  { return parseInt(document.getElementById('trace-text-limit')?.value  || '500',  10); }
function _getToolLimit()  { return parseInt(document.getElementById('trace-tool-limit')?.value  || '200',  10); }

// ── Context bar ─────────────────────────────────────────────────────────────

/** Approximate token count for a text string (4 chars ≈ 1 token). */
function _textTokens(s) { return Math.ceil((s || '').length / 4); }

/** Sum up tokens across all tracked messages + any pending attachment. */
function estimateConversationTokens() {
    let tokens = SYSTEM_PROMPT_TOKENS;

    for (const msg of localConvMessages) {
        if (typeof msg.content === 'string') {
            tokens += _textTokens(msg.content);
        } else if (Array.isArray(msg.content)) {
            for (const block of msg.content) {
                if (block.type === 'text')      tokens += _textTokens(block.text);
                else if (block.type === 'image_url') tokens += TOKENS_PER_PDF_PAGE;
            }
        }
    }

    // Pending PDF (attached but not yet sent)
    if (uploadedFileData) {
        tokens += _textTokens(uploadedFileData.text_content);
        if (pdfVisionEnabled && modelCapabilities.vision && uploadedFileData.images?.length) {
            tokens += uploadedFileData.images.length * TOKENS_PER_PDF_PAGE;
        }
    }

    return tokens;
}

/** Format token count compactly: 1234 → "1.2k", 500 → "500". */
function _fmtTokens(n) {
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
    return String(n);
}

/** Repaint the context status bar. */
function updateContextBar() {
    const numsEl = document.getElementById('ctx-nums');
    const fillEl = document.getElementById('ctx-bar-fill');
    if (!numsEl || !fillEl) return;

    const used = estimateConversationTokens();

    // Effective context window = min(user-set num_ctx, model native max)
    const userNumCtx = parseInt(document.getElementById('gen-num-ctx')?.value, 10) || 10240;
    const effectiveTotal = modelNativeCtxLength
        ? Math.min(userNumCtx, modelNativeCtxLength)
        : userNumCtx;

    const pct = Math.min(100, Math.round(used / effectiveTotal * 100));

    numsEl.textContent = `~${_fmtTokens(used)} / ${_fmtTokens(effectiveTotal)}`;
    fillEl.style.width = pct + '%';
    fillEl.className   = 'ctx-bar-fill'
        + (pct > 85 ? ' danger' : pct > 65 ? ' warn' : '');
}

function escapeHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/** Brief non-blocking toast banner (auto-dismiss after `durationMs`). */
function _showToast(msg, durationMs = 3500) {
    let toast = document.getElementById('_ctx-bump-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = '_ctx-bump-toast';
        Object.assign(toast.style, {
            position: 'fixed', bottom: '72px', left: '50%', transform: 'translateX(-50%)',
            background: 'rgba(30,40,55,0.95)', color: '#94a3b8',
            border: '1px solid rgba(148,163,184,0.2)', borderRadius: '6px',
            padding: '8px 16px', fontSize: '13px', fontFamily: 'inherit',
            zIndex: '9999', whiteSpace: 'nowrap', pointerEvents: 'none',
            transition: 'opacity 0.3s',
        });
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = '1';
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { toast.style.opacity = '0'; }, durationMs);
}

// Remove empty-state placeholder once first entry arrives
function _removeTraceEmptyState() {
    const empty = document.querySelector('#trace-body .trace-empty');
    if (empty) empty.remove();
}

function clearTrace() {
    traceEntries    = [];
    traceEventCount = 0;
    _traceThinkId   = null;
    _traceAnswerId  = null;
    const body = document.getElementById('trace-body');
    if (body) {
        body.innerHTML = `
        <div class="trace-empty">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" style="color:#22d3ee33;margin-bottom:8px">
                <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
            </svg>
            <span>No trace yet</span>
            <span>Agent reasoning, tool calls,<br>and thinking blocks appear here</span>
        </div>`;
    }
    _updateTraceCount();
}

function _updateTraceCount() {
    const el = document.getElementById('trace-count');
    if (el) el.textContent = traceEventCount > 0 ? `${traceEventCount} events` : '';
}

function addTraceMessageSep(label) {
    _removeTraceEmptyState();
    const body = document.getElementById('trace-body');
    if (!body) return;
    const sep = document.createElement('div');
    sep.className = 'trace-msg-sep';
    sep.innerHTML = `<span>${escapeHtml(label)}</span>`;
    body.appendChild(sep);
    _scrollTrace();
}

/** Create a new bubble in the trace body. Returns the entry id. */
function addTraceBubble(type, label, limitOverride) {
    _removeTraceEmptyState();
    const body = document.getElementById('trace-body');
    if (!body) return null;

    const limit = limitOverride !== undefined
        ? limitOverride
        : (['tool_call', 'tool_result'].includes(type) ? _getToolLimit()
          : ['agent_turn', 'progress'].includes(type) ? Number.MAX_SAFE_INTEGER
          : _getTextLimit());

    const id = 'tb-' + Date.now() + '-' + Math.random().toString(36).slice(2, 7);
    const entry = { id, type, label, content: '', streaming: true, expanded: false, limit };
    traceEntries.push(entry);
    traceEventCount++;
    _updateTraceCount();

    const el = document.createElement('div');
    el.className = `trace-bubble trace-bubble--${type} streaming`;
    el.id = id;
    el.innerHTML = _bubbleInner(entry);
    body.appendChild(el);
    _scrollTrace();
    return id;
}

/** Append streaming tokens to an existing bubble. */
function appendTraceBubble(id, chunk) {
    if (!chunk) return;
    const entry = traceEntries.find(e => e.id === id);
    if (!entry) return;
    entry.content += chunk;
    _repaintBubble(entry);
    _scrollTrace();
}

/** Set the full content of a bubble at once (non-streaming). */
function setTraceBubbleContent(id, content) {
    const entry = traceEntries.find(e => e.id === id);
    if (!entry) return;
    entry.content   = content;
    entry.streaming = false;
    const el = document.getElementById(id);
    if (el) el.classList.remove('streaming');
    _repaintBubble(entry);
    _scrollTrace();
}

/** Mark a streaming bubble as done. */
function finalizeTraceBubble(id) {
    const entry = traceEntries.find(e => e.id === id);
    if (!entry) return;
    entry.streaming = false;
    const el = document.getElementById(id);
    if (el) el.classList.remove('streaming');
    _repaintBubble(entry);
}

function _repaintBubble(entry) {
    const el = document.getElementById(entry.id);
    if (!el) return;

    const shown = entry.expanded ? entry.content : entry.content.slice(0, entry.limit);
    const cursor = entry.streaming ? '<span class="trace-cursor">▋</span>' : '';
    const overflow = !entry.expanded && entry.content.length > entry.limit;

    // Content area
    let contentEl = el.querySelector('.tb-content');
    if (!contentEl) {
        el.innerHTML = _bubbleInner(entry);
        contentEl = el.querySelector('.tb-content');
    } else {
        contentEl.innerHTML = escapeHtml(shown) + cursor;
    }

    // "Show more" button
    let moreBtn = el.querySelector('.tb-more-btn');
    if (overflow) {
        if (!moreBtn) {
            moreBtn = document.createElement('button');
            moreBtn.className = 'tb-more-btn';
            moreBtn.setAttribute('data-entry', entry.id);
            el.appendChild(moreBtn);
        }
        moreBtn.textContent = `… show ${entry.content.length - entry.limit} more chars`;
    } else if (moreBtn && !overflow) {
        moreBtn.remove();
    }
}

function _bubbleInner(entry) {
    const shown  = entry.expanded ? entry.content : entry.content.slice(0, entry.limit);
    const cursor = entry.streaming ? '<span class="trace-cursor">▋</span>' : '';
    return `<div class="tb-label">${escapeHtml(entry.label)}</div>
<div class="tb-content">${escapeHtml(shown)}${cursor}</div>`;
}

function _scrollTrace() {
    const body = document.getElementById('trace-body');
    if (body) body.scrollTop = body.scrollHeight;
}

function toggleTraceSidebar() {
    const sidebar = document.getElementById('trace-sidebar');
    if (sidebar) sidebar.classList.toggle('collapsed');
}

async function downloadContext() {
    if (!currentConversationId) {
        alert('No conversation yet — send a message first.');
        return;
    }
    try {
        const resp = await fetch(`${API_BASE_URL}/conversation/${encodeURIComponent(currentConversationId)}/messages?truncate_images=true`);
        if (!resp.ok) {
            const err = await resp.text();
            alert(`Failed to fetch context: ${resp.status} ${err}`);
            return;
        }
        const data = await resp.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url  = URL.createObjectURL(blob);
        const ts   = new Date().toISOString().replace(/[:.]/g, '-');
        const a    = document.createElement('a');
        a.href = url;
        a.download = `context-${currentConversationId.slice(0, 8)}-${ts}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (e) {
        alert(`Failed to download context: ${e.message}`);
    }
}

function maximizeTrace() {
    if (traceEntries.length === 0) return;
    const html = _buildTraceHtml();
    const blob = new Blob([html], { type: 'text/html; charset=utf-8' });
    const url  = URL.createObjectURL(blob);
    window.open(url, '_blank');
    // revoke after 60 s — tab has loaded by then
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

function _buildTraceHtml() {
    const cfg = {
        agent_turn:  { label: 'Agent Turn',  color: '#22d3ee' },
        think:       { label: 'Thinking',    color: '#a78bfa' },
        tool_call:   { label: 'Tool Call',   color: '#f5b731' },
        tool_result: { label: 'Tool Result', color: '#4ade80' },
        answer:      { label: 'Answer',      color: '#c4c6d8' },
        progress:    { label: 'Progress',    color: '#3b3d52' },
        error:       { label: 'Error',       color: '#f87171' },
    };

    const bubblesHtml = traceEntries.map(entry => {
        const c = cfg[entry.type] ?? { label: entry.type, color: '#6b6e85' };
        return `<div style="margin-bottom:14px;padding:11px 14px;background:rgba(255,255,255,0.028);border-left:2px solid ${c.color};border-radius:5px;">
  <div style="font-size:8.5px;text-transform:uppercase;letter-spacing:0.22em;color:${c.color};margin-bottom:7px;opacity:0.85;">${escapeHtml(entry.label)}</div>
  <pre style="white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.68;color:rgba(200,202,212,0.85);margin:0;font-family:inherit;">${escapeHtml(entry.content)}</pre>
</div>`;
    }).join('\n');

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Jarvis — Thought Trace</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#03040a;color:#eaebf4;font-family:'IBM Plex Mono',ui-monospace,'Cascadia Code',monospace;padding:32px;line-height:1.6;font-size:13px}
  h1{font-size:18px;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#f5b731;margin-bottom:6px}
  .meta{font-size:10.5px;color:#3b3d52;margin-bottom:28px;letter-spacing:0.06em}
  ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:#252736;border-radius:2px}
</style>
</head>
<body>
<h1>Thought Trace</h1>
<div class="meta">Generated ${new Date().toLocaleString()} &nbsp;·&nbsp; ${traceEntries.length} events</div>
${bubblesHtml}
</body>
</html>`;
}

// ── Alpine.js agent selector component ───────────────────
function agentSelector() {
    return {
        backends: [],
        selectedId: 'deep_agent',
        config: {},           // flat map of field → value
        schema: {},           // config_schema of selected backend

        init() {
            window._agentSelector = this;
            fetch(`${API_BASE_URL}/backends`)
                .then(r => r.json())
                .then(data => {
                    this.backends = data.backends;
                    this.setBackend(data.default_backend || 'deep_agent');
                })
                .catch(e => console.error('Failed to load backends:', e));
        },

        setBackend(id) {
            const backend = this.backends.find(b => b.id === id);
            if (!backend) return;
            this.selectedId = id;
            this.schema = backend.config_schema || {};
            // Reset config to defaults
            const newConfig = {};
            for (const [key, field] of Object.entries(this.schema)) {
                newConfig[key] = field.default !== undefined
                    ? (Array.isArray(field.default) ? [...field.default] : field.default)
                    : '';
            }
            this.config = newConfig;
            currentBackend = id;
            currentBackendConfig = this._serialisedConfig();
        },

        onBackendChange(e) { this.setBackend(e.target.value); },

        onConfigChange() {
            currentBackendConfig = this._serialisedConfig();
        },

        _serialisedConfig() {
            const out = {};
            for (const [key, field] of Object.entries(this.schema)) {
                let val = this.config[key];
                if (field.type === 'model_list' && typeof val === 'string') {
                    val = val.split(',').map(s => s.trim()).filter(Boolean);
                } else if (field.type === 'integer') {
                    val = parseInt(val, 10) || field.default;
                }
                out[key] = val;
            }
            return out;
        },

        get hasConfig() {
            return Object.keys(this.schema).length > 0;
        },

        configLabel(key) {
            return this.schema[key]?.label || key;
        },

        configHint(key) {
            return this.schema[key]?.description || '';
        },

        configType(key) {
            return this.schema[key]?.type || 'string';
        },

        // Display value in textarea — arrays shown as comma-separated
        configDisplayValue(key) {
            const v = this.config[key];
            return Array.isArray(v) ? v.join(', ') : (v ?? '');
        },

        updateListField(key, rawValue) {
            this.config[key] = rawValue;
            this.onConfigChange();
        }
    };
}

// ── Alpine.js combobox component ─────────────────────────
function modelCombobox() {
    return {
        query: '',
        open: false,
        models: [],
        cursor: -1,
        _selected: '',

        // Flat filtered list used for cursor/keyboard navigation
        get filtered() {
            return [...this.localFiltered, ...this.cloudFiltered];
        },

        get localFiltered() {
            const q = this.query.toLowerCase();
            return this.models.filter(m =>
                m.group === 'local' &&
                (!q || m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q))
            );
        },

        get cloudFiltered() {
            const q = this.query.toLowerCase();
            return this.models.filter(m =>
                m.group === 'cloud' &&
                (!q || m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q))
            );
        },

        init() {
            window._modelCombobox = this;
            fetch(`${API_BASE_URL}/models`)
                .then(r => r.json())
                .then(data => {
                    this.models = data.models;
                    this.setModel(data.default_model);
                })
                .catch(e => console.error('Failed to load models:', e));
        },

        setModel(id) {
            this._selected = id;
            this.query = id;
            this.open = false;
            this.cursor = -1;
            currentModel = id;
            updateCurrentModelDisplay();
            checkModelCapabilities(id);   // update thinking badge + toggle
        },

        select(model) { this.setModel(model.id); },

        onInput() {
            currentModel = this.query.trim();
            this.cursor = -1;
            updateCurrentModelDisplay();
        },

        confirmSelection() {
            if (this.cursor >= 0 && this.cursor < this.filtered.length) {
                this.select(this.filtered[this.cursor]);
            } else if (this.query.trim()) {
                this.setModel(this.query.trim());
            }
        },

        // Open dropdown: clear search so all models are visible.
        // Close without selection: restore previously selected model ID.
        toggleOpen() {
            if (!this.open) {
                this.open = true;
                this.cursor = -1;
                this.query = '';          // show all groups on open
                this.$nextTick(() => this.$refs.cbInput && this.$refs.cbInput.focus());
            } else {
                this.open = false;
                this.query = this._selected || '';  // restore label
            }
        },

        moveDown() {
            if (!this.open) { this.open = true; this.query = ''; return; }
            this.cursor = Math.min(this.cursor + 1, this.filtered.length - 1);
        },

        moveUp() {
            this.cursor = Math.max(this.cursor - 1, -1);
        }
    };
}

marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true
});

async function loadModels() {
    // Handled by Alpine.js modelCombobox component
}

function updateCurrentModelDisplay() {
    const el = document.getElementById('current-model');
    if (el) el.textContent = currentModel || '—';
}

function startNewChat() {
    currentConversationId = null;
    uploadedFileData      = null;
    localConvMessages     = [];
    updateContextBar();
    
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">
                <svg width="26" height="26" viewBox="0 0 16 16" fill="none">
                    <path d="M8 1L1 5v6l7 4 7-4V5L8 1z" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/>
                    <path d="M1 5l7 4M8 9l7-4M8 9v6" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
                </svg>
            </div>
            <h2>How can I help?</h2>
            <p>Ask anything — I can search the web, read PDFs,<br>save artifacts, and reason through complex tasks.</p>
        </div>
    `;
    
    removeUploadedFile();
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

async function sendMessage() {
    const input   = document.getElementById('message-input');
    const message = input.value.trim();

    if (!message && !uploadedFileData) return;
    if (isStreaming) return;

    const webSearchEnabled = document.getElementById('web-search-toggle').checked;
    const numSearchResults = parseInt(document.getElementById('search-results-count')?.value || '5', 10);

    // ── Build message content ────────────────────────────────────────────────
    // messageContent: multimodal list sent when PDF images should go to model.
    // messageForBackend: plain text fallback (no images).
    // messageForDisplay: what appears in the chat UI bubble.
    let messageContent   = null;   // list of {type, ...} blocks — null = plain text
    let messageForBackend;
    const isExcel = uploadedFileData?.file_type === 'excel';
    console.debug('[sendMessage] uploadedFileData:', uploadedFileData ? {filename: uploadedFileData.filename, file_type: uploadedFileData.file_type, text_len: uploadedFileData.text_content?.length} : null);
    const messageForDisplay = message || `[Attached: ${uploadedFileData?.filename}]`;

    if (uploadedFileData) {
        // Sanitise filename for use inside an XML-style tag name
        const safeFilename = uploadedFileData.filename.replace(/[^A-Za-z0-9._-]/g, '_');

        // Vision path: PDF only — Excel/CSV is always text-only (it's JSON, not images).
        const sendImages = !isExcel
            && pdfVisionEnabled
            && modelCapabilities.vision
            && (uploadedFileData.images?.length > 0);

        if (sendImages) {
            const images = uploadedFileData.images;

            // Ollama's /api/chat wire format flattens multimodal content:
            //   { content: "<all text joined>", images: ["b64_1", "b64_2", ...] }
            // LangChain's ChatOllama adapter concatenates all type:text blocks and
            // puts all type:image_url blocks into a flat images[] array — positional
            // interleaving (caption immediately before its image) is lost.
            // Solution: number every image explicitly in the text so all Ollama
            // vision models can correlate "Image N" → Nth entry in images[].
            // Cloud APIs (Groq / OpenAI) handle the same numbered references fine.
            const imagePreamble = images.map((_, i) =>
                `[Image ${i + 1} = Page ${i + 1} of ${safeFilename}]`
            ).join('\n');

            const textContent = `<pdf-filename-${safeFilename}>\n${uploadedFileData.text_content}\n</pdf-filename-${safeFilename}>\n\n${imagePreamble}\n\nUser question: ${message}`;

            const imageBlocks = images.map(img => ({
                type: 'image_url',
                image_url: { url: `data:image/png;base64,${img.data}` }
            }));

            messageContent    = [{ type: 'text', text: textContent }, ...imageBlocks];
            messageForBackend = null;   // backend uses message_content
        } else {
            // Text-only path — covers: PDF without vision, PDF on non-vision model,
            // and all Excel/CSV uploads. Tag name differs by file type.
            const tagName = isExcel
                ? `excel-filename-${safeFilename}`
                : `pdf-filename-${safeFilename}`;
            const textContent = `<${tagName}>\n${uploadedFileData.text_content}\n</${tagName}>\n\nUser question: ${message}`;
            messageContent    = [{ type: 'text', text: textContent }];
            messageForBackend = null;
        }
    } else {
        messageForBackend = message;
    }

    console.debug('[sendMessage] messageContent set:', messageContent ? `${messageContent.length} block(s), text_len=${messageContent[0]?.text?.length}` : 'null (plain text)');

    // num_ctx is taken verbatim from the left-sidebar UI value — no auto-bump.
    // The user controls the context window; if the conversation exceeds it,
    // Ollama will truncate from the start, and the context status bar in the
    // right sidebar (red when >85%) is the signal to raise num_ctx manually.
    const effectiveOllamaOptions = { ...ollamaOptions };

    // Track locally for context estimation
    localConvMessages.push({ role: 'user', content: messageContent || messageForBackend });

    // ── Clear the pending attachment immediately ───────────────────────────
    // The PDF content is now in localConvMessages above. If uploadedFileData
    // stays set, estimateConversationTokens() counts the PDF a second time via
    // the "pending" branch — causing the ~2× overcount seen in the context bar
    // (e.g. "~46.9k / 24k" right after pressing send). Capture the images
    // ref first (for addPdfImages below), then call removeUploadedFile() which
    // nulls uploadedFileData, hides the attachment pill, and updates the bar.
    const pendingImages = uploadedFileData?.images || [];
    removeUploadedFile();   // sets uploadedFileData = null, calls updateContextBar()

    input.value = '';
    input.style.height = 'auto';

    removeWelcomeMessage();
    addMessage('user', messageForDisplay);

    if (pendingImages.length > 0) {
        addPdfImages(pendingImages);
    }

    const assistantMessageId = addMessage('assistant', '', true);

    isStreaming = true;
    updateSendButton();

    // ── Trace: add message separator and reset per-message state ──────────
    addTraceMessageSep(new Date().toLocaleTimeString());
    _traceThinkId  = null;
    _traceAnswerId = null;

    try {
        const reqBody = {
            message:            messageForDisplay,
            model_id:           currentModel,
            conversation_id:    currentConversationId,
            web_search_enabled: webSearchEnabled,
            num_search_results: numSearchResults,
            backend_id:         currentBackend,
            backend_config:     currentBackendConfig,
            ollama_options:     effectiveOllamaOptions,
            allow_network:      !!(document.getElementById('sandbox-network-toggle')?.checked),
        };
        if (messageContent) reqBody.message_content = messageContent;

        const response = await fetch(`${API_BASE_URL}/chat`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(reqBody),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        // ── agent_turn: new LLM call started ──────────────
                        if (data.type === 'agent_turn') {
                            // Finalize any open think bubble from previous turn
                            if (_traceThinkId) {
                                finalizeTraceBubble(_traceThinkId);
                                _traceThinkId = null;
                            }
                            const turnId = addTraceBubble('agent_turn', `LLM Call ${data.turn}`);
                            finalizeTraceBubble(turnId);

                        // ── think: streaming reasoning tokens ─────────────
                        } else if (data.type === 'think') {
                            if (!_traceThinkId) {
                                _traceThinkId = addTraceBubble('think', 'Thinking');
                            }
                            appendTraceBubble(_traceThinkId, data.content);

                        // ── content: final answer tokens ──────────────────
                        } else if (data.type === 'content') {
                            // Close any open think bubble before the answer starts
                            if (_traceThinkId) {
                                finalizeTraceBubble(_traceThinkId);
                                _traceThinkId = null;
                            }
                            // Main chat
                            fullContent += data.content;
                            updateMessageContent(assistantMessageId, fullContent);
                            // Mirror to trace sidebar answer bubble
                            if (!_traceAnswerId) {
                                _traceAnswerId = addTraceBubble('answer', 'Final Answer');
                            }
                            appendTraceBubble(_traceAnswerId, data.content);

                        // ── progress: step label ──────────────────────────
                        } else if (data.type === 'progress') {
                            addProgressStep(assistantMessageId, data.step, data.description, true);
                            addTraceBubble('progress', 'Progress');
                            setTraceBubbleContent(
                                traceEntries[traceEntries.length - 1].id,
                                data.description
                            );

                        // ── tool_start: tool invoked ──────────────────────
                        } else if (data.type === 'tool_start') {
                            // Finalize any open think bubble
                            if (_traceThinkId) {
                                finalizeTraceBubble(_traceThinkId);
                                _traceThinkId = null;
                            }
                            // Main chat indicator
                            showToolIndicator(assistantMessageId, data.tool, true, data.description);
                            // Trace: show tool name + args
                            const argsStr = data.args && Object.keys(data.args).length
                                ? JSON.stringify(data.args, null, 2)
                                : '(no args)';
                            const toolBubbleId = addTraceBubble('tool_call', `Tool: ${data.tool}`);
                            setTraceBubbleContent(toolBubbleId, argsStr);

                        // ── tool_end ──────────────────────────────────────
                        } else if (data.type === 'tool_end') {
                            showToolIndicator(assistantMessageId, data.tool, false);

                        // ── tool_result: raw tool output ──────────────────
                        } else if (data.type === 'tool_result') {
                            const resultBubbleId = addTraceBubble('tool_result', `Result: ${data.tool}`);
                            setTraceBubbleContent(resultBubbleId, data.result || '(empty)');

                        // ── done ──────────────────────────────────────────
                        } else if (data.type === 'done') {
                            currentConversationId = data.conversation_id;
                            if (_traceThinkId)  { finalizeTraceBubble(_traceThinkId);  _traceThinkId  = null; }
                            if (_traceAnswerId) { finalizeTraceBubble(_traceAnswerId); _traceAnswerId = null; }
                            // Track assistant reply for context bar
                            if (fullContent) {
                                localConvMessages.push({ role: 'assistant', content: fullContent });
                            }
                            updateContextBar();

                        // ── chain events (pass-through) ───────────────────
                        } else if (data.type === 'chain_start') {
                            addProgressStep(assistantMessageId, 'chain', `Running: ${data.name}`, true);
                        } else if (data.type === 'chain_end') {
                            updateProgressStep(assistantMessageId, 'chain', `Completed: ${data.name}`, true);

                        // ── error ─────────────────────────────────────────
                        } else if (data.type === 'error') {
                            updateMessageContent(assistantMessageId, `Error: ${data.error}`);
                            const errId = addTraceBubble('error', 'Error');
                            setTraceBubbleContent(errId, data.error);
                        }
                    } catch (e) {
                        console.error('Failed to parse SSE data:', e);
                    }
                }
            }
        }

        removeLoadingIndicator(assistantMessageId);
        
    } catch (error) {
        console.error('Failed to send message:', error);
        updateMessageContent(assistantMessageId, `Error: ${error.message}`);
    } finally {
        isStreaming = false;
        updateSendButton();
        // uploadedFileData was already cleared at send time (see above).
        // removeUploadedFile() is a safe no-op if called again but we skip
        // the redundant updateContextBar() it would trigger.
        updateContextBar();
        
        // Save conversation to history
        const messages = getCurrentMessages();
        if (messages.length > 0) {
            saveCurrentConversation(messages);
        }
    }
}

function removeWelcomeMessage() {
    const welcome = document.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }
}

function addMessage(role, content, isLoading = false) {
    const messagesContainer = document.getElementById('messages-container');
    const messageId = 'msg-' + Date.now();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.id = messageId;
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    headerDiv.textContent = role === 'user' ? 'You' : 'Jarvis';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isLoading) {
        contentDiv.innerHTML = `
            <div class="loading-indicator">
                <span>Thinking</span>
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
    } else {
        contentDiv.innerHTML = renderMarkdown(content);
    }
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageId;
}

function updateMessageContent(messageId, content) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) return;
    
    const loadingIndicator = contentDiv.querySelector('.loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
    
    contentDiv.innerHTML = renderMarkdown(content);
    
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeLoadingIndicator(messageId) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const loadingIndicator = messageDiv.querySelector('.loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
}

function getOrCreateProgressContainer(messageId) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return null;
    
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) return null;
    
    let progressContainer = contentDiv.querySelector('.progress-container');
    if (!progressContainer) {
        progressContainer = document.createElement('div');
        progressContainer.className = 'progress-container';
        
        const header = document.createElement('div');
        header.className = 'progress-header';
        header.textContent = 'Progress';
        progressContainer.appendChild(header);
        
        const loadingIndicator = contentDiv.querySelector('.loading-indicator');
        if (loadingIndicator) {
            contentDiv.insertBefore(progressContainer, loadingIndicator);
        } else {
            contentDiv.insertBefore(progressContainer, contentDiv.firstChild);
        }
    }
    return progressContainer;
}

function addProgressStep(messageId, stepNumber, description, isActive = true) {
    const progressContainer = getOrCreateProgressContainer(messageId);
    if (!progressContainer) return;
    
    // Check if step already exists
    let stepDiv = progressContainer.querySelector(`.progress-step[data-step="${stepNumber}"]`);
    
    if (!stepDiv) {
        stepDiv = document.createElement('div');
        stepDiv.className = 'progress-step' + (isActive ? ' active' : '');
        stepDiv.setAttribute('data-step', stepNumber);
        
        const stepNum = document.createElement('span');
        stepNum.className = 'progress-step-number';
        stepNum.textContent = stepNumber;
        
        const stepText = document.createElement('span');
        stepText.className = 'progress-step-text';
        stepText.textContent = description;
        
        stepDiv.appendChild(stepNum);
        stepDiv.appendChild(stepText);
        progressContainer.appendChild(stepDiv);
    } else {
        // Update existing step
        const stepText = stepDiv.querySelector('.progress-step-text');
        if (stepText) {
            stepText.textContent = description;
        }
    }
    
    // Scroll to keep progress visible
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateProgressStep(messageId, stepNumber, description, isComplete = false) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const progressContainer = messageDiv.querySelector('.progress-container');
    if (!progressContainer) return;
    
    const stepDiv = progressContainer.querySelector(`.progress-step[data-step="${stepNumber}"]`);
    if (stepDiv) {
        stepDiv.className = 'progress-step' + (isComplete ? ' complete' : ' active');
        const stepText = stepDiv.querySelector('.progress-step-text');
        if (stepText && description) {
            stepText.textContent = description;
        }
    }
}

function addToolProgress(messageId, toolName, description, isActive = true) {
    const progressContainer = getOrCreateProgressContainer(messageId);
    if (!progressContainer) return;
    
    const toolId = `tool-${toolName}-${Date.now()}`;
    
    if (isActive) {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'progress-step active';
        stepDiv.setAttribute('data-tool', toolName);
        stepDiv.setAttribute('data-tool-id', toolId);
        
        const stepNum = document.createElement('span');
        stepNum.className = 'progress-step-number';
        stepNum.innerHTML = '&#8226;'; // bullet point for tools
        
        const stepText = document.createElement('span');
        stepText.className = 'progress-step-text';
        stepText.textContent = description || `Using ${toolName}...`;
        
        stepDiv.appendChild(stepNum);
        stepDiv.appendChild(stepText);
        progressContainer.appendChild(stepDiv);
    }
    
    // Scroll to keep progress visible
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return toolId;
}

function completeToolProgress(messageId, toolName) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const progressContainer = messageDiv.querySelector('.progress-container');
    if (!progressContainer) return;
    
    // Find the most recent active tool step with this name
    const toolSteps = progressContainer.querySelectorAll(`.progress-step.active[data-tool="${toolName}"]`);
    if (toolSteps.length > 0) {
        const lastStep = toolSteps[toolSteps.length - 1];
        lastStep.className = 'progress-step complete';
    }
}

function showToolIndicator(messageId, toolName, isActive, description = null) {
    if (isActive) {
        addToolProgress(messageId, toolName, description, true);
    } else {
        completeToolProgress(messageId, toolName);
    }
}

function preprocessLatex(content) {
    if (!content) return '';
    
    // Protect code blocks from LaTeX processing
    const codeBlocks = [];
    let processed = content.replace(/```[\s\S]*?```/g, (match) => {
        codeBlocks.push(match);
        return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
    });
    
    // Protect inline code
    const inlineCode = [];
    processed = processed.replace(/`[^`]+`/g, (match) => {
        inlineCode.push(match);
        return `__INLINE_CODE_${inlineCode.length - 1}__`;
    });
    
    // Convert standalone [ and ] on their own lines to $$ delimiters for display math
    // Match patterns like:\n[\nequation\n]\n
    processed = processed.replace(/\n\s*\[\s*\n([\s\S]*?)\n\s*\]\s*\n/g, '\n$$\n$1\n$$\n');
    
    // Also handle [ at start of content
    processed = processed.replace(/^\s*\[\s*\n([\s\S]*?)\n\s*\]\s*\n/g, '$$\n$1\n$$\n');
    
    // Convert \[ and \] to $$ for display math (in case they're escaped differently)
    processed = processed.replace(/\\\[/g, '$$');
    processed = processed.replace(/\\\]/g, '$$');
    
    // Convert \( and \) to $ for inline math
    processed = processed.replace(/\\\(/g, '$');
    processed = processed.replace(/\\\)/g, '$');
    
    // Restore code blocks
    codeBlocks.forEach((block, i) => {
        processed = processed.replace(`__CODE_BLOCK_${i}__`, block);
    });
    
    // Restore inline code
    inlineCode.forEach((code, i) => {
        processed = processed.replace(`__INLINE_CODE_${i}__`, code);
    });
    
    return processed;
}

/**
 * Strip a single outer ```markdown / ```md / ``` fence that wraps the entire
 * message. Some models (qwen3 variants especially) emit their full reply inside
 * a fenced block, which `marked` then renders as literal code instead of as
 * markdown. We only unwrap when the fence spans the whole trimmed content,
 * and the inner language tag is empty, 'markdown', or 'md' — so legitimate
 * code blocks (```python, ```json, etc.) are left alone.
 */
function _unwrapOuterMarkdownFence(s) {
    if (!s) return s;
    const trimmed = s.trim();

    // Case 1: complete fence wrapping the whole message (```markdown ... ```)
    const full = trimmed.match(/^```([A-Za-z0-9_-]*)\s*\n([\s\S]*?)\n?```\s*$/);
    if (full) {
        const lang = full[1].toLowerCase();
        if (lang === '' || lang === 'markdown' || lang === 'md') {
            return full[2];
        }
        return s;  // leave ```python / ```json / … intact
    }

    // Case 2: in-progress stream — opening ```markdown/```md seen, closing not
    // yet. Strip only the opening fence so the partial content renders as
    // markdown while tokens are still arriving. We require a language tag of
    // "markdown" or "md" here (not the bare ```) to avoid stripping the start
    // of a legitimate code block mid-stream.
    const openOnly = trimmed.match(/^```(markdown|md)\s*\n([\s\S]*)$/i);
    if (openOnly) {
        return openOnly[2];
    }

    return s;
}

function renderMarkdown(content) {
    if (!content) return '';

    // Unwrap full-message ```markdown fences before parsing
    const unwrapped = _unwrapOuterMarkdownFence(content);

    // Pre-process LaTeX delimiters
    const processedContent = preprocessLatex(unwrapped);
    
    let html = marked.parse(processedContent);
    
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = html;
    
    renderMathInElement(tempDiv, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ],
        throwOnError: false,
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    });
    
    return tempDiv.innerHTML;
}

function addPdfImages(images) {
    const messagesContainer = document.getElementById('messages-container');
    const lastUserMessage = messagesContainer.querySelector('.message.user:last-of-type');
    
    if (lastUserMessage && images.length > 0) {
        const imagesDiv = document.createElement('div');
        imagesDiv.className = 'pdf-images';
        
        images.slice(0, 5).forEach(img => {
            const imgEl = document.createElement('img');
            imgEl.className = 'pdf-image-thumb';
            imgEl.src = `data:image/png;base64,${img.data}`;
            imgEl.alt = `Page ${img.page}`;
            imgEl.title = `Page ${img.page}`;
            imagesDiv.appendChild(imgEl);
        });
        
        if (images.length > 5) {
            const moreSpan = document.createElement('span');
            moreSpan.style.color = 'var(--text-muted)';
            moreSpan.style.fontSize = '0.75rem';
            moreSpan.style.alignSelf = 'center';
            moreSpan.textContent = `+${images.length - 5} more pages`;
            imagesDiv.appendChild(moreSpan);
        }
        
        lastUserMessage.appendChild(imagesDiv);
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const allowedExts = ['.pdf', '.xlsx', '.xls', '.csv'];
    if (!allowedExts.some(ext => file.name.toLowerCase().endsWith(ext))) {
        alert('Supported file types: PDF, XLSX, XLS, CSV');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        document.getElementById('file-name').textContent = 'Uploading...';
        document.getElementById('uploaded-file').style.display = 'inline-flex';
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        uploadedFileData = await response.json();
        document.getElementById('file-name').textContent = uploadedFileData.filename;
        updateContextBar();   // reflect pending PDF tokens immediately

    } catch (error) {
        console.error('Failed to upload file:', error);
        alert('Failed to upload file: ' + error.message);
        removeUploadedFile();
    }
}

function removeUploadedFile() {
    uploadedFileData = null;
    document.getElementById('uploaded-file').style.display = 'none';
    document.getElementById('file-name').textContent = '';
    document.getElementById('file-input').value = '';
    updateContextBar();
}

function updateSendButton() {
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = isStreaming;
}

async function downloadConversation() {
    if (!currentConversationId) {
        alert('No conversation to download');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/download-pdf`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                conversation_id: currentConversationId
            })
        });
        
        if (!response.ok) {
            throw new Error('Download failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `conversation_${currentConversationId.slice(0, 8)}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
    } catch (error) {
        console.error('Failed to download conversation:', error);
        alert('Failed to download conversation: ' + error.message);
    }
}

async function viewArtifacts() {
    const modal = document.getElementById('artifacts-modal');
    const artifactsList = document.getElementById('artifacts-list');
    
    modal.classList.add('active');
    artifactsList.innerHTML = 'Loading artifacts...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/artifacts`);
        const data = await response.json();
        
        if (data.artifacts.length === 0) {
            artifactsList.innerHTML = '<p style="color: var(--text-muted);">No artifacts found.</p>';
            return;
        }
        
        artifactsList.innerHTML = '';
        data.artifacts.forEach(artifact => {
            const item = document.createElement('div');
            item.className = 'artifact-item';
            item.onclick = () => viewArtifact(artifact.name);
            
            item.innerHTML = `
                <span class="artifact-name">${artifact.name}</span>
                <span class="artifact-meta">${formatFileSize(artifact.size)}</span>
            `;
            
            artifactsList.appendChild(item);
        });
        
    } catch (error) {
        console.error('Failed to load artifacts:', error);
        artifactsList.innerHTML = '<p style="color: var(--error);">Failed to load artifacts.</p>';
    }
}

async function viewArtifact(filename) {
    closeArtifactsModal();
    
    const modal = document.getElementById('artifact-view-modal');
    const title = document.getElementById('artifact-view-title');
    const content = document.getElementById('artifact-content');
    
    modal.classList.add('active');
    title.textContent = filename;
    content.textContent = 'Loading...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/artifacts/${encodeURIComponent(filename)}`);
        const data = await response.json();
        content.textContent = data.content;
        
    } catch (error) {
        console.error('Failed to load artifact:', error);
        content.textContent = 'Failed to load artifact.';
    }
}

function closeArtifactsModal() {
    document.getElementById('artifacts-modal').classList.remove('active');
}

function closeArtifactViewModal() {
    document.getElementById('artifact-view-modal').classList.remove('active');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

document.addEventListener('click', (e) => {
    // Close modals when clicking the backdrop
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
    // "Show more" buttons inside trace bubbles (delegated)
    const moreBtn = e.target.closest('.tb-more-btn');
    if (moreBtn) {
        const entryId = moreBtn.getAttribute('data-entry');
        const entry   = traceEntries.find(en => en.id === entryId);
        if (entry) {
            entry.expanded = true;
            _repaintBubble(entry);
        }
    }
});

// Chat history functions
function loadConversationsFromStorage() {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            conversations = JSON.parse(stored);
        }
    } catch (e) {
        console.error('Failed to load conversations from storage:', e);
        conversations = {};
    }
}

function saveConversationsToStorage() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (e) {
        console.error('Failed to save conversations to storage:', e);
    }
}

function saveCurrentConversation(messages) {
    if (!currentConversationId) return;
    
    const title = generateConversationTitle(messages);
    conversations[currentConversationId] = {
        id: currentConversationId,
        title: title,
        messages: messages,
        model: currentModel,
        backend_id: currentBackend,
        updatedAt: new Date().toISOString()
    };
    saveConversationsToStorage();
    renderChatHistory();
}

function generateConversationTitle(messages) {
    const firstUserMsg = messages.find(m => m.role === 'user');
    if (firstUserMsg) {
        // content may be a multimodal list — extract text from first text block
        let text = '';
        const c = firstUserMsg.content;
        if (typeof c === 'string') {
            text = c;
        } else if (Array.isArray(c)) {
            const textBlock = c.find(b => b.type === 'text');
            text = textBlock?.text || '';
        }
        text = text
            .replace(/<pdf-filename-[^>]+>[\s\S]*?<\/pdf-filename-[^>]+>/g, '')
            .replace(/<excel-filename-[^>]+>[\s\S]*?<\/excel-filename-[^>]+>/g, '')
            .replace(/^\[Image \d+ = Page \d+ of [^\]]+\]\n?/gm, '')
            .replace(/User question:/g, '')
            .trim();
        return text.length > 40 ? text.substring(0, 40) + '...' : text || 'New Chat';
    }
    return 'New Chat';
}

function renderChatHistory() {
    const historyContainer = document.getElementById('chat-history');
    if (!historyContainer) return;
    
    // Sort conversations by updatedAt, most recent first
    const sortedConversations = Object.values(conversations)
        .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
    
    if (sortedConversations.length === 0) {
        historyContainer.innerHTML = '<p class="no-history">No chat history</p>';
        return;
    }
    
    historyContainer.innerHTML = '';
    sortedConversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = 'chat-history-item' + (conv.id === currentConversationId ? ' active' : '');
        item.onclick = () => loadConversation(conv.id);
        
        const title = document.createElement('span');
        title.className = 'chat-history-title';
        title.textContent = conv.title;
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'chat-history-delete';
        deleteBtn.textContent = 'x';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deleteConversation(conv.id);
        };
        
        item.appendChild(title);
        item.appendChild(deleteBtn);
        historyContainer.appendChild(item);
    });
}

function loadConversation(conversationId) {
    const conv = conversations[conversationId];
    if (!conv) return;
    
    currentConversationId = conversationId;
    currentModel = conv.model || currentModel;

    // Update model selector via Alpine combobox
    if (window._modelCombobox && conv.model) {
        window._modelCombobox.setModel(conv.model);
    } else if (conv.model) {
        currentModel = conv.model;
        updateCurrentModelDisplay();
    }

    // Restore backend selection
    if (window._agentSelector && conv.backend_id) {
        window._agentSelector.setBackend(conv.backend_id);
    } else if (conv.backend_id) {
        currentBackend = conv.backend_id;
    }
    
    // Clear and render messages
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.innerHTML = '';

    conv.messages.forEach(msg => {
        addMessage(msg.role, msg.content, false);
    });

    // Re-seed context tracking from saved messages (text-only, good enough estimate)
    localConvMessages = conv.messages.map(m => ({ role: m.role, content: m.content }));
    updateContextBar();

    renderChatHistory();
}

function deleteConversation(conversationId) {
    delete conversations[conversationId];
    saveConversationsToStorage();
    
    if (currentConversationId === conversationId) {
        startNewChat();
    }
    
    renderChatHistory();
}

function getCurrentMessages() {
    const messagesContainer = document.getElementById('messages-container');
    const messageElements = messagesContainer.querySelectorAll('.message');
    const messages = [];
    
    messageElements.forEach(el => {
        const role = el.classList.contains('user') ? 'user' : 'assistant';
        const contentEl = el.querySelector('.message-content');
        if (contentEl) {
            // Get text content, removing progress indicators
            const clone = contentEl.cloneNode(true);
            const progressContainer = clone.querySelector('.progress-container');
            if (progressContainer) progressContainer.remove();
            messages.push({ role, content: clone.textContent.trim() });
        }
    });
    
    return messages;
}

document.addEventListener('DOMContentLoaded', () => {
    // Models are loaded by Alpine.js modelCombobox on init
    loadConversationsFromStorage();
    renderChatHistory();
});
