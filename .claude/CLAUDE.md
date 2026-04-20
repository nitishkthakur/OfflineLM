## Functionality:

### Search Functionality

- If the user clicks search toggle:
  
  - then the user query -> Tavily search wrapper for LLM -> (Groq / Ollama selected model on screen) -> stream on screen.
  
  - Number of search results to be fed to tavily to be picked from screen

### PDF Attachment

- User attaches a PDF → backend (`POST /upload`) extracts text (pypdf) and page images (pdf2image, 150 DPI, max 10 pages, base64 PNG).
- **PDF Vision Toggle** (left sidebar, Generation section, ON by default):
  - ON + model has `vision` capability → first message sent as **multimodal content** (`message_content` list: `[{type:text,...}, {type:image_url,...}×N]`).
  - OFF, or model lacks `vision` → text-only with XML-tagged extracted text only.
  - Backend safety net (`_strip_images()` in `react_agent.py`) also strips images if model lacks `vision`, regardless of frontend toggle.
  - If toggle is ON but model has no vision, badge turns **amber** (warn state) — images will be silently dropped.
- **PDF persistence**: `message_content` (or plain text fallback) is stored in `conversations[id]["messages"]` on the backend. Subsequent messages in the same conversation naturally include the PDF via conversation history — no re-injection needed.
- PDF export (`/download-pdf`) handles multimodal content: extracts text blocks, notes `[N page image(s) attached]`.

### PDF Image Provenance — Numbered Image References

**Caveat — Ollama flattens multimodal content:**
Ollama's `/api/chat` wire format is:
```json
{ "content": "<all text concatenated>", "images": ["b64_1", "b64_2", ...] }
```
LangChain's `ChatOllama` adapter concatenates ALL `type:text` blocks into one string and collects ALL `type:image_url` blocks into a flat `images[]` array. Positional interleaving — placing a caption text block immediately before its corresponding image block — is **structurally lost** before the HTTP request is even sent. This means the "interleaved caption" approach (`[caption_text, image, caption_text, image, ...]`) does NOT work for any Ollama-served local vision model (qwen2-vl, gemma3, llava-series, moondream, etc.).

Cloud/OpenAI-compatible APIs (Groq, OpenAI, Anthropic) DO preserve interleaved content lists. However, since we need a single approach that works across both local Ollama and cloud backends:

**Decision — Numbered image references in the text preamble:**
Build an explicit numbered map inside the text content, BEFORE the images:
```
[Image 1 = Page 1 of document.pdf]
[Image 2 = Page 2 of document.pdf]
...
```
Then append all image blocks in the same order after the text block. Every Ollama vision model processes the flat `images[]` array in order and can correlate "Image N" → Nth entry. Cloud models handle the same numbered references correctly.

**Implementation** (in `sendMessage()`, `frontend/app.js`):
- When `sendImages` is true: build `imagePreamble` string from `images.map((_, i) => \`[Image ${i+1} = Page ${i+1} of ${safeFilename}]\`)`, prepend it to the user question in the text block, then append all `type:image_url` blocks after.
- When `sendImages` is false: text-only path unchanged (XML-tagged extracted text + user question).
- `generateConversationTitle()` strips both `<pdf-filename-...>` tags AND `[Image N = ...]` preamble lines before deriving the conversation title.

### Model Capability Detection

- On every model selection, `GET /model-info/{model_id}` pings Ollama `/api/show` (single call) and returns:
  - `capabilities[]`, `supports_thinking`, `supports_vision`, `supports_tools`, `context_length`.
- Left sidebar shows **capability tags** (Tools / Vision / Thinking) — cyan=active, dim=inactive.
- `num_ctx` input max is clamped to model's native `context_length`; user can never exceed it.
- Thinking badge: green "supported" / gray "not supported". Vision badge: same + amber "warn" when toggle ON but no vision.

### Context Status Bar (right trace sidebar)

- Fixed strip above the trace config, always visible when sidebar is open.
- Shows: `~<used> / <effective_total>` tokens + progress bar.
- **Effective total** = `min(user-set num_ctx, model's context_length from Ollama)`.
- **Token estimation** (approximate):
  - Text: `chars ÷ 4`
  - Images: `1000 tokens/page` (at 150 DPI, ~1240×1754px → ~12 tiles × 85 tok)
  - System prompt overhead: `150 tokens`
- Updates after: model switch, every message sent/received, PDF upload/remove, num_ctx change.
- Bar color: cyan (normal) → amber >65% → red >85%.
- `localConvMessages[]` in `app.js` mirrors conversation for counting; reset on new chat.

### Generation Hyperparameters (left sidebar)

- `temperature` (default 0.8), `num_ctx` (default 2048, capped to model max), `num_predict` (default -1 = unlimited), `reasoning` toggle.
- All sent as `ollama_options` with every `/chat` request.
- Safety checks in `react_agent.py`: single `/api/show` call per request gates both thinking and tools capability before invoking `create_agent`.

### Thought Trace Sidebar (right)

- Streams `agent_turn`, `think`, `tool_call`, `tool_result`, `answer`, `progress`, `error` bubbles.
- Accumulates across messages; time-stamped separators per message.
- Configurable truncation limits; "Show more" expand; Maximize opens full HTML in new tab.
- Collapses to 38px strip with vertical "TRACE" label.

### Python Code Execution Sandbox (`python_execute` tool)

- **Architecture**: Persistent Jupyter-style kernel — one subprocess per conversation, reusing a `globals` dict across cells. Host ↔ jail protocol: JSON lines on stdin/stdout, prefixed `<<<SANDBOX>>>` so user-printed output doesn't pollute the protocol.
- **Sandbox mode**: `bwrap_works()` probed at runtime (not just `which bwrap`). On this host bwrap is blocked by AppArmor (`kernel.apparmor_restrict_unprivileged_userns=1`), so it falls back to `rlimit_only` mode (RLIMIT_AS/CPU/NPROC/FSIZE via `preexec_fn`).
- **SIGINT isolation fix**: Sandbox child is launched with `start_new_session=True` in `subprocess.Popen` so it's in its own PGID and doesn't receive SIGINT signals from uvicorn's process group. Additionally, `sandbox_runner.py` ignores SIGINT (`signal.signal(signal.SIGINT, signal.SIG_IGN)`) as belt-and-suspenders. Only SIGALRM (via `signal.alarm(timeout)`) drives cell timeouts.
- **Preinstalled packages** (in `.sandbox-venv`, managed by `ensure_sandbox_venv` in `sandbox.py`):
  - `ipykernel`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `openpyxl`, `matplotlib`, `requests`
  - Packages list is authoritative in `DEFAULT_PACKAGES` (sandbox.py) and mirrored in `config.json → code_execution.packages`
  - The `python_execute` docstring in `tools.py` lists all packages with purpose + warns against reaching for unavailable ones (statsmodels, torch, seaborn, plotly, etc.)
- **Venv marker**: `.sandbox-venv/.pkgs-installed` caches the sorted package list; venv is only rebuilt when the list changes.
- **Config** (`config.json → code_execution`): `enabled`, `timeout_seconds` (30), `max_output_chars` (10000), `idle_kill_minutes` (10), `sandbox_venv` (".sandbox-venv").
- **SandboxRegistry** in `sandbox.py`: keyed by `(conversation_id, allow_network)`. Network toggle change → kernel restart (state cleared). Idle kernels killed after `idle_kill_minutes`.















Design Perspective - Professional & Elegant UIs
Typography & Visual Hierarchy

    Font System: Use system fonts (SF Pro, Segoe UI, Roboto) or professional typefaces like Inter, Source Sans Pro
    Hierarchy: Establish clear type scale (H1: 32px, H2: 24px, H3: 20px, Body: 16px, Small: 14px)
    Font Weights: Limit to 2-3 weights maximum (Regular 400, Medium 500, Semibold 600)
    Line Height: Use 1.4-1.6 for body text, 1.2-1.3 for headings
    Letter Spacing: Subtle tracking on headings (-0.01em to 0.02em)

Color Strategy

    Primary Palette: Choose 1-2 core brand colors with 5-7 tonal variations each
    Neutral Scale: 7-9 carefully calibrated grays from white to near-black
    Semantic Colors: Success (green), warning (amber), error (red), info (blue)
    Text Colors: High contrast ratios (4.5:1 minimum, 7:1 preferred)
    Background Strategy: Subtle off-whites/grays instead of pure white

Layout & Spacing

    Grid System: 12-column responsive grid with consistent gutters
    Spacing Scale: Geometric progression (4px, 8px, 12px, 16px, 24px, 32px, 48px, 64px)
    Component Padding: Internal spacing follows spacing scale consistently
    Section Margins: Generous whitespace between major sections (48px-96px)
    Container Widths: Max-width constraints (1200px-1440px) with proper centering

Visual Elements

    Borders: 1px solid borders, subtle border-radius (4px-8px maximum)
    Shadows: Minimal, realistic shadows (0 1px 3px rgba(0,0,0,0.1))
    Icons: Consistent icon family (Heroicons, Lucide, Feather), 16px-24px sizes
    No Decorative Elements: Avoid emojis, excessive gradients, or playful illustrations
    Subtle Interactions: Gentle hover states, smooth transitions (200-300ms)

Content Organization

    Information Architecture: Clear navigation hierarchies, logical grouping
    Data Tables: Clean headers, alternating row colors, proper alignment
    Forms: Logical field grouping, clear labels, validation states
    Cards: Consistent padding, clear content hierarchy, subtle elevation

Coding Perspective - Technical Implementation
Technology Stack

    Default Framework: Alpine.js for reactive behavior, HTMX for server interactions
    CSS Strategy: Utility-first approach (Tailwind CSS) with custom component classes
    Progressive Enhancement: Core functionality works without JavaScript
    Semantic HTML: Proper HTML5 elements, ARIA attributes for accessibility

Performance Optimization

    Asset Loading: Compress images, use WebP format, lazy loading for below-fold content
    CSS Optimization: Purge unused styles, critical CSS inlining
    JavaScript: Minimal bundle sizes, tree-shaking, async/defer loading
    Caching Strategy: Proper cache headers, versioned assets

Responsive Design

    Mobile-First: Design for smallest screen, progressively enhance
    Breakpoints: 640px (sm), 768px (md), 1024px (lg), 1280px (xl)
    Flexible Layouts: CSS Grid and Flexbox for adaptive components
    Touch Targets: Minimum 44px clickable areas on mobile
    Content Strategy: Hide/show content appropriately across devices

Cross-Browser Compatibility

    Browser Support: Modern browsers (Chrome, Firefox, Safari, Edge)
    CSS Fallbacks: Graceful degradation for newer CSS features
    Testing: Regular testing across target browsers and devices
    Polyfills: Minimal, targeted polyfills only when necessary

Code Quality & Maintenance

    Component Architecture: Reusable, self-contained components
    State Management: Clear data flow, minimal global state
    Error Handling: Graceful error states, user-friendly messages
    Documentation: Code comments for complex logic, component documentation
    Accessibility: WCAG 2.1 AA compliance, keyboard navigation, screen reader support
