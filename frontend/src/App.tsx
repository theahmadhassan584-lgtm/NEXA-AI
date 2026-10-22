import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type KeyboardEvent,
} from 'react';
import {
  Activity,
  ArrowUp,
  Bot,
  CheckCircle2,
  CircleAlert,
  FileText,
  Globe,
  Loader2,
  Menu,
  MessageSquare,
  Paperclip,
  Plus,
  Settings2,
  ShieldCheck,
  Sparkles,
  Wrench,
  X,
} from 'lucide-react';
import type {
  AgentStatus,
  ChatMessage,
  ChatResponse,
  FileUploadResponse,
  Language,
  SessionInfo,
  ToolDefinition,
} from './types';

type HealthResponse = {
  status: string;
  version: string;
  timestamp: string;
};

type ApiError = {
  error?: string;
  detail?: string;
};

type SessionMessagesResponse = {
  session_id: string;
  messages: Array<{
    role: string;
    content: string;
    timestamp?: string | null;
    tool_name?: string | null;
  }>;
};

type Notice = {
  kind: 'success' | 'error' | 'info';
  text: string;
};

async function requestJson<T>(
  url: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(url, init);
  const raw = await response.text();

  let payload: unknown = {};

  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = {};
    }
  }

  if (!response.ok) {
    const data = payload as ApiError;
    const message =
      (typeof data.error === 'string' && data.error) ||
      (typeof data.detail === 'string' && data.detail) ||
      `Request failed with HTTP ${response.status}`;

    throw new Error(message);
  }

  return payload as T;
}

function readableError(error: unknown): string {
  return error instanceof Error ? error.message : 'An unexpected error occurred.';
}

function formatTime(timestamp?: string | null): string {
  if (!timestamp) return '';

  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return '';

  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

const suggestions = [
  {
    icon: Globe,
    title: 'Research a topic',
    description: 'Explore a question with available research tools.',
    prompt: 'Help me research an important topic and identify reliable sources.',
  },
  {
    icon: FileText,
    title: 'Analyze a document',
    description: 'Upload a supported file for processing.',
    prompt: 'What information can you extract from a document?',
  },
  {
    icon: Wrench,
    title: 'Explore my tools',
    description: 'See which agent tools are currently available.',
    prompt: 'Explain which tools are available to you and what they do.',
  },
  {
    icon: Sparkles,
    title: 'Solve a problem',
    description: 'Break a task into practical steps.',
    prompt: 'Help me break a complex task into clear, actionable steps.',
  },
];

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [input, setInput] = useState('');
  const [language, setLanguage] = useState<Language>('auto');
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [toolSettingsOpen, setToolSettingsOpen] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refreshSessions() {
    try {
      const result = await requestJson<{ sessions: SessionInfo[] }>(
        '/api/sessions',
      );
      setSessions(result.sessions ?? []);
    } catch {
      // Session history is optional when the backend is unavailable.
    }
  }

  async function refreshDashboard() {
    setRefreshing(true);

    const results = await Promise.allSettled([
      requestJson<HealthResponse>('/api/health'),
      requestJson<AgentStatus>('/api/status'),
      requestJson<{ tools: ToolDefinition[] }>('/api/tools'),
      requestJson<{ sessions: SessionInfo[] }>('/api/sessions'),
    ]);

    const healthResult = results[0];
    const statusResult = results[1];
    const toolsResult = results[2];
    const sessionsResult = results[3];

    if (healthResult.status === 'fulfilled') {
      setHealth(healthResult.value);
      setBackendAvailable(healthResult.value.status === 'ok');
    } else {
      setBackendAvailable(false);
      setNotice({
        kind: 'error',
        text: 'Cannot connect to the backend. Check that FastAPI is running on port 8000.',
      });
    }

    if (statusResult.status === 'fulfilled') {
      setStatus(statusResult.value);
    }

    if (toolsResult.status === 'fulfilled') {
      setTools(toolsResult.value.tools ?? []);
    }

    if (sessionsResult.status === 'fulfilled') {
      setSessions(sessionsResult.value.sessions ?? []);
    }

    setRefreshing(false);
  }

  useEffect(() => {
    void refreshDashboard();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, busy]);

  function startNewChat() {
    setSessionId(undefined);
    setMessages([]);
    setInput('');
    setNotice(null);
  }

  async function openSession(id: string) {
    if (busy) return;

    try {
      const result = await requestJson<SessionMessagesResponse>(
        `/api/sessions/${encodeURIComponent(id)}/messages`,
      );

      setSessionId(result.session_id);
      setMessages(
        result.messages.map((message) => ({
          role: message.role === 'user' ? 'user' : 'assistant',
          content: message.content,
          timestamp: message.timestamp ?? undefined,
        })),
      );
      setNotice(null);
    } catch (error) {
      setNotice({ kind: 'error', text: readableError(error) });
    }
  }

  async function sendMessage(event?: FormEvent) {
    event?.preventDefault();

    const text = input.trim();
    if (!text || busy) return;

    const history = messages
      .filter((message) => message.role === 'user' || message.role === 'assistant')
      .map((message) => ({
        role: message.role,
        content: message.content,
      }));

    const userMessage: ChatMessage = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessages((previous) => [...previous, userMessage]);
    setInput('');
    setBusy(true);
    setNotice(null);

    try {
      const result = await requestJson<ChatResponse>('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
          history,
          language,
          enabled_tools: tools
            .filter((tool) => tool.enabled)
            .map((tool) => tool.name),
        }),
      });

      setSessionId(result.session_id);

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: result.reply,
          timestamp: new Date().toISOString(),
          toolCalls: result.tool_calls,
        },
      ]);

      await refreshSessions();

      if (result.model) {
        setNotice({
          kind: 'success',
          text: `Response received from ${result.model}.`,
        });
      }
    } catch (error) {
      setNotice({
        kind: 'error',
        text: readableError(error),
      });
    } finally {
      setBusy(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  async function toggleTool(tool: ToolDefinition) {
    const nextEnabled = !tool.enabled;

    setTools((previous) =>
      previous.map((item) =>
        item.name === tool.name ? { ...item, enabled: nextEnabled } : item,
      ),
    );

    try {
      await requestJson<{ tool: string; enabled: boolean }>(
        `/api/tools/toggle/${encodeURIComponent(tool.name)}?enabled=${nextEnabled}`,
        { method: 'POST' },
      );

      setNotice({
        kind: 'success',
        text: `${tool.name} ${nextEnabled ? 'enabled' : 'disabled'}.`,
      });
    } catch (error) {
      setTools((previous) =>
        previous.map((item) =>
          item.name === tool.name ? { ...item, enabled: !nextEnabled } : item,
        ),
      );

      setNotice({ kind: 'error', text: readableError(error) });
    }
  }

  async function uploadDocument(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setNotice({ kind: 'info', text: `Uploading ${file.name}…` });

    try {
      const result = await requestJson<FileUploadResponse>('/api/upload', {
        method: 'POST',
        body: formData,
      });

      setNotice({
        kind: 'success',
        text: `Uploaded ${result.filename} (${Math.ceil(result.size_bytes / 1024)} KB). The backend has stored the file.`,
      });
    } catch (error) {
      setNotice({ kind: 'error', text: readableError(error) });
    } finally {
      event.target.value = '';
    }
  }

  const enabledToolCount = tools.filter((tool) => tool.enabled).length;

  return (
    <div className={`app-shell ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
      <aside className={`sidebar ${sidebarOpen ? '' : 'sidebar-hidden'}`}>
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={21} strokeWidth={2.2} />
          </div>
          <div className="brand-copy">
            <span className="brand-name">NEXA<span>AI</span></span>
            <span className="brand-subtitle">INTELLIGENT AGENT</span>
          </div>
          <button
            className="icon-button sidebar-close"
            title="Close sidebar"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={17} />
          </button>
        </div>

        <button className="new-chat-button" onClick={startNewChat}>
          <Plus size={18} />
          <span>New conversation</span>
          <span className="shortcut">+</span>
        </button>

        <div className="sidebar-section">
          <div className="sidebar-heading">
            <span>WORKSPACE</span>
          </div>
          <div className="sidebar-static-item active">
            <MessageSquare size={16} />
            <span>AI Assistant</span>
          </div>
          <button
            className="sidebar-static-item"
            onClick={() => setToolSettingsOpen((previous) => !previous)}
          >
            <Settings2 size={16} />
            <span>Agent tools</span>
            <span className="item-count">{enabledToolCount}</span>
          </button>
        </div>

        <div className="sidebar-section session-section">
          <div className="sidebar-heading">
            <span>RECENT CONVERSATIONS</span>
            <button
              className="tiny-icon-button"
              title="Refresh history"
              onClick={() => void refreshSessions()}
            >
              <Activity size={13} />
            </button>
          </div>

          <div className="session-list">
            {sessions.length === 0 ? (
              <p className="sidebar-empty">Your conversations will appear here.</p>
            ) : (
              sessions.map((session) => (
                <button
                  key={session.session_id}
                  className={`session-item ${sessionId === session.session_id ? 'selected' : ''}`}
                  onClick={() => void openSession(session.session_id)}
                  title={`${session.message_count} messages`}
                >
                  <MessageSquare size={15} />
                  <span>
                    Session {session.session_id.slice(0, 8)}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>

        <div className="sidebar-section tools-section">
          <button
            className="sidebar-heading heading-button"
            onClick={() => setToolSettingsOpen((previous) => !previous)}
          >
            <span>AVAILABLE TOOLS</span>
            <Settings2 size={13} />
          </button>

          {toolSettingsOpen && (
            <div className="tool-list">
              {tools.length === 0 ? (
                <p className="sidebar-empty">No tools loaded.</p>
              ) : (
                tools.map((tool) => (
                  <label className="tool-row" key={tool.name}>
                    <span className={`tool-status ${tool.enabled ? 'enabled' : ''}`} />
                    <span className="tool-label" title={tool.description}>
                      {tool.name.replace(/_/g, ' ')}
                    </span>
                    <input
                      aria-label={`Enable ${tool.name}`}
                      type="checkbox"
                      checked={tool.enabled}
                      onChange={() => void toggleTool(tool)}
                    />
                  </label>
                ))
              )}
            </div>
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="security-note">
            <ShieldCheck size={17} />
            <span>Privacy-conscious by design</span>
          </div>
          <div className="profile-row">
            <div className="profile-avatar">N</div>
            <div className="profile-copy">
              <strong>NEXA AI</strong>
              <span>Personal workspace</span>
            </div>
            <Sparkles size={16} className="profile-sparkle" />
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-left">
            {!sidebarOpen && (
              <button
                className="icon-button"
                title="Open sidebar"
                onClick={() => setSidebarOpen(true)}
              >
                <Menu size={19} />
              </button>
            )}
            <div className="breadcrumb">
              <span>NEXA AI</span>
              <span className="breadcrumb-divider">/</span>
              <strong>AI Assistant</strong>
            </div>
          </div>

          <div className="topbar-right">
            <label className="language-select">
              <Globe size={15} />
              <select
                value={language}
                onChange={(event) => setLanguage(event.target.value as Language)}
                aria-label="Response language"
              >
                <option value="auto">Auto language</option>
                <option value="en">English</option>
                <option value="ur">Urdu / Roman Urdu</option>
              </select>
            </label>

            <div className={`connection-state ${backendAvailable ? 'connected' : backendAvailable === false ? 'disconnected' : ''}`}>
              <span className="connection-dot" />
              {backendAvailable === null
                ? 'Connecting'
                : backendAvailable
                  ? 'Backend online'
                  : 'Offline'}
            </div>

            <button
              className="icon-button refresh-button"
              title="Refresh agent status"
              disabled={refreshing}
              onClick={() => void refreshDashboard()}
            >
              <Activity size={17} className={refreshing ? 'spin' : ''} />
            </button>
          </div>
        </header>

        <div className="workspace">
          <section className="conversation-area">
            <div className="conversation-header">
              <div className="conversation-title">
                <div className="mini-brand-icon">
                  <Bot size={19} />
                </div>
                <div>
                  <h1>AI Assistant</h1>
                  <p>
                    {status?.model
                      ? `Powered by ${status.model}`
                      : 'Your intelligent workspace'}
                  </p>
                </div>
              </div>

              <div className="header-tags">
                <span className="small-tag"><ShieldCheck size={13} /> Tool controls</span>
                <span className="small-tag"><Activity size={13} /> Multi-turn chat</span>
              </div>
            </div>

            {backendAvailable && status && !status.api_key_configured && (
              <div className="api-warning">
                <CircleAlert size={19} />
                <div>
                  <strong>Claude API key not configured</strong>
                  <p>
                    Add a valid ANTHROPIC_API_KEY to the backend environment
                    configuration and restart the backend to enable AI responses.
                    Never put the secret key in frontend code.
                  </p>
                </div>
              </div>
            )}

            {notice && (
              <div className={`notice notice-${notice.kind}`} role="status">
                {notice.kind === 'success' ? (
                  <CheckCircle2 size={17} />
                ) : notice.kind === 'error' ? (
                  <CircleAlert size={17} />
                ) : (
                  <Activity size={17} />
                )}
                <span>{notice.text}</span>
                <button
                  className="notice-close"
                  title="Dismiss"
                  onClick={() => setNotice(null)}
                >
                  <X size={15} />
                </button>
              </div>
            )}

            <div className="messages-scroll">
              {messages.length === 0 ? (
                <div className="welcome-screen">
                  <div className="welcome-orb">
                    <div className="orb-core"><Sparkles size={31} /></div>
                    <span className="orb-ring ring-one" />
                    <span className="orb-ring ring-two" />
                  </div>

                  <span className="welcome-eyebrow">YOUR INTELLIGENT WORKSPACE</span>
                  <h2>What can we work on<br /><span>together today?</span></h2>
                  <p className="welcome-description">
                    Ask a question, investigate a problem, or work through a
                    task. NEXA will use the enabled tools available to it.
                  </p>

                  <div className="suggestion-grid">
                    {suggestions.map((item) => {
                      const Icon = item.icon;

                      return (
                        <button
                          key={item.title}
                          className="suggestion-card"
                          onClick={() => setInput(item.prompt)}
                        >
                          <span className="suggestion-icon"><Icon size={18} /></span>
                          <span className="suggestion-text">
                            <strong>{item.title}</strong>
                            <span>{item.description}</span>
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  <div className="welcome-footnote">
                    <ShieldCheck size={15} />
                    Review tool permissions and verify important results.
                  </div>
                </div>
              ) : (
                <div className="message-list">
                  {messages.map((message, index) => (
                    <article
                      key={`${message.timestamp ?? index}-${index}`}
                      className={`message-row ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
                    >
                      <div className={`message-avatar ${message.role === 'user' ? 'user-avatar' : 'assistant-avatar'}`}>
                        {message.role === 'user' ? <span>Y</span> : <Bot size={19} />}
                      </div>

                      <div className="message-body">
                        <div className="message-meta">
                          <strong>{message.role === 'user' ? 'You' : 'NEXA AI'}</strong>
                          <span>{formatTime(message.timestamp)}</span>
                        </div>

                        <div className="message-content">{message.content}</div>

                        {message.toolCalls && message.toolCalls.length > 0 && (
                          <div className="tool-call-list">
                            {message.toolCalls.map((call, callIndex) => (
                              <div className="tool-call-card" key={`${call.tool_name}-${callIndex}`}>
                                <div className="tool-call-icon"><Wrench size={15} /></div>
                                <div className="tool-call-copy">
                                  <strong>{call.tool_name.replace(/_/g, ' ')}</strong>
                                  <span>
                                    {call.success ? 'Completed' : call.error || 'Tool failed'}
                                    {typeof call.duration_ms === 'number' && ` · ${call.duration_ms} ms`}
                                  </span>
                                </div>
                                <span className={`tool-call-result ${call.success ? 'ok' : 'bad'}`}>
                                  {call.success ? 'Success' : 'Failed'}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </article>
                  ))}

                  {busy && (
                    <div className="message-row assistant-message">
                      <div className="message-avatar assistant-avatar"><Bot size={19} /></div>
                      <div className="message-body">
                        <div className="message-meta"><strong>NEXA AI</strong></div>
                        <div className="typing-indicator">
                          <Loader2 size={17} className="spin" />
                          <span>Thinking and processing your request…</span>
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={bottomRef} />
                </div>
              )}
            </div>

            <div className="composer-section">
              <form className="composer" onSubmit={(event) => void sendMessage(event)}>
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  placeholder="Message NEXA AI…"
                  rows={1}
                  maxLength={32000}
                  disabled={busy}
                  aria-label="Message NEXA AI"
                />

                <div className="composer-actions">
                  <div className="composer-left-actions">
                    <input
                      ref={fileRef}
                      className="hidden-input"
                      type="file"
                      accept=".pdf,.txt,.md,.csv,.json,.docx"
                      onChange={(event) => void uploadDocument(event)}
                    />
                    <button
                      type="button"
                      className="composer-tool-button"
                      title="Upload a document"
                      onClick={() => fileRef.current?.click()}
                      disabled={busy}
                    >
                      <Paperclip size={17} />
                      <span>Attach file</span>
                    </button>
                    <span className="composer-context">
                      <Wrench size={13} /> {enabledToolCount} tools enabled
                    </span>
                  </div>

                  <button
                    type="submit"
                    className="send-button"
                    disabled={busy || !input.trim()}
                    aria-label="Send message"
                  >
                    {busy ? <Loader2 size={17} className="spin" /> : <ArrowUp size={19} />}
                  </button>
                </div>
              </form>

              <p className="composer-disclaimer">
                AI can make mistakes. Verify important information before acting on it.
              </p>
            </div>
          </section>

          <aside className="insights-panel">
            <div className="insights-title">
              <span className="insights-title-icon"><Activity size={17} /></span>
              <div>
                <h2>Agent overview</h2>
                <p>Live configuration</p>
              </div>
            </div>

            <div className="overview-card">
              <div className="overview-card-top">
                <span className="overview-label">BACKEND HEALTH</span>
                <span className={`health-pill ${health?.status === 'ok' ? 'healthy' : 'unhealthy'}`}>
                  <span className="health-dot" />
                  {health?.status === 'ok' ? 'Healthy' : backendAvailable === false ? 'Offline' : 'Checking'}
                </span>
              </div>
              <div className="overview-main">
                <span className="overview-main-icon"><Activity size={19} /></span>
                <div>
                  <strong>{health?.status === 'ok' ? 'API is responding' : 'Checking connection'}</strong>
                  <span>{health?.version ? `API version ${health.version}` : 'FastAPI service'}</span>
                </div>
              </div>
              <div className="overview-detail">
                <span>Address</span>
                <code>127.0.0.1:8000</code>
              </div>
            </div>

            <div className="overview-card">
              <span className="overview-label">MODEL CONFIGURATION</span>
              <div className="model-row">
                <div className="model-icon"><Sparkles size={19} /></div>
                <div>
                  <strong>{status?.model || 'Claude model'}</strong>
                  <span>Anthropic provider</span>
                </div>
              </div>
              <div className={`credential-state ${status?.api_key_configured ? 'credential-ok' : 'credential-missing'}`}>
                {status?.api_key_configured ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}
                {status?.api_key_configured ? 'API key configured' : 'API key required'}
              </div>
            </div>

            <div className="overview-card tool-overview">
              <div className="overview-card-top">
                <span className="overview-label">TOOLS</span>
                <span className="tool-count">{enabledToolCount} / {tools.length}</span>
              </div>
              <div className="tool-meter">
                <span style={{ width: `${tools.length ? (enabledToolCount / tools.length) * 100 : 0}%` }} />
              </div>
              <p className="muted-copy">
                Toggle individual tools in the sidebar. Availability depends on the
                integration and its configuration.
              </p>
              <button className="text-button" onClick={() => setToolSettingsOpen(true)}>
                Configure tools <Settings2 size={14} />
              </button>
            </div>

            <div className="privacy-card">
              <ShieldCheck size={20} />
              <strong>Use tools thoughtfully</strong>
              <p>
                External content may be unreliable. Check sources and use caution
                before relying on consequential results.
              </p>
            </div>

            <div className="status-footer">
              <span className={`status-light ${backendAvailable ? 'light-on' : ''}`} />
              <span>{backendAvailable ? 'Connected to local backend' : 'Backend connection not confirmed'}</span>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}