import { useState } from "react";
import type { BackendStatus } from "../lib/hooks";
import type { ThemeName } from "../lib/theme";
import type { Health } from "../types";

interface Props {
  status: BackendStatus;
  health: Health | null;
  apiUrl: string;
  onApiUrlChange: (url: string | null) => void;
  theme: ThemeName;
  onToggleTheme: () => void;
}

const REPO_URL = "https://github.com/peerzadarabet1807/frostguard-scab";

// [full label, compact phone label]
const STATUS_TEXT: Record<BackendStatus, [string, string]> = {
  connecting: ["Connecting", "Connecting"],
  waking: ["Waking backend", "Waking"],
  online: ["Backend online", "Online"],
  offline: ["Backend offline", "Offline"],
};

export function Header({ status, health, apiUrl, onApiUrlChange, theme, onToggleTheme }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(apiUrl);

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <a className="brand" href="./" aria-label="FrostGuard-Scab home">
          <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
            <path d="M16 9c-3-3-10-2-10 6 0 7 5 13 10 13s10-6 10-13c0-8-7-9-10-6z" fill="var(--brand-apple)" />
            <path d="M16 9c0-3 1-5 4-6" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
            <circle cx="12" cy="17" r="2" fill="var(--brand-lesion)" />
            <circle cx="19" cy="21" r="1.6" fill="var(--brand-lesion)" />
          </svg>
          <span>
            <strong>FrostGuard-Scab</strong>
            <small>Apple scab &amp; frost early warning · Kashmir</small>
          </span>
        </a>

        <div className="topbar-actions">
          <button
            type="button"
            className={`status-pill status-${status}`}
            onClick={() => {
              setDraft(apiUrl);
              setEditing((v) => !v);
            }}
            aria-expanded={editing}
            title={`API: ${apiUrl}`}
          >
            <span className="dot" aria-hidden="true" />
            <span className="hide-sm">{STATUS_TEXT[status][0]}</span>
            <span className="show-sm">{STATUS_TEXT[status][1]}</span>
            {status === "online" && health && (
              <span className="pill-meta">
                {health.vision_backend === "trained" ? "YOLO11n" : "fallback model"} · v{health.version}
              </span>
            )}
          </button>
          <button type="button" className="icon-btn" onClick={onToggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
            {theme === "dark" ? "☀" : "☾"}
          </button>
          <a className="icon-btn" href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Source code on GitHub">
            <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">
              <path
                fill="currentColor"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
              />
            </svg>
          </a>
        </div>
      </div>

      {editing && (
        <form
          className="api-editor"
          onSubmit={(e) => {
            e.preventDefault();
            onApiUrlChange(draft.trim() || null);
            setEditing(false);
          }}
        >
          <label htmlFor="api-url">Backend URL</label>
          <input id="api-url" type="url" value={draft} onChange={(e) => setDraft(e.target.value)} spellCheck={false} />
          <button type="submit" className="btn btn-primary">
            Connect
          </button>
          <button type="button" className="btn" onClick={() => (onApiUrlChange(null), setEditing(false))}>
            Reset to default
          </button>
          <a className="btn-link" href={`${apiUrl}/docs`} target="_blank" rel="noreferrer">
            API docs ↗
          </a>
        </form>
      )}
    </header>
  );
}
