import { Router, type Request, type Response } from 'express';
import { listRuns, getRunDetail } from '../services/gcsService.js';
import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPLAY_FILE = readFileSync(resolve(__dirname, '../replay.html'), 'utf-8');

export const dashboardRouter = Router();

// API endpoints
dashboardRouter.get('/api/runs', async (_req: Request, res: Response) => {
  try {
    const runs = await listRuns(50);
    res.json(runs);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

dashboardRouter.get('/api/runs/:filename', async (req: Request, res: Response) => {
  try {
    const data = await getRunDetail(req.params.filename);
    if (!data) return res.status(404).json({ error: 'Not found' });
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

// Score upload — match score to closest run by timestamp
dashboardRouter.post('/api/score', async (req: Request, res: Response) => {
  try {
    const score = req.body;
    if (!score || !score.completed_at) {
      return res.status(400).json({ error: 'Missing completed_at in score JSON' });
    }

    // Find closest result file by matching completed_at timestamp
    const runs = await listRuns(100);
    const scoreTime = new Date(score.completed_at).getTime();
    const scoreDuration = score.duration_ms || 0;
    const scoreStartTime = scoreTime - scoreDuration;

    // Match: find run whose timestamp is closest to score's start or end time
    let bestMatch: string | null = null;
    let bestDiff = Infinity;

    for (const run of runs) {
      // Parse run timestamp from filename: result-2026-03-21T15-37-22-907Z.json
      const m = run.filename.match(/result-(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})/);
      if (!m) continue;
      const runTime = new Date(`${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}Z`).getTime();
      const diff = Math.abs(runTime - scoreTime);
      if (diff < bestDiff) {
        bestDiff = diff;
        bestMatch = run.filename;
      }
    }

    if (!bestMatch || bestDiff > 120000) { // within 2 minutes
      return res.status(404).json({ error: 'No matching run found', bestMatch, bestDiff });
    }

    // Download existing result, add score, re-upload
    const detail = await getRunDetail(bestMatch);
    if (!detail) return res.status(404).json({ error: 'Run detail not found' });

    const updated = { ...(detail as any), score };
    const { Storage } = await import('@google-cloud/storage');
    const storage = new Storage();
    await storage.bucket('tripletex-solve-requests').file(bestMatch).save(
      JSON.stringify(updated, null, 2),
      { contentType: 'application/json' }
    );

    res.json({ ok: true, matched: bestMatch, diff: bestDiff });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

// Bulk score upload — array of scores
dashboardRouter.post('/api/scores', async (req: Request, res: Response) => {
  try {
    const scores = Array.isArray(req.body) ? req.body : [req.body];
    const results = [];
    for (const score of scores) {
      try {
        const resp = await fetch(`http://localhost:${process.env.PORT || 5000}/api/score`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(score),
        });
        results.push(await resp.json());
      } catch (e) {
        results.push({ error: String(e) });
      }
    }
    res.json(results);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

// Dashboard HTML
dashboardRouter.get('/dashboard', (_req: Request, res: Response) => {
  res.type('html').send(DASHBOARD_HTML);
});

// Replay view — chat-style replay of the latest run
dashboardRouter.get('/replay', (_req: Request, res: Response) => {
  res.type('html').send(REPLAY_FILE);
});

const DASHBOARD_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Accounting Agent — Live Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px; }
  a { color: #58a6ff; text-decoration: none; }

  .header { background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .live-dot { width: 8px; height: 8px; background: #3fb950; border-radius: 50%; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .header-right { margin-left: auto; color: #8b949e; font-size: 12px; }

  .stats { display: flex; gap: 16px; padding: 16px 24px; background: #161b22; border-bottom: 1px solid #30363d; }
  .stat { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; min-width: 140px; }
  .stat-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-value { font-size: 24px; font-weight: 600; margin-top: 4px; }
  .stat-value.green { color: #3fb950; }
  .stat-value.red { color: #f85149; }
  .stat-value.blue { color: #58a6ff; }

  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }

  .run-list { display: flex; flex-direction: column; gap: 8px; }

  .run-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; cursor: pointer; transition: border-color 0.2s; }
  .run-card:hover { border-color: #58a6ff; }
  .run-card.expanded { border-color: #58a6ff; }

  .run-summary { display: grid; grid-template-columns: 180px 1fr 80px 80px 90px 90px; align-items: center; padding: 12px 16px; gap: 12px; }
  .run-time { color: #8b949e; font-size: 12px; font-family: monospace; }
  .run-prompt { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .run-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; text-align: center; }
  .badge-tools { background: #1f2937; color: #93c5fd; }
  .badge-errors { background: #f8514922; color: #f85149; }
  .badge-errors.zero { background: #3fb95022; color: #3fb950; }
  .badge-time { background: #1f2937; color: #d1d5db; font-family: monospace; }
  .badge-verified { background: #3fb95022; color: #3fb950; }
  .badge-unverified { background: #f8514922; color: #f85149; }
  .badge-na { background: #30363d; color: #8b949e; }

  .run-detail { display: none; padding: 16px; border-top: 1px solid #30363d; }
  .run-card.expanded .run-detail { display: block; }

  .detail-section { margin-bottom: 16px; }
  .detail-section h3 { font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }

  .prompt-full { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.5; }

  .timeline { display: flex; flex-direction: column; gap: 4px; }
  .tl-item { display: flex; gap: 8px; padding: 6px 10px; border-radius: 4px; font-family: monospace; font-size: 12px; align-items: flex-start; }
  .tl-tool { background: #0d2137; border-left: 3px solid #58a6ff; }
  .tl-result { background: #0d1117; border-left: 3px solid #3fb950; color: #8b949e; }
  .tl-error { background: #1a0d0d; border-left: 3px solid #f85149; color: #f85149; }
  .tl-search { background: #0d1117; border-left: 3px solid #8b949e; color: #8b949e; font-style: italic; }
  .tl-text { background: #0d1117; border-left: 3px solid #d2a8ff; }
  .tl-label { font-weight: 600; min-width: 100px; flex-shrink: 0; }
  .tl-content { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 900px; }
  .tl-content.expanded { white-space: pre-wrap; word-break: break-all; }
  .tl-item:hover .tl-content { white-space: pre-wrap; word-break: break-all; }

  .model-tag { display: inline-block; background: #1f2937; border: 1px solid #30363d; padding: 2px 8px; border-radius: 4px; font-size: 11px; color: #d2a8ff; margin-left: 8px; }

  .empty { text-align: center; color: #8b949e; padding: 60px 20px; }
  .loading { text-align: center; color: #8b949e; padding: 40px; }
</style>
</head>
<body>

<div class="header">
  <div class="live-dot"></div>
  <h1>AI Accounting Agent</h1>
  <div class="header-right">Auto-refresh: 10s</div>
</div>

<div class="stats" id="stats">
  <div class="stat"><div class="stat-label">Total Runs</div><div class="stat-value blue" id="s-total">-</div></div>
  <div class="stat"><div class="stat-label">Avg Tool Calls</div><div class="stat-value" id="s-tools">-</div></div>
  <div class="stat"><div class="stat-label">Avg Errors</div><div class="stat-value" id="s-errors">-</div></div>
  <div class="stat"><div class="stat-label">Avg Time</div><div class="stat-value" id="s-time">-</div></div>
  <div class="stat"><div class="stat-label">Verified</div><div class="stat-value green" id="s-verified">-</div></div>
</div>

<div class="container">
  <div class="run-list" id="runs"><div class="loading">Loading runs...</div></div>
</div>

<script>
let runs = [];
let expanded = new Set();
let detailCache = {};

async function fetchRuns() {
  try {
    const res = await fetch('/api/runs');
    runs = await res.json();
    updateStats();
    renderRuns();
  } catch (e) {
    console.error('Failed to fetch runs:', e);
  }
}

function updateStats() {
  document.getElementById('s-total').textContent = runs.length;
  if (runs.length === 0) return;
  const avgTools = (runs.reduce((s, r) => s + r.toolCallCount, 0) / runs.length).toFixed(1);
  const avgErrors = (runs.reduce((s, r) => s + r.errorCount, 0) / runs.length).toFixed(1);
  const avgTime = (runs.reduce((s, r) => s + r.elapsedMs, 0) / runs.length / 1000).toFixed(1);
  const verified = runs.filter(r => r.verified === true).length;
  document.getElementById('s-tools').textContent = avgTools;
  document.getElementById('s-errors').textContent = avgErrors;
  document.getElementById('s-errors').className = 'stat-value ' + (parseFloat(avgErrors) > 1 ? 'red' : 'green');
  document.getElementById('s-time').textContent = avgTime + 's';
  document.getElementById('s-verified').textContent = verified + '/' + runs.length;
}

function renderRuns() {
  const el = document.getElementById('runs');
  if (runs.length === 0) { el.innerHTML = '<div class="empty">No runs yet. Submit tasks to see results here.</div>'; return; }

  el.innerHTML = runs.map(r => {
    const time = formatTimestamp(r.filename);
    const errClass = r.errorCount === 0 ? 'zero' : '';
    const vBadge = r.verified === true ? 'badge-verified' : r.verified === false ? 'badge-unverified' : 'badge-na';
    const vText = r.verified === true ? 'YES' : r.verified === false ? 'NO' : 'n/a';
    const isExp = expanded.has(r.filename);

    return '<div class="run-card ' + (isExp ? 'expanded' : '') + '" data-file="' + r.filename + '">'
      + '<div class="run-summary" onclick="toggle(\\'' + r.filename + '\\')">'
      + '<div class="run-time">' + time + '</div>'
      + '<div class="run-prompt">' + esc(r.prompt.slice(0, 100)) + '</div>'
      + '<div><span class="run-badge badge-tools">' + r.toolCallCount + ' calls</span></div>'
      + '<div><span class="run-badge badge-errors ' + errClass + '">' + r.errorCount + ' err</span></div>'
      + '<div><span class="run-badge badge-time">' + (r.elapsedMs / 1000).toFixed(1) + 's</span></div>'
      + '<div><span class="run-badge ' + vBadge + '">' + vText + '</span></div>'
      + '</div>'
      + '<div class="run-detail" id="detail-' + r.filename + '">' + (isExp ? renderDetail(detailCache[r.filename]) : '') + '</div>'
      + '</div>';
  }).join('');
}

async function toggle(filename) {
  if (expanded.has(filename)) {
    expanded.delete(filename);
    renderRuns();
    return;
  }

  expanded.clear();
  expanded.add(filename);

  if (!detailCache[filename]) {
    const el = document.getElementById('detail-' + filename);
    if (el) el.innerHTML = '<div class="loading">Loading details...</div>';
    try {
      const res = await fetch('/api/runs/' + filename);
      detailCache[filename] = await res.json();
    } catch (e) {
      detailCache[filename] = { error: String(e) };
    }
  }
  renderRuns();
}

function renderDetail(data) {
  if (!data) return '<div class="loading">Loading...</div>';
  if (data.error) return '<div style="color:#f85149">' + esc(data.error) + '</div>';

  let html = '';

  // Prompt
  html += '<div class="detail-section"><h3>Prompt</h3><div class="prompt-full">' + esc(data.prompt || '') + '</div></div>';

  // Model + system prompt info
  const sysModules = (data.systemPrompt || []).length;
  html += '<div class="detail-section"><h3>Config</h3>'
    + '<span class="model-tag">' + esc(data.model || 'unknown') + '</span>'
    + '<span class="model-tag">' + sysModules + ' prompt blocks</span>'
    + '<span class="model-tag">' + (data.elapsedMs / 1000).toFixed(1) + 's elapsed</span>'
    + '</div>';

  // Tool call timeline
  const timeline = extractTimeline(data.messages || []);
  if (timeline.length > 0) {
    html += '<div class="detail-section"><h3>Tool Timeline (' + timeline.length + ' events)</h3><div class="timeline">';
    for (const ev of timeline) {
      html += '<div class="tl-item ' + ev.cls + '">'
        + '<span class="tl-label">' + esc(ev.label) + '</span>'
        + '<span class="tl-content">' + esc(ev.content) + '</span>'
        + '</div>';
    }
    html += '</div></div>';
  }

  // Verification
  if (data.verification) {
    const v = data.verification;
    const color = v.verified ? '#3fb950' : '#f85149';
    html += '<div class="detail-section"><h3>Verification</h3>'
      + '<div style="color:' + color + ';font-weight:600">' + (v.verified ? 'VERIFIED' : 'NOT VERIFIED') + '</div>'
      + '<div style="margin-top:4px;color:#8b949e">' + esc(v.summary || '') + '</div>'
      + '</div>';
  }

  return html;
}

function extractTimeline(messages) {
  const events = [];
  for (const msg of messages) {
    if (!Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (block.type === 'mcp_tool_use') {
        events.push({ cls: 'tl-tool', label: block.name, content: JSON.stringify(block.input || {}) });
      } else if (block.type === 'mcp_tool_result') {
        const preview = JSON.stringify(block.content || '').slice(0, 200);
        events.push({ cls: block.is_error ? 'tl-error' : 'tl-result', label: block.is_error ? 'ERROR' : 'RESULT', content: preview });
      } else if (block.type === 'server_tool_use') {
        events.push({ cls: 'tl-search', label: 'SEARCH', content: JSON.stringify(block.input || {}) });
      } else if (block.type === 'text' && block.text) {
        events.push({ cls: 'tl-text', label: 'CLAUDE', content: block.text.slice(0, 300) });
      }
    }
  }
  return events;
}

function formatTimestamp(filename) {
  // result-2026-03-20T18-15-52-051Z.json -> 2026-03-20 18:15:52
  return filename.replace('result-', '').replace('.json', '')
    .replace('T', ' ').replace(/-/g, (m, offset, str) => {
      // First 10 chars are date (keep dashes), rest convert to colons
      return offset > 9 && offset < 19 ? ':' : m;
    }).slice(0, 19);
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Init
fetchRuns();
setInterval(fetchRuns, 10000);
</script>
</body>
</html>`;
