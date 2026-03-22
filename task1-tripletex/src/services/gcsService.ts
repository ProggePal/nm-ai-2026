import { Storage } from '@google-cloud/storage';

const storage = new Storage();
const BUCKET = 'tripletex-solve-requests';

export interface RunSummary {
  filename: string;
  timestamp: string;
  prompt: string;
  model: string;
  elapsedMs: number;
  toolCallCount: number;
  errorCount: number;
  verified: boolean | null;
  summary: string | null;
  score: { score_raw: number; score_max: number; normalized_score: number; feedback?: any } | null;
}

// Simple in-memory cache (10s TTL)
let listCache: { data: RunSummary[]; expiry: number } | null = null;

export async function listRuns(limit = 50): Promise<RunSummary[]> {
  if (listCache && Date.now() < listCache.expiry) return listCache.data;

  // Fetch ALL result files (no maxResults limit), then sort and take latest
  const [files] = await storage.bucket(BUCKET).getFiles({ prefix: 'result-' });

  // Sort by name descending (names contain timestamps) — newest first
  files.sort((a, b) => b.name.localeCompare(a.name));

  const runs: RunSummary[] = [];
  for (const file of files.slice(0, limit)) {
    try {
      const [content] = await file.download();
      const data = JSON.parse(content.toString());
      runs.push({
        filename: file.name,
        timestamp: file.name.replace('result-', '').replace('.json', '').replace(/T/, ' ').replace(/-/g, (m, i) => i > 9 ? ':' : '-'),
        prompt: data.prompt || '',
        model: data.model || 'unknown',
        elapsedMs: data.elapsedMs || 0,
        toolCallCount: data.toolCallCount || 0,
        errorCount: (data.errors || []).length,
        verified: data.verification?.verified ?? null,
        summary: data.verification?.summary ?? null,
        score: data.score ? { score_raw: data.score.score_raw, score_max: data.score.score_max, normalized_score: data.score.normalized_score, feedback: data.score.feedback } : null,
      });
    } catch {
      // skip corrupt files
    }
  }

  listCache = { data: runs, expiry: Date.now() + 10_000 };
  return runs;
}

export async function getRunDetail(filename: string): Promise<object | null> {
  if (!/^result-[\w-]+\.json$/.test(filename)) return null;

  try {
    const [content] = await storage.bucket(BUCKET).file(filename).download();
    return JSON.parse(content.toString());
  } catch {
    return null;
  }
}
