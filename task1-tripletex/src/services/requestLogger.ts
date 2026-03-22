import { Storage } from '@google-cloud/storage';
import type { AgentResult } from './claudeAgent.js';

const storage = new Storage();
const BUCKET = 'tripletex-solve-requests';

export async function logRequest(body: unknown): Promise<string> {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `request-${timestamp}.json`;

  const content = JSON.stringify(body, null, 2);

  // Always log to stdout (visible in Cloud Run logs)
  console.log(`[INCOMING REQUEST] ${filename}`);
  console.log(content);

  // Save to GCS
  try {
    await storage.bucket(BUCKET).file(filename).save(content, {
      contentType: 'application/json',
    });
    console.log(`[SAVED TO GCS] gs://${BUCKET}/${filename}`);
  } catch (err) {
    console.error(`[GCS SAVE FAILED]`, err);
  }

  return filename;
}

export async function logResult(
  requestFilename: string,
  prompt: string,
  agentResult: AgentResult,
  verification?: { verified: boolean; summary: string },
  elapsedMs?: number,
): Promise<void> {
  const resultFilename = requestFilename.replace('request-', 'result-');

  const payload = {
    prompt,
    model: 'claude-opus-4-6',
    elapsedMs,
    toolCallCount: agentResult.toolCallCount,
    errors: agentResult.errors,
    systemPrompt: (agentResult.systemPrompt || []).map((b: any) => b?.text || ''),
    messages: agentResult.messages,
    verification,
  };

  let content: string;
  try {
    content = JSON.stringify(payload, null, 2);
  } catch (err) {
    // If messages are too large or have circular refs, save without messages
    console.error('[RESULT SERIALIZE FAILED]', err);
    content = JSON.stringify({ ...payload, messages: '[too large to serialize]' }, null, 2);
  }

  // Always log summary to stdout
  console.log(`[RESULT] ${resultFilename} tools=${agentResult.toolCallCount} errors=${agentResult.errors.length} verified=${verification?.verified ?? 'n/a'}`);

  // Save to GCS
  try {
    await storage.bucket(BUCKET).file(resultFilename).save(content, {
      contentType: 'application/json',
    });
    console.log(`[SAVED TO GCS] gs://${BUCKET}/${resultFilename}`);
  } catch (err) {
    console.error(`[GCS RESULT SAVE FAILED]`, err);
  }
}
