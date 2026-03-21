import Anthropic from '@anthropic-ai/sdk';
import { buildSystemPrompt } from './systemPrompt.js';
import { TripletexApi } from './tripletexApi.js';
import { extractCode, executeCode } from './codeExecutor.js';
import type { TripletexCredentials } from '../types.js';

const claude = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const AGENT_TIMEOUT_MS = 4.5 * 60 * 1000; // 4.5 min

// Use streaming to avoid SDK timeout with high max_tokens
async function createMessage(params: {
  model: string;
  max_tokens: number;
  system: any;
  messages: any[];
}): Promise<Anthropic.Message> {
  const stream = claude.messages.stream(params);
  return await stream.finalMessage();
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AgentResult {
  toolCallCount: number;
  errors: Array<{ tool: string; status: number }>;
  messages: any[];
  systemPrompt: any[];
  generatedCode?: string;
  executionLogs?: string[];
  retried?: boolean;
}

// ---------------------------------------------------------------------------
// Main agent — code-execution approach
// ---------------------------------------------------------------------------

export async function runAgent(
  prompt: string,
  credentials: TripletexCredentials,
  imageAttachments: Array<{ mimeType: string; data: string }> = [],
  pdfAttachments: Array<{ filename: string; data: string }> = [],
): Promise<AgentResult> {
  const startTime = Date.now();
  const systemPrompt = buildSystemPrompt(prompt);

  // Build user message with attachments
  const content: any[] = [];

  for (const img of imageAttachments) {
    content.push({
      type: 'image',
      source: {
        type: 'base64',
        media_type: img.mimeType as 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp',
        data: img.data,
      },
    });
  }

  for (const pdf of pdfAttachments) {
    content.push({
      type: 'document',
      source: {
        type: 'base64',
        media_type: 'application/pdf',
        data: pdf.data,
      },
    });
  }

  content.push({ type: 'text', text: prompt });

  const messages: any[] = [{ role: 'user', content }];

  // --- Turn 1: Generate code ---
  console.log(`[AGENT] Calling Claude to generate code...`);
  const response = await createMessage({
    model: 'claude-opus-4-6',
    max_tokens: 32768,
    system: systemPrompt,
    messages,
  });

  const responseText = response.content.find((b) => b.type === 'text')?.text ?? '';
  const code = extractCode(responseText);
  const elapsed1 = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[AGENT] Code generated in ${elapsed1}s (${code.length} chars)`);

  // --- Execute code ---
  const api = new TripletexApi(credentials);
  console.log(`[AGENT] Executing code...`);
  const execResult = await executeCode(code, api);

  const elapsed2 = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[AGENT] Execution done in ${elapsed2}s — success=${execResult.success} calls=${api.callLog.length} errors=${api.getErrorCount()}`);

  if (execResult.logs.length > 0) {
    console.log(`[AGENT LOGS] ${execResult.logs.join('\n')}`);
  }

  if (execResult.success) {
    messages.push({ role: 'assistant', content: response.content });
    return buildResult(api, code, execResult.logs, false, messages, systemPrompt);
  }

  // --- Turn 2: Retry on failure ---
  console.log(`[AGENT] Execution failed: ${execResult.error?.slice(0, 200)}`);

  // Check timeout before retrying
  if (Date.now() - startTime > AGENT_TIMEOUT_MS * 0.6) {
    console.warn(`[AGENT] Not enough time for retry, returning partial result`);
    messages.push({ role: 'assistant', content: response.content });
    return buildResult(api, code, execResult.logs, false, messages, systemPrompt);
  }

  messages.push({ role: 'assistant', content: response.content });
  messages.push({
    role: 'user',
    content: `Code execution failed with error:\n${execResult.error}\n\nAPI calls made before failure:\n${api.getCallLog()}\n\nConsole output:\n${execResult.logs.join('\n')}\n\nFix the code. Output ONLY the corrected code block.`,
  });

  console.log(`[AGENT] Retrying with error context...`);
  const retryResponse = await createMessage({
    model: 'claude-opus-4-6',
    max_tokens: 32768,
    system: systemPrompt,
    messages,
  });

  const retryText = retryResponse.content.find((b) => b.type === 'text')?.text ?? '';
  const retryCode = extractCode(retryText);
  const elapsed3 = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[AGENT] Retry code generated in ${elapsed3}s (${retryCode.length} chars)`);

  // Fresh API instance for retry
  const retryApi = new TripletexApi(credentials);
  console.log(`[AGENT] Executing retry code...`);
  const retryResult = await executeCode(retryCode, retryApi);

  const elapsed4 = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[AGENT] Retry done in ${elapsed4}s — success=${retryResult.success} calls=${retryApi.callLog.length}`);

  if (retryResult.logs.length > 0) {
    console.log(`[AGENT RETRY LOGS] ${retryResult.logs.join('\n')}`);
  }

  messages.push({ role: 'assistant', content: retryResponse.content });
  return buildResult(retryApi, retryCode, retryResult.logs, true, messages, systemPrompt);
}

// ---------------------------------------------------------------------------
// Build result from API call log
// ---------------------------------------------------------------------------

function buildResult(
  api: TripletexApi,
  code: string,
  logs: string[],
  retried: boolean,
  messages: any[],
  systemPrompt: any[],
): AgentResult {
  const errors = api.callLog
    .filter((c) => c.status >= 400)
    .map((c) => ({ tool: `${c.method} ${c.path}`, status: c.status }));

  return {
    toolCallCount: api.callLog.length,
    errors,
    messages,
    systemPrompt,
    generatedCode: code,
    executionLogs: logs,
    retried,
  };
}

// ---------------------------------------------------------------------------
// Sandbox verification — asks Claude to summarise what happened
// ---------------------------------------------------------------------------

export async function verifySandboxResult(
  prompt: string,
  result: AgentResult
): Promise<{ verified: boolean; summary: string }> {
  try {
    const response = await claude.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 512,
      system: 'You are verifying whether an accounting task was completed successfully in Tripletex. Be concise.',
      messages: [
        {
          role: 'user',
          content: `Task: "${prompt}"\n\nGenerated code:\n\`\`\`typescript\n${result.generatedCode}\n\`\`\`\n\nExecution logs:\n${result.executionLogs?.join('\n') ?? 'none'}\n\nAPI calls: ${result.toolCallCount}, Errors: ${result.errors.length}\nRetried: ${result.retried}\n\nWas the task completed successfully? Reply with:\nVERIFIED: yes/no\nSUMMARY: one sentence.`,
        },
      ],
    });

    const text = response.content.find((b) => b.type === 'text')?.text ?? '';
    const verified = /VERIFIED:\s*yes/i.test(text);
    const summaryMatch = text.match(/SUMMARY:\s*(.+)/i);
    const summary = summaryMatch?.[1]?.trim() ?? text.trim();

    console.log(`[VERIFY] verified=${verified} summary=${summary}`);
    return { verified, summary };
  } catch (err) {
    console.error(`[VERIFY ERROR]`, err);
    return { verified: false, summary: `Verification failed: ${err}` };
  }
}
