import Anthropic from '@anthropic-ai/sdk';
import { buildSystemPrompt } from './systemPrompt.js';
import { TripletexApi } from './tripletexApi.js';
import type { TripletexCredentials } from '../types.js';

const claude = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const AGENT_TIMEOUT_MS = 4.5 * 60 * 1000; // 4.5 min

// ---------------------------------------------------------------------------
// Tool definitions — 5 thin Tripletex tools + code_execution
// ---------------------------------------------------------------------------

const TOOLS: Anthropic.Tool[] = [
  {
    name: 'tripletex_get',
    description: 'GET request to Tripletex API. Returns parsed JSON. Use for searching/fetching entities.',
    input_schema: {
      type: 'object' as const,
      properties: {
        path: { type: 'string', description: 'API path e.g. /customer, /invoice, /ledger/account' },
        params: { type: 'object', description: 'Query parameters e.g. {fields: "id,name", count: 100, number: 1920}' },
      },
      required: ['path'],
    },
    allowed_callers: ['code_execution_20260120'],
  },
  {
    name: 'tripletex_post',
    description: 'POST request to Tripletex API. Returns parsed JSON. Use for creating entities.',
    input_schema: {
      type: 'object' as const,
      properties: {
        path: { type: 'string', description: 'API path e.g. /customer, /employee, /ledger/voucher' },
        body: { type: 'object', description: 'Request body with entity fields' },
      },
      required: ['path'],
    },
    allowed_callers: ['code_execution_20260120'],
  },
  {
    name: 'tripletex_put',
    description: 'PUT request to Tripletex API. Returns parsed JSON. Use for updating entities and action endpoints (/:action).',
    input_schema: {
      type: 'object' as const,
      properties: {
        path: { type: 'string', description: 'API path e.g. /customer/{id}, /invoice/{id}/:send' },
        body: { type: 'object', description: 'Request body (use {} for action endpoints)' },
        params: { type: 'object', description: 'Query parameters (for action endpoints like /:payment)' },
      },
      required: ['path'],
    },
    allowed_callers: ['code_execution_20260120'],
  },
  {
    name: 'tripletex_del',
    description: 'DELETE request to Tripletex API. Use for deleting entities.',
    input_schema: {
      type: 'object' as const,
      properties: {
        path: { type: 'string', description: 'API path e.g. /travelExpense/{id}' },
      },
      required: ['path'],
    },
    allowed_callers: ['code_execution_20260120'],
  },
  {
    name: 'tripletex_post_list',
    description: 'POST array body to Tripletex /list endpoints. Use for batch creating multiple entities.',
    input_schema: {
      type: 'object' as const,
      properties: {
        path: { type: 'string', description: 'API list path e.g. /order/orderline/list' },
        items: { type: 'array', description: 'Array of entities to create', items: { type: 'object' } },
      },
      required: ['path', 'items'],
    },
    allowed_callers: ['code_execution_20260120'],
  },
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AgentResult {
  toolCallCount: number;
  errors: Array<{ tool: string; status: number }>;
  messages: any[];
  systemPrompt: any[];
}

// ---------------------------------------------------------------------------
// Main agent — programmatic tool calling
// ---------------------------------------------------------------------------

export async function runAgent(
  prompt: string,
  credentials: TripletexCredentials,
  imageAttachments: Array<{ mimeType: string; data: string }> = [],
  pdfAttachments: Array<{ filename: string; data: string }> = [],
): Promise<AgentResult> {
  const startTime = Date.now();
  const systemPrompt = buildSystemPrompt(prompt);
  const api = new TripletexApi(credentials);

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
  let containerId: string | undefined;

  // Tool-use loop
  while (true) {
    if (Date.now() - startTime > AGENT_TIMEOUT_MS) {
      console.warn(`[TIMEOUT] Agent loop exceeded ${AGENT_TIMEOUT_MS / 1000}s limit`);
      break;
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[AGENT] Calling Claude (${elapsed}s elapsed, ${api.callLog.length} API calls so far)...`);

    const params: any = {
      model: 'claude-opus-4-6',
      max_tokens: 16384,
      system: systemPrompt,
      messages,
      tools: [
        { type: 'code_execution_20260120', name: 'code_execution' },
        ...TOOLS,
      ],
    };

    if (containerId) {
      params.container = containerId;
    }

    // Use non-streaming with explicit timeout to get container field
    const response = await (claude.messages.create as Function)({
      ...params,
      timeout: 300000, // 5 min timeout
    }) as any;

    // Track container for reuse
    if (response.container?.id) {
      containerId = response.container.id;
      console.log(`[AGENT] Container: ${containerId}`);
    }

    // Log activity
    for (const block of response.content) {
      if (block.type === 'tool_use') {
        const callerType = block.caller?.type ?? 'direct';
        console.log(`[TOOL] ${block.name} (${callerType}) ${JSON.stringify(block.input).slice(0, 200)}`);
      } else if (block.type === 'server_tool_use') {
        console.log(`[CODE_EXEC] Writing Python code...`);
      } else if (block.type === 'code_execution_tool_result') {
        const result = block.content;
        if (result?.stdout) console.log(`[CODE_STDOUT] ${result.stdout.slice(0, 500)}`);
        if (result?.stderr) console.log(`[CODE_STDERR] ${result.stderr.slice(0, 500)}`);
      } else if (block.type === 'text' && block.text?.trim()) {
        console.log(`[TEXT] ${block.text.slice(0, 200)}`);
      }
    }

    const elapsed2 = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[AGENT] stop_reason=${response.stop_reason} elapsed=${elapsed2}s`);

    messages.push({ role: 'assistant', content: response.content });

    // Check if we're done
    if (response.stop_reason === 'end_turn') break;

    // Fulfill tool calls
    if (response.stop_reason === 'tool_use') {
      const toolResults: any[] = [];

      for (const block of response.content) {
        if (block.type === 'tool_use') {
          const result = await fulfillToolCall(api, block.name, block.input);
          toolResults.push({
            type: 'tool_result',
            tool_use_id: block.id,
            content: typeof result === 'string' ? result : JSON.stringify(result),
          });
        }
      }

      if (toolResults.length > 0) {
        messages.push({ role: 'user', content: toolResults });
      }
      continue;
    }

    // Safety: break if no tool activity
    const hasActivity = response.content.some(
      (b: any) => b.type === 'tool_use' || b.type === 'server_tool_use'
    );
    if (!hasActivity) break;
  }

  const errors = api.callLog
    .filter((c) => c.status >= 400)
    .map((c) => ({ tool: `${c.method} ${c.path}`, status: c.status }));

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[AGENT] Done: ${api.callLog.length} API calls, ${errors.length} errors, ${elapsed}s`);

  return { toolCallCount: api.callLog.length, errors, messages, systemPrompt };
}

// ---------------------------------------------------------------------------
// Fulfill a tool call by calling Tripletex API
// ---------------------------------------------------------------------------

async function fulfillToolCall(
  api: TripletexApi,
  toolName: string,
  input: any,
): Promise<any> {
  try {
    switch (toolName) {
      case 'tripletex_get':
        return await api.get(input.path, input.params);
      case 'tripletex_post':
        return await api.post(input.path, input.body);
      case 'tripletex_put':
        return await api.put(input.path, input.body, input.params);
      case 'tripletex_del':
        return await api.del(input.path);
      case 'tripletex_post_list':
        return await api.postList(input.path, input.items);
      default:
        return { error: `Unknown tool: ${toolName}` };
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { error: message };
  }
}

// ---------------------------------------------------------------------------
// Sandbox verification
// ---------------------------------------------------------------------------

export async function verifySandboxResult(
  prompt: string,
  result: AgentResult,
): Promise<{ verified: boolean; summary: string }> {
  try {
    const response = await claude.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 512,
      system: 'You are verifying whether an accounting task was completed successfully in Tripletex. Be concise.',
      messages: [
        {
          role: 'user',
          content: `Task: "${prompt}"\n\nAPI calls made: ${result.toolCallCount}, Errors: ${result.errors.length}\n\nWas the task completed successfully? Reply with:\nVERIFIED: yes/no\nSUMMARY: one sentence.`,
        },
      ],
    });

    const text = response.content.find((b) => b.type === 'text')?.text ?? '';
    const verified = /VERIFIED:\s*yes/i.test(text);
    const summaryMatch = text.match(/SUMMARY:\s*(.+)/i);
    const summary = summaryMatch?.[1]?.trim() ?? text.trim();

    return { verified, summary };
  } catch (err) {
    return { verified: false, summary: `Verification failed: ${err}` };
  }
}
