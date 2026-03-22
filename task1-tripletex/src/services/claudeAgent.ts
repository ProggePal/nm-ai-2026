import Anthropic from '@anthropic-ai/sdk';
import { buildSystemPrompt } from './systemPrompt.js';
import { TripletexApi } from './tripletexApi.js';
import { createInvoiceFlow, registerPayment, createSupplierInvoiceVoucher, setupProject } from './compoundTools.js';
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

// Compound tools — multi-step accounting flows in a single call
const COMPOUND_TOOLS: Anthropic.Tool[] = [
  {
    name: 'create_invoice',
    description: 'Complete invoice flow: finds customer, products, VAT, sets bank account, creates order+lines, converts to invoice, optionally sends. Handles everything.',
    input_schema: {
      type: 'object' as const,
      properties: {
        customerOrgNumber: { type: 'string', description: 'Customer organization number' },
        lines: {
          type: 'array',
          description: 'Invoice lines',
          items: {
            type: 'object',
            properties: {
              productNumber: { type: 'string', description: 'Product number (optional if description+unitPrice given)' },
              description: { type: 'string', description: 'Line description' },
              quantity: { type: 'number', description: 'Quantity' },
              unitPrice: { type: 'number', description: 'Unit price excl VAT in NOK' },
              vatRate: { type: 'number', description: '0, 15, or 25 (default 25)' },
            },
          },
        },
        invoiceDate: { type: 'string', description: 'Invoice date YYYY-MM-DD' },
        send: { type: 'boolean', description: 'Send invoice by email (default false)' },
      },
      required: ['customerOrgNumber', 'lines', 'invoiceDate'],
    },
    allowed_callers: ['code_execution_20260120'],
  },
  {
    name: 'register_payment',
    description: 'Register payment on unpaid invoice: finds customer, finds unpaid invoice, looks up bank payment type, registers payment. Handles foreign currency with paidAmountCurrency.',
    input_schema: {
      type: 'object' as const,
      properties: {
        customerOrgNumber: { type: 'string', description: 'Customer organization number' },
        amount: { type: 'number', description: 'Payment amount in NOK (default: full invoice amount)' },
        paymentDate: { type: 'string', description: 'Payment date YYYY-MM-DD' },
        paidAmountCurrency: { type: 'number', description: 'Amount in foreign currency (for EUR invoices etc)' },
      },
      required: ['customerOrgNumber', 'paymentDate'],
    },
    allowed_callers: ['code_execution_20260120'],
  },
  {
    name: 'create_supplier_invoice',
    description: 'Register supplier invoice as voucher: finds/creates supplier, looks up expense account+AP+VAT, creates voucher with correct postings. Handles all the posting structure.',
    input_schema: {
      type: 'object' as const,
      properties: {
        supplierName: { type: 'string', description: 'Supplier name' },
        supplierOrgNumber: { type: 'string', description: 'Supplier organization number' },
        invoiceNumber: { type: 'string', description: 'Invoice reference number' },
        grossAmount: { type: 'number', description: 'Total amount INCLUDING VAT' },
        expenseAccountNumber: { type: 'number', description: 'Expense account number (e.g. 6500, 6300)' },
        vatRate: { type: 'number', description: 'VAT rate percentage (default 25)' },
        date: { type: 'string', description: 'Invoice date YYYY-MM-DD' },
      },
      required: ['supplierName', 'invoiceNumber', 'grossAmount', 'expenseAccountNumber', 'date'],
    },
    allowed_callers: ['code_execution_20260120'],
  },
  {
    name: 'setup_project',
    description: 'Create project with entitlements: finds employee+customer, grants entitlements, creates project, optionally creates linked activity. Handles the entitlement-before-project requirement.',
    input_schema: {
      type: 'object' as const,
      properties: {
        projectName: { type: 'string', description: 'Project name' },
        customerOrgNumber: { type: 'string', description: 'Customer org number (optional for internal projects)' },
        projectManagerEmail: { type: 'string', description: 'Project manager email' },
        startDate: { type: 'string', description: 'Start date YYYY-MM-DD' },
        budget: { type: 'number', description: 'Fixed price budget (optional)' },
        isInternal: { type: 'boolean', description: 'Is internal project' },
        createActivity: { type: 'boolean', description: 'Create linked activity (default true)' },
        activityName: { type: 'string', description: 'Activity name (default: project name)' },
      },
      required: ['projectName', 'projectManagerEmail', 'startDate'],
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
        ...COMPOUND_TOOLS,
      ],
    };

    if (containerId) {
      params.container = containerId;
    }

    // Use non-streaming — the SDK handles timeout internally
    const response = await claude.messages.create(params, {
      timeout: 300000, // 5 min request timeout
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
        return await api.post(input.path, input.body, input.params);
      case 'tripletex_put':
        return await api.put(input.path, input.body, input.params);
      case 'tripletex_del':
        return await api.del(input.path);
      case 'tripletex_post_list':
        return await api.postList(input.path, input.items);
      // Compound tools
      case 'create_invoice':
        return await createInvoiceFlow(api, input);
      case 'register_payment':
        return await registerPayment(api, input);
      case 'create_supplier_invoice':
        return await createSupplierInvoiceVoucher(api, input);
      case 'setup_project':
        return await setupProject(api, input);
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
