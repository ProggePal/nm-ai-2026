import Anthropic from '@anthropic-ai/sdk';
import { buildSystemPrompt } from './systemPrompt.js';
import type { TripletexCredentials } from '../types.js';

const claude = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const MCP_SERVER_URL = 'https://trippeltex-mcp-production-0d9b.up.railway.app/mcp';

const AGENT_TIMEOUT_MS = 4.5 * 60 * 1000; // 4.5 min — leave margin before 5 min competition timeout

// ---------------------------------------------------------------------------
// Main agent
// ---------------------------------------------------------------------------

export interface AgentResult {
  toolCallCount: number;
  errors: Array<{ tool: string; status: number }>;
  messages: any[];
  systemPrompt: any[];
}

export async function runAgent(
  prompt: string,
  credentials: TripletexCredentials,
  imageAttachments: Array<{ mimeType: string; data: string }> = [],
  pdfAttachments: Array<{ filename: string; data: string }> = [],
): Promise<AgentResult> {
  const content: any[] = [];

  // Add images
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

  // Add PDFs as documents
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
  const systemPrompt = buildSystemPrompt(prompt);
  let toolCallCount = 0;
  const errors: AgentResult['errors'] = [];
  const startTime = Date.now();

  // Agentic loop — MCP tools execute server-side, but we loop in case
  // Claude needs multiple turns (e.g. max_tokens reached mid-response)
  while (true) {
    // Timeout guard
    if (Date.now() - startTime > AGENT_TIMEOUT_MS) {
      console.warn(`[TIMEOUT] Agent loop exceeded ${AGENT_TIMEOUT_MS / 1000}s limit`);
      break;
    }

    const response = await (claude.beta.messages.create as Function)({
      model: 'claude-sonnet-4-6',
      max_tokens: 4096,
      betas: ['mcp-client-2025-11-20'],
      system: systemPrompt,
      mcp_servers: [
        {
          type: 'url',
          url: `${MCP_SERVER_URL}?base_url=${encodeURIComponent(credentials.base_url)}`,
          name: 'tripletex',
          authorization_token: credentials.session_token,
        },
      ],
      tools: [
        { type: 'tool_search_tool_bm25_20251119', name: 'tool_search_tool_bm25' },
        {
          type: 'mcp_toolset',
          mcp_server_name: 'tripletex',
          default_config: { defer_loading: true },
          configs: {
            // Preload all commonly needed tools to eliminate tool_search calls
            // -- Customers --
            tripletex_customer_search: { defer_loading: false },
            tripletex_customer_create: { defer_loading: false },
            tripletex_customer_delete: { defer_loading: false },
            tripletex_customer_update: { defer_loading: false },
            // -- Employees --
            tripletex_employee_search: { defer_loading: false },
            tripletex_employee_create: { defer_loading: false },
            tripletex_employee_update: { defer_loading: false },
            tripletex_employee_entitlement_grant_entitlements_by_template: { defer_loading: false },
            // -- Invoicing --
            tripletex_invoice_search: { defer_loading: false },
            tripletex_invoice_create: { defer_loading: false },
            tripletex_invoice_payment: { defer_loading: false },
            tripletex_invoice_send: { defer_loading: false },
            tripletex_invoice_create_credit_note: { defer_loading: false },
            tripletex_invoice_payment_type_search: { defer_loading: false },
            // -- Orders --
            tripletex_order_create: { defer_loading: false },
            tripletex_order_orderline_create: { defer_loading: false },
            tripletex_order_orderline_create_many: { defer_loading: false },
            tripletex_order_invoice: { defer_loading: false },
            tripletex_order_invoice_multiple_orders: { defer_loading: false },
            // -- Products --
            tripletex_product_search: { defer_loading: false },
            tripletex_product_create: { defer_loading: false },
            // -- Departments --
            tripletex_department_search: { defer_loading: false },
            tripletex_department_create: { defer_loading: false },
            // -- Projects --
            tripletex_project_search: { defer_loading: false },
            tripletex_project_create: { defer_loading: false },
            // -- Timesheet --
            tripletex_activity_for_time_sheet: { defer_loading: false },
            tripletex_activity_search: { defer_loading: false },
            tripletex_timesheet_entry_create: { defer_loading: false },
            // -- Travel expenses --
            tripletex_travel_expense_create: { defer_loading: false },
            tripletex_travel_expense_search: { defer_loading: false },
            tripletex_travel_expense_delete: { defer_loading: false },
            tripletex_travel_expense_cost_create: { defer_loading: false },
            tripletex_travel_expense_mileage_allowance_create: { defer_loading: false },
            tripletex_travel_expense_per_diem_compensation_create: { defer_loading: false },
            tripletex_travel_expense_payment_type_search: { defer_loading: false },
            tripletex_travel_expense_rate_category_search: { defer_loading: false },
            tripletex_travel_expense_rate_category_group_search: { defer_loading: false },
            tripletex_travel_expense_cost_category_search: { defer_loading: false },
            tripletex_travel_expense_deliver: { defer_loading: false },
            // -- Ledger / Accounting --
            tripletex_ledger_voucher_create: { defer_loading: false },
            tripletex_ledger_voucher_reverse: { defer_loading: false },
            tripletex_ledger_voucher_delete: { defer_loading: false },
            tripletex_ledger_voucher_search: { defer_loading: false },
            tripletex_ledger_vat_type_search: { defer_loading: false },
            tripletex_ledger_account_search: { defer_loading: false },
            tripletex_ledger_account_update: { defer_loading: false },
            tripletex_ledger_posting_search: { defer_loading: false },
            // -- Accounting dimensions --
            tripletex_accounting_dimension_name_search: { defer_loading: false },
            tripletex_accounting_dimension_name_create: { defer_loading: false },
            tripletex_accounting_dimension_value_search: { defer_loading: false },
            tripletex_accounting_dimension_value_create: { defer_loading: false },
            // -- Supplier / Incoming invoices --
            tripletex_supplier_search: { defer_loading: false },
            tripletex_supplier_create: { defer_loading: false },
            tripletex_supplier_invoice_search: { defer_loading: false },
            tripletex_supplier_invoice_approve: { defer_loading: false },
            // NOTE: tripletex_incoming_invoice_create intentionally NOT preloaded — BETA endpoint, use voucher approach
            // -- Salary --
            tripletex_salary_type_search: { defer_loading: false },
            tripletex_salary_transaction_create: { defer_loading: false },
            // -- Company --
            tripletex_company_get: { defer_loading: false },
            tripletex_company_update: { defer_loading: false },
          },
        },
      ],
      messages,
    }) as any;

    // Count and log tool activity
    for (const block of response.content as any[]) {
      if (block.type === 'mcp_tool_use') {
        toolCallCount++;
        console.log(`[MCP TOOL] ${block.name} ${JSON.stringify(block.input)}`);
      } else if (block.type === 'mcp_tool_result') {
        const preview = JSON.stringify(block.content).slice(0, 200);
        console.log(`[MCP RESULT] ${block.is_error ? 'ERROR ' : ''}${preview}`);
        if (block.is_error) {
          errors.push({ tool: block.tool_use_id ?? 'mcp_tool', status: 0 });
        }
      } else if (block.type === 'server_tool_use') {
        console.log(`[TOOL SEARCH] query=${JSON.stringify(block.input)}`);
      }
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[CLAUDE] stop_reason=${response.stop_reason} mcp_tool_calls=${toolCallCount} elapsed=${elapsed}s`);

    messages.push({ role: 'assistant', content: response.content });

    if (response.stop_reason === 'end_turn') break;

    // Safety: break if no tool activity in this turn
    const hasToolActivity = (response.content as any[]).some(
      (b) => b.type === 'mcp_tool_use' || b.type === 'tool_use' || b.type === 'server_tool_use'
    );
    if (!hasToolActivity) break;
  }

  return { toolCallCount, errors, messages, systemPrompt };
}

// ---------------------------------------------------------------------------
// Sandbox verification — asks Claude to summarise what happened
// ---------------------------------------------------------------------------

export async function verifySandboxResult(
  prompt: string,
  result: AgentResult
): Promise<{ verified: boolean; summary: string }> {
  // Strip MCP tool blocks from messages — the verification call doesn't have MCP configured
  const cleanMessages = result.messages.map((msg: any) => {
    if (!Array.isArray(msg.content)) return msg;
    const cleaned = msg.content
      .filter((b: any) => ['text', 'mcp_tool_use', 'mcp_tool_result'].includes(b.type))
      .map((b: any) => {
        if (b.type === 'mcp_tool_use') {
          return { type: 'text', text: `[TOOL CALL] ${b.name}(${JSON.stringify(b.input)})` };
        }
        if (b.type === 'mcp_tool_result') {
          const preview = JSON.stringify(b.content).slice(0, 500);
          return { type: 'text', text: `[TOOL RESULT] ${b.is_error ? 'ERROR ' : ''}${preview}` };
        }
        return b;
      });
    return { ...msg, content: cleaned.length > 0 ? cleaned : [{ type: 'text', text: '(no text)' }] };
  });

  try {
    const response = await claude.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 512,
      system: 'You are verifying whether an accounting task was completed successfully in Tripletex. Be concise.',
      messages: [
        ...cleanMessages,
        {
          role: 'user',
          content: `The original task was: "${prompt}"\n\nBased on the API calls and responses above, was the task completed successfully? Reply with:\nVERIFIED: yes/no\nSUMMARY: one sentence describing what was done (or what failed).`,
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
