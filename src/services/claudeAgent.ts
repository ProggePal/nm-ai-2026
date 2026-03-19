import Anthropic from '@anthropic-ai/sdk';
import { tripletexRequest } from './tripletexClient.js';
import type { TripletexCredentials } from '../types.js';

const claude = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

const TOOLS: Anthropic.Tool[] = [
  {
    name: 'tripletex_get',
    description:
      'Perform a GET request against the Tripletex v2 REST API. ' +
      'Use this to list or retrieve resources. ' +
      'List responses: {from, count, values:[...]}. ' +
      'Use fields param to limit response size. ' +
      'Paginate with count and from params.',
    input_schema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'API path, e.g. /employee, /customer, /invoice/1234',
        },
        params: {
          type: 'object',
          description: 'Query parameters, e.g. {"fields": "id,firstName", "count": "100"}',
          additionalProperties: true,
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'tripletex_post',
    description: 'Perform a POST request to create a new resource in Tripletex.',
    input_schema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'API path, e.g. /employee, /order',
        },
        body: {
          type: 'object',
          description: 'JSON request body',
          additionalProperties: true,
        },
      },
      required: ['path', 'body'],
    },
  },
  {
    name: 'tripletex_put',
    description:
      'Perform a PUT request to update a resource or trigger an action endpoint ' +
      '(e.g. /invoice/{id}/:send, /order/{id}/:invoice, /ledger/voucher/{id}/:reverse).',
    input_schema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'API path, e.g. /employee/42, /invoice/5/:send',
        },
        body: {
          type: 'object',
          description: 'JSON request body (use empty object {} for action endpoints)',
          additionalProperties: true,
        },
        params: {
          type: 'object',
          description: 'Query parameters, e.g. {"sendType": "EMAIL"} for /:send',
          additionalProperties: true,
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'tripletex_delete',
    description: 'Perform a DELETE request to remove a resource from Tripletex.',
    input_schema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'API path with ID, e.g. /employee/42',
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'tripletex_post_list',
    description:
      'Perform a POST /*/list request to create multiple resources in one call. ' +
      'Always prefer this over multiple individual POSTs when creating more than one resource.',
    input_schema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'API path ending in /list, e.g. /order/orderline/list',
        },
        body: {
          type: 'array',
          description: 'Array of objects to create',
          items: { type: 'object', additionalProperties: true },
        },
      },
      required: ['path', 'body'],
    },
  },
];

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are an expert accounting AI agent completing tasks in Tripletex, a Norwegian accounting system.

You have tools to call the Tripletex v2 REST API. Authentication is handled automatically — just call the tools.

## Key API paths
- /employee, /employee/{id}
- /customer, /customer/{id}
- /product, /product/{id}
- /order, /order/{id}, /order/{id}/:invoice
- /order/orderline, /order/orderline/list
- /invoice, /invoice/{id}, /invoice/{id}/:send, /invoice/{id}/:payment, /invoice/{id}/:createCreditNote
- /travelExpense, /travelExpense/{id}, /travelExpense/:deliver, /travelExpense/:approve, /travelExpense/:createVouchers
- /travelExpense/cost, /travelExpense/cost/list
- /travelExpense/mileageAllowance, /travelExpense/accommodationAllowance
- /project, /project/{id}
- /department, /department/{id}
- /ledger/voucher, /ledger/voucher/{id}/:reverse, /ledger/voucher/{id}/:sendToLedger
- /ledger/account, /ledger/vatType
- /product/unit

## API conventions
- List responses: { from, count, values: [...] }
- Single responses: { value: {...} }
- Always use fields param: ?fields=id,firstName to avoid large responses
- Linked resources use {id: N} — e.g. { customer: {id: 123}, department: {id: 5} }
- Dates: YYYY-MM-DD
- IDs: integers

## Common task flows

**Create employee:**
POST /employee → { firstName, lastName, email, employeeNumber, department: {id} }

**Create customer:**
POST /customer → { name, email, organizationNumber, customerAccountNumber }

**Create product:**
First GET /ledger/vatType?fields=id,name to find VAT type
POST /product → { name, number, costExcludingVatCurrency, vatType: {id} }

**Invoice flow:**
1. POST /order → { customer: {id}, deliveryDate, orderDate }
2. POST /order/orderline → { order: {id}, product: {id}, quantity, unitPriceExcludingVatCurrency, description }
3. PUT /order/{id}/:invoice → creates invoice (response contains invoice id in value.id)
4. PUT /invoice/{id}/:send?sendType=EMAIL → sends to customer

**Travel expense:**
1. POST /travelExpense → { employee: {id}, from, to, description }
2. POST /travelExpense/cost → { travelExpense: {id}, category: {id}, amountCurrencyIncVat, paymentType: {id} }
   OR POST /travelExpense/mileageAllowance → { travelExpense: {id}, departureDate, km, departureLocation, destination }
3. PUT /travelExpense/:deliver?id={id}

**Create project:**
POST /project → { name, number, customer: {id}, projectManager: {id}, startDate, endDate }

**Reverse voucher:**
PUT /ledger/voucher/{id}/:reverse

## Efficiency rules — CRITICAL for scoring
1. Think through the full plan before making any API calls
2. Always use fields param — never fetch full objects when you only need id
3. Batch creates using /list endpoints
4. Do NOT make verification GETs after successful creates — trust the 201 response
5. Avoid trial-and-error — read error messages and fix correctly on retry
6. Minimize total number of API calls

The task prompt may be in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French.
Complete the task fully then stop.`;

// ---------------------------------------------------------------------------
// Tool executor
// ---------------------------------------------------------------------------

async function executeTool(
  name: string,
  input: Record<string, unknown>,
  credentials: TripletexCredentials
): Promise<string> {
  const path = input.path as string;

  let result: unknown;

  switch (name) {
    case 'tripletex_get':
      result = await tripletexRequest('GET', credentials, path, input.params as Record<string, unknown> | undefined);
      break;
    case 'tripletex_post':
      result = await tripletexRequest('POST', credentials, path, undefined, input.body);
      break;
    case 'tripletex_put':
      result = await tripletexRequest('PUT', credentials, path, input.params as Record<string, unknown> | undefined, input.body ?? {});
      break;
    case 'tripletex_delete':
      result = await tripletexRequest('DELETE', credentials, path);
      break;
    case 'tripletex_post_list':
      result = await tripletexRequest('POST', credentials, path, undefined, input.body);
      break;
    default:
      result = { error: `Unknown tool: ${name}` };
  }

  return JSON.stringify(result);
}

// ---------------------------------------------------------------------------
// Main agent
// ---------------------------------------------------------------------------

export async function runAgent(
  prompt: string,
  credentials: TripletexCredentials,
  imageAttachments: Array<{ mimeType: string; data: string }> = []
): Promise<void> {
  const content: Anthropic.MessageParam['content'] = [];

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
  content.push({ type: 'text', text: prompt });

  const messages: Anthropic.MessageParam[] = [{ role: 'user', content }];

  // Agentic loop
  while (true) {
    const response = await claude.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 8192,
      system: SYSTEM_PROMPT,
      tools: TOOLS,
      messages,
    });

    console.log(`[CLAUDE] stop_reason=${response.stop_reason} tool_calls=${response.content.filter(b => b.type === 'tool_use').length}`);

    const toolUseBlocks = response.content.filter(
      (b): b is Anthropic.ToolUseBlock => b.type === 'tool_use'
    );

    messages.push({ role: 'assistant', content: response.content });

    if (response.stop_reason === 'end_turn' || toolUseBlocks.length === 0) break;

    // Execute all tool calls in parallel
    const toolResults: Anthropic.ToolResultBlockParam[] = await Promise.all(
      toolUseBlocks.map(async (block) => {
        console.log(`[TOOL] ${block.name} ${JSON.stringify(block.input)}`);
        const result = await executeTool(block.name, block.input as Record<string, unknown>, credentials);
        console.log(`[TOOL RESULT] ${block.name} → ${result.slice(0, 200)}`);
        return {
          type: 'tool_result' as const,
          tool_use_id: block.id,
          content: result,
        };
      })
    );

    messages.push({ role: 'user', content: toolResults });
  }
}
