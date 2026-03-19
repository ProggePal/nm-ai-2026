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

const SYSTEM_PROMPT_BLOCKS: Anthropic.TextBlockParam[] = [
  {
    type: 'text',
    text: `You are an expert accounting AI agent completing tasks in Tripletex, a Norwegian accounting system.

You have tools to call the Tripletex v2 REST API. Authentication is handled automatically — just call the tools.

## API conventions
- List responses: { from, count, values: [...] }
- Single responses: { value: {...} }
- Always use fields param: ?fields=id,name to avoid large responses
- Linked resources use {id: N} — e.g. { customer: {id: 123}, department: {id: 5} }
- Dates: YYYY-MM-DD
- IDs: integers
- Address objects: { addressLine1, postalCode, city, country: {id} } — country is optional for Norway

## Field schemas — use these exactly

**Customer** POST /customer:
{ name, organizationNumber, email, invoiceEmail, phoneNumber, isPrivateIndividual,
  postalAddress: { addressLine1, addressLine2, postalCode, city },
  physicalAddress: { addressLine1, addressLine2, postalCode, city },
  invoiceSendMethod: "EMAIL"|"EHF"|"EFAKTURA"|"AVTALEGIRO"|"VIPPS"|"PAPER"|"MANUAL",
  currency: {id}, department: {id}, customerNumber }
- Use postalAddress for mailing address. physicalAddress for physical location.

**Employee** POST /employee:
{ firstName, lastName, email, employeeNumber, phoneNumberMobile, dateOfBirth,
  address: { addressLine1, postalCode, city },
  department: {id} }

**Product** POST /product:
{ name, number, description, costExcludingVatCurrency, priceExcludingVatCurrency,
  vatType: {id}, productUnit: {id}, isInactive }
- Always GET /ledger/vatType?fields=id,name first to find correct VAT type id

**Order** POST /order:
{ customer: {id}, orderDate, deliveryDate, department: {id}, project: {id},
  ourContactEmployee: {id}, invoiceComment, currency: {id} }

**OrderLine** POST /order/orderline:
{ order: {id}, product: {id}, description, count, unitPriceExcludingVatCurrency,
  discount, vatType: {id} }
- "count" is quantity (not "quantity")

**Invoice actions:**
- PUT /order/{id}/:invoice → converts order to invoice, returns { value: { id } }
- PUT /invoice/{id}/:send?sendType=EMAIL&overrideEmailAddress=x@y.com → send
- PUT /invoice/{id}/:payment → body: { date, amount, paymentTypeId, paidAmount }
  - To register full payment: { paymentTypeId: 1, paidAmount: <full amount>, date: "YYYY-MM-DD" }
  - paymentTypeId 1 = bank transfer (most common)

**Travel expense** POST /travelExpense:
{ employee: {id}, description, from, to, project: {id}, department: {id} }
- from/to are dates (YYYY-MM-DD)

**TravelExpense cost** POST /travelExpense/cost:
{ travelExpense: {id}, category: {id}, amountCurrencyIncVat, paymentType: {id} }

**MileageAllowance** POST /travelExpense/mileageAllowance:
{ travelExpense: {id}, date, km, departureLocation, destination, isCompanyCar }

**Project** POST /project:
{ name, number, customer: {id}, projectManager: {id}, startDate, endDate,
  description, isInternal, department: {id} }

**Department** POST /department:
{ name, departmentNumber, departmentManager: {id} }

**Voucher** POST /ledger/voucher:
{ date, description, voucherType: {id},
  postings: [{ account: {id}, amount, amountCurrency, date, description, vatType: {id} }] }

## Common task flows

**Create customer with address:**
POST /customer → { name, organizationNumber, email,
  postalAddress: { addressLine1: "Street 1", postalCode: "1234", city: "Oslo" } }

**Find customer by org number:**
GET /customer?organizationNumber=123456789&fields=id,name

**Find invoice by customer:**
GET /invoice?customerId={id}&fields=id,invoiceNumber,amountExcludingVatCurrency,amountOutstanding,invoiceDate

**Register full payment on invoice:**
1. GET /customer?organizationNumber=X&fields=id,name → get customer id
2. GET /invoice?customerId={id}&fields=id,amountCurrency,amountOutstanding → find invoice
3. PUT /invoice/{id}/:payment with body { paymentTypeId: 1, paidAmount: <amountCurrency>, date: "today" }

**Create invoice and send:**
1. POST /order → { customer: {id}, orderDate, deliveryDate }
2. POST /order/orderline → { order: {id}, product: {id}, count, unitPriceExcludingVatCurrency, description }
3. PUT /order/{id}/:invoice → get invoice id from response value.id
4. PUT /invoice/{id}/:send?sendType=EMAIL

**Travel expense full flow:**
1. POST /travelExpense → { employee: {id}, from, to, description }
2. POST /travelExpense/cost or /travelExpense/mileageAllowance
3. PUT /travelExpense/:deliver?id={id}

**Reverse voucher:**
PUT /ledger/voucher/{id}/:reverse

## Search params (all GET list endpoints)
- fields=id,name (always use to limit payload)
- from=0&count=100 (pagination)
- organizationNumber, name, email (filter params)

## Batch endpoints
POST /employee/list, /customer/list, /order/orderline/list, /travelExpense/cost/list, /project/list
Always use batch endpoints when creating more than one of the same resource.

## Efficiency rules — CRITICAL for scoring
1. Plan all steps before making any API calls
2. Always use fields param — never omit it
3. Trust 201 responses — do NOT verify with a GET after successful create
4. Read error messages carefully and fix correctly on first retry
5. Minimize total API calls — combine lookups when possible

The task prompt may be in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French.
Complete the task fully then stop.`,
    cache_control: { type: 'ephemeral' },
  },
];

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
      model: 'claude-opus-4-6',
      max_tokens: 8192,
      system: SYSTEM_PROMPT_BLOCKS,
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
