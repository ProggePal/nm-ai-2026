import { VertexAI, SchemaType, type Tool, type Part } from '@google-cloud/vertexai';
import { tripletexRequest } from './tripletexClient.js';
import type { TripletexCredentials } from '../types.js';

const PROJECT = process.env.GCP_PROJECT_ID ?? 'ai-nm26osl-1886';
const REGION = process.env.GCP_REGION ?? 'europe-west4';
const MODEL = process.env.GEMINI_MODEL ?? 'gemini-2.5-pro-preview-05-06';

const vertexAI = new VertexAI({ project: PROJECT, location: REGION });

// ---------------------------------------------------------------------------
// Tool definitions for Tripletex API
// ---------------------------------------------------------------------------

const TRIPLETEX_TOOLS: Tool[] = [
  {
    functionDeclarations: [
      {
        name: 'tripletex_get',
        description:
          'Perform a GET request against the Tripletex v2 REST API. ' +
          'Use this to list or retrieve resources. ' +
          'List responses are wrapped in {from, count, values:[...]}. ' +
          'Use the fields param to select only needed fields. ' +
          'Paginate with count and from params.',
        parameters: {
          type: SchemaType.OBJECT,
          properties: {
            path: {
              type: SchemaType.STRING,
              description: 'API path, e.g. /employee, /customer, /invoice/1234',
            },
            params: {
              type: SchemaType.OBJECT,
              description: 'Query parameters as key-value pairs, e.g. {"fields": "id,firstName", "count": "100"}',
            },
          },
          required: ['path'],
        },
      },
      {
        name: 'tripletex_post',
        description:
          'Perform a POST request against the Tripletex v2 REST API. ' +
          'Use this to create new resources.',
        parameters: {
          type: SchemaType.OBJECT,
          properties: {
            path: {
              type: SchemaType.STRING,
              description: 'API path, e.g. /employee',
            },
            body: {
              type: SchemaType.OBJECT,
              description: 'JSON request body',
            },
          },
          required: ['path', 'body'],
        },
      },
      {
        name: 'tripletex_put',
        description:
          'Perform a PUT request against the Tripletex v2 REST API. ' +
          'Use this to update existing resources or trigger action endpoints (e.g. /:send, /:invoice).',
        parameters: {
          type: SchemaType.OBJECT,
          properties: {
            path: {
              type: SchemaType.STRING,
              description: 'API path with ID, e.g. /employee/42 or /invoice/5/:send',
            },
            body: {
              type: SchemaType.OBJECT,
              description: 'JSON request body (can be empty object {} for action endpoints)',
            },
            params: {
              type: SchemaType.OBJECT,
              description: 'Query parameters, e.g. {"sendType": "EMAIL"} for /:send',
            },
          },
          required: ['path'],
        },
      },
      {
        name: 'tripletex_delete',
        description: 'Perform a DELETE request against the Tripletex v2 REST API.',
        parameters: {
          type: SchemaType.OBJECT,
          properties: {
            path: {
              type: SchemaType.STRING,
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
          'More efficient than creating one at a time.',
        parameters: {
          type: SchemaType.OBJECT,
          properties: {
            path: {
              type: SchemaType.STRING,
              description: 'API path ending in /list, e.g. /order/orderline/list',
            },
            body: {
              type: SchemaType.ARRAY,
              description: 'Array of objects to create',
              items: { type: SchemaType.OBJECT },
            },
          },
          required: ['path', 'body'],
        },
      },
    ],
  },
];

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are an expert accounting AI agent that completes tasks in Tripletex, a Norwegian accounting system.

You have access to the Tripletex v2 REST API through tools. All API calls go through an authenticated proxy — you do not need to handle authentication yourself.

## Key API paths
- /employee, /employee/{id}
- /customer, /customer/{id}
- /product, /product/{id}
- /order, /order/{id}, /order/{id}/:invoice
- /order/orderline, /order/orderline/{id}, /order/orderline/list
- /invoice, /invoice/{id}, /invoice/{id}/:send, /invoice/{id}/:payment, /invoice/{id}/:createCreditNote
- /travelExpense, /travelExpense/{id}, /travelExpense/:deliver, /travelExpense/:approve
- /travelExpense/cost, /travelExpense/mileageAllowance, /travelExpense/accommodationAllowance
- /project, /project/{id}
- /department, /department/{id}
- /ledger/voucher, /ledger/voucher/{id}/:reverse
- /ledger/account, /ledger/vatType
- /product/unit

## API conventions
- List responses: { from, count, values: [...] }
- Single responses: { value: {...} }
- Use fields param to limit response: ?fields=id,firstName,lastName
- Linked resources always use {id: N} — e.g. { customer: {id: 123} }
- Dates: YYYY-MM-DD
- IDs: integers

## Common task flows

**Create employee:**
POST /employee with { firstName, lastName, email, employeeNumber, department: {id} }

**Create customer:**
POST /customer with { name, email, organizationNumber, customerAccountNumber }

**Create product:**
GET /ledger/vatType to find VAT type id, GET /product/unit to find unit id
POST /product with { name, number, costExcludingVatCurrency, vatType: {id}, unit: {id} }

**Invoice flow (most common multi-step task):**
1. POST /order → { customer: {id}, deliveryDate, orderDate }
2. POST /order/orderline → { order: {id}, product: {id}, quantity, unitPriceExcludingVatCurrency, description }
3. PUT /order/{id}/:invoice → converts to invoice (response contains invoice id)
4. PUT /invoice/{id}/:send?sendType=EMAIL → send to customer

**Travel expense:**
1. POST /travelExpense → { employee: {id}, from, to, description }
2. POST /travelExpense/cost → { travelExpense: {id}, category: {id}, amountCurrencyIncVat, paymentType: {id} }
   OR POST /travelExpense/mileageAllowance → { travelExpense: {id}, departureDate, km, departureLocation, destination }
3. PUT /travelExpense/:deliver (pass ids as query param)

**Create project:**
POST /project with { name, number, customer: {id}, projectManager: {id}, startDate, endDate }

**Reverse/correct voucher:**
PUT /ledger/voucher/{id}/:reverse

## Efficiency rules (CRITICAL — scored on number of API calls)
1. Plan before calling — think through what you need
2. Use fields param always — never fetch full objects when you only need id
3. Batch with /list endpoints when creating multiple resources
4. Do NOT make extra verification GETs after creating — trust the 201 response
5. Read error messages carefully — fix on first retry, not trial-and-error

The task prompt may be in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French.
Complete the task fully then stop. Do not make unnecessary calls.`;

// ---------------------------------------------------------------------------
// Tool executor
// ---------------------------------------------------------------------------

async function executeToolCall(
  name: string,
  args: Record<string, unknown>,
  credentials: TripletexCredentials
): Promise<unknown> {
  const path = args.path as string;

  switch (name) {
    case 'tripletex_get':
      return tripletexRequest('GET', credentials, path, args.params as Record<string, unknown> | undefined);

    case 'tripletex_post':
      return tripletexRequest('POST', credentials, path, undefined, args.body);

    case 'tripletex_put':
      return tripletexRequest(
        'PUT',
        credentials,
        path,
        args.params as Record<string, unknown> | undefined,
        args.body ?? {}
      );

    case 'tripletex_delete':
      return tripletexRequest('DELETE', credentials, path);

    case 'tripletex_post_list':
      return tripletexRequest('POST', credentials, path, undefined, args.body);

    default:
      return { error: `Unknown tool: ${name}` };
  }
}

// ---------------------------------------------------------------------------
// Main agent function
// ---------------------------------------------------------------------------

export async function runAgent(
  prompt: string,
  credentials: TripletexCredentials,
  imageAttachments: Array<{ mimeType: string; data: string }> = []
): Promise<void> {
  const model = vertexAI.getGenerativeModel({
    model: MODEL,
    systemInstruction: SYSTEM_PROMPT,
  });

  const chat = model.startChat({ tools: TRIPLETEX_TOOLS });

  // Build initial message
  const initialParts: Part[] = [];

  for (const img of imageAttachments) {
    initialParts.push({
      inlineData: { mimeType: img.mimeType, data: img.data },
    });
  }
  initialParts.push({ text: prompt });

  let result = await chat.sendMessage(initialParts);

  // Agentic loop
  while (true) {
    const parts = result.response.candidates?.[0]?.content?.parts ?? [];
    const functionCalls = parts.filter((p) => p.functionCall);

    if (functionCalls.length === 0) break;

    // Execute all tool calls in parallel
    const functionResponses: Part[] = await Promise.all(
      functionCalls.map(async (part) => {
        const { name, args } = part.functionCall!;
        const output = await executeToolCall(
          name,
          (args ?? {}) as Record<string, unknown>,
          credentials
        );
        return {
          functionResponse: {
            name,
            response: { output: JSON.stringify(output) },
          },
        };
      })
    );

    result = await chat.sendMessage(functionResponses);
  }
}
