# AI Accounting Agent

An LLM-powered service that automates accounting tasks in [Tripletex](https://www.tripletex.no/) via natural language prompts. Built for the NM AI competition (Task 2).

The agent receives a task description in any language, reasons over it, and executes the necessary Tripletex API calls autonomously until the task is complete.

## How it works

1. A POST request arrives at `/solve` with a prompt and Tripletex credentials
2. Claude (Opus 4.6) receives the prompt along with a system prompt describing the Tripletex API
3. Claude decides which API calls to make and calls them as tools
4. The results are fed back to Claude, which continues until the task is done
5. Optionally, a second Claude call (Haiku) verifies whether the task was completed correctly

```
User prompt
  → Claude reasons → calls tripletex_get / tripletex_post / ...
  → Agent executes the call → returns result to Claude
  → Claude reasons again → calls next tool
  → ... (repeats until done)
  → Returns { status: "completed" }
```

## API

### `POST /solve`

Requires a Bearer JWT token in the `Authorization` header.

**Request body:**
```json
{
  "prompt": "Opprett kunden Strandvik AS med organisasjonsnummer 808795132.",
  "tripletex_credentials": {
    "base_url": "https://your-tripletex-instance/v2",
    "session_token": "your-session-token"
  },
  "files": [
    {
      "filename": "invoice.png",
      "content": "<base64-encoded>",
      "mime_type": "image/png"
    }
  ],
  "use_sandbox": false
}
```

| Field | Required | Description |
|---|---|---|
| `prompt` | Yes | The accounting task in any language |
| `tripletex_credentials` | Yes | Tripletex API base URL and session token |
| `files` | No | Image attachments (base64-encoded) |
| `use_sandbox` | No | Enables verification mode (see below) |

**Response (normal mode):**
```json
{ "status": "completed" }
```

**Response (sandbox mode):**
```json
{
  "status": "completed",
  "sandbox": true,
  "verified": true,
  "summary": "Created customer Strandvik AS with org number 808795132.",
  "tool_calls": 3,
  "errors": []
}
```

## Supported tasks

The agent can handle any Tripletex task expressed in natural language, including:

- Customer, employee, product, project, and department management
- Order creation → invoice conversion → sending
- Invoice payment registration
- Travel expenses and mileage allowances
- Supplier invoice approval and payment
- Timesheet entries and month approval
- Salary transactions
- Voucher creation and reversal
- Bank reconciliation
- Fixed asset management and depreciation
- Balance sheets and financial reporting
- Salary/payroll reconciliation

Prompts can be written in Norwegian, Nynorsk, English, or any other language.

## System prompt

The system prompt is assembled dynamically per request (`src/services/systemPrompt.ts`):

- **Base block** — always loaded. Contains API conventions, field schemas, and common task flows for all standard accounting operations.
- **Dynamic modules** — loaded only when keywords in the prompt match. Four modules exist: Bank Reconciliation, Asset Management, Financial Reporting, and Salary Reconciliation.

All blocks use Anthropic prompt caching (`cache_control: ephemeral`) to reduce latency and cost on repeated calls.

## Tools

The agent has five tools it can call against the Tripletex API:

| Tool | HTTP method | Purpose |
|---|---|---|
| `tripletex_get` | GET | List or retrieve resources |
| `tripletex_post` | POST | Create a single resource |
| `tripletex_post_list` | POST `/*/list` | Create multiple resources in one call |
| `tripletex_put` | PUT | Update a resource or trigger an action (`:send`, `:invoice`, `:approve`) |
| `tripletex_delete` | DELETE | Remove a resource |

Multiple tool calls within the same turn execute in parallel. The loop aborts immediately on 401/403 auth errors.

## Sandbox mode

When `use_sandbox: true`, the service:
1. Uses sandbox Tripletex credentials from environment variables instead of the request
2. After the agent finishes, sends the full conversation history to Claude Haiku for verification
3. Returns `verified: true/false`, a one-sentence `summary`, `tool_calls` count, and any HTTP errors

This is useful for testing prompt changes and evaluating task success rate without manual inspection.

## Development

**Prerequisites:** [Bun](https://bun.sh/)

```bash
bun install
```

**Run locally:**
```bash
bun run dev         # with hot reload
bun run start       # production
```

**Environment variables:**
```
ANTHROPIC_API_KEY=          # Required
JWT_SECRET=                 # Required — used to verify Bearer tokens
TRIPLETEX_BASE_URL=         # Required for sandbox mode
TRIPLETEX_SESSION_TOKEN=    # Required for sandbox mode
GCS_BUCKET_NAME=            # Optional — logs requests to Google Cloud Storage
```

**Test a prompt against the sandbox:**
```bash
bun run solve -- --sandbox "Opprett kunden Bergendahl Consulting AS med org.nr 912345678"
bun run solve:local -- --sandbox "Your prompt here"   # targets localhost:3000
```

## Project structure

```
src/
  app.ts                        # Express app setup
  index.ts                      # Server entry point
  types.ts                      # Shared TypeScript types
  errors.ts                     # Custom error classes
  routes/
    solveRouter.ts              # POST /solve route + validation
  services/
    claudeAgent.ts              # Agent loop + tool execution + sandbox verification
    solveService.ts             # Orchestrates agent, handles sandbox mode
    systemPrompt.ts             # Dynamic system prompt assembly
    tripletexClient.ts          # HTTP client for Tripletex API
    requestLogger.ts            # Logs requests to stdout and GCS
  middleware/
    authMiddleware.ts           # JWT Bearer token verification
    errorHandler.ts             # Global error handler
scripts/
  solve.ts                      # CLI for manual testing
test-requests/                  # Archived real request payloads
```

## Tech stack

- **Runtime:** Bun
- **Framework:** Express
- **AI:** Anthropic Claude (Opus 4.6 for agent, Haiku for verification)
- **Validation:** Joi / celebrate
- **Auth:** JWT
- **Logging:** Google Cloud Storage
