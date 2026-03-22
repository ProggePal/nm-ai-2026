# Tripletex Accounting Agent

Autonomous LLM agent that solves accounting tasks in [Tripletex](https://www.tripletex.no/) via natural language. Built for NM i AI 2026, Task 1.

A `POST /solve` request arrives with a task prompt and Tripletex credentials. Claude Opus 4.6 reasons over the prompt and executes Tripletex API calls as tools in a loop until the task is complete.

## API

### `POST /solve`

Requires `Authorization: Bearer <token>`.

```json
{
  "prompt": "Opprett kunden Strandvik AS med organisasjonsnummer 808795132.",
  "tripletex_credentials": {
    "base_url": "https://your-instance.tripletex.no/v2",
    "session_token": "your-session-token"
  }
}
```

Response:
```json
{ "status": "completed" }
```

## Setup

**Requirements:** [Bun](https://bun.sh/)

```bash
bun install
bun run start
```

**Environment variables:**

```
ANTHROPIC_API_KEY=
JWT_SECRET=
TRIPLETEX_BASE_URL=        # sandbox mode only
TRIPLETEX_SESSION_TOKEN=   # sandbox mode only
```

## Stack

TypeScript · Bun · Hono · Claude Opus 4.6

## License

[MIT](./LICENSE)
