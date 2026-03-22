import { describe, it, expect, beforeAll, afterAll, mock } from 'bun:test';
import jwt from 'jsonwebtoken';

// Must match the secret used in authMiddleware.test.ts since modules are cached across test files
const TEST_SECRET = 'shared-test-secret';
const TEST_PORT = 9876;

const TEST_CREDS = {
  base_url: 'https://test.tripletex.dev/v2',
  session_token: 'test-token',
};

// Set env before any imports
process.env.JWT_SECRET = TEST_SECRET;
process.env.PORT = String(TEST_PORT);

// Mock the GCS request logger to avoid cloud dependency
mock.module('../../src/services/requestLogger.js', () => ({
  logRequest: async () => 'mock-filename.json',
}));

// Mock the Claude agent to avoid API calls
mock.module('../../src/services/claudeAgent.js', () => ({
  runAgent: async (prompt: string) => ({
    toolCallCount: 3,
    errors: [],
    messages: [
      { role: 'user', content: [{ type: 'text', text: prompt }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Task completed.' }] },
    ],
  }),
  verifySandboxResult: async (prompt: string) => ({
    verified: true,
    summary: 'Task was completed successfully.',
  }),
}));

const { app } = await import('../../src/app.js');

let server: ReturnType<typeof app.listen>;
let baseUrl: string;

function validToken(payload: Record<string, unknown> = { sub: 'test' }) {
  return jwt.sign(payload, TEST_SECRET);
}

beforeAll(() => {
  server = app.listen(TEST_PORT);
  baseUrl = `http://localhost:${TEST_PORT}`;
});

afterAll(() => {
  server?.close();
});

describe('GET /', () => {
  it('returns health check with valid auth', async () => {
    const res = await fetch(`${baseUrl}/`, {
      headers: { Authorization: `Bearer ${validToken()}` },
    });
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.message).toBe('Hello AI-world');
  });

  it('returns 401 without a token', async () => {
    const res = await fetch(`${baseUrl}/`);
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error.code).toBe('11401');
  });
});

describe('404 handling', () => {
  it('returns 404 for unknown routes', async () => {
    const res = await fetch(`${baseUrl}/nonexistent`, {
      headers: { Authorization: `Bearer ${validToken()}` },
    });
    expect(res.status).toBe(404);
    const body = await res.json();
    expect(body.error.code).toBe('NOT_FOUND');
  });
});

describe('POST /solve', () => {
  it('returns 401 without auth token', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: 'Create a customer', tripletex_credentials: TEST_CREDS }),
    });
    expect(res.status).toBe(401);
  });

  it('returns { status: "completed" } for a valid request (non-sandbox)', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${validToken()}`,
      },
      body: JSON.stringify({
        prompt: 'Opprett kunden Acme AS med org.nr 123456789',
        files: [],
        tripletex_credentials: TEST_CREDS,
      }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe('completed');
    expect(body.verified).toBeUndefined();
  });

  it('returns sandbox verification result when use_sandbox=true', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${validToken()}`,
      },
      body: JSON.stringify({
        prompt: 'Opprett kunden Acme AS med org.nr 123456789',
        files: [],
        tripletex_credentials: TEST_CREDS,
        use_sandbox: true,
      }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe('completed');
    expect(body.verified).toBe(true);
    expect(body.summary).toBe('Task was completed successfully.');
    expect(body.tool_calls).toBe(3);
  });

  it('handles request with image file attachments', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${validToken()}`,
      },
      body: JSON.stringify({
        prompt: 'Register this invoice',
        files: [
          {
            filename: 'invoice.png',
            content_base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==',
            mime_type: 'image/png',
          },
        ],
        tripletex_credentials: TEST_CREDS,
      }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe('completed');
  });

  it('handles request with PDF file attachments', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${validToken()}`,
      },
      body: JSON.stringify({
        prompt: 'Process this PDF invoice',
        files: [
          {
            filename: 'invoice.pdf',
            content_base64: 'JVBERi0xLjQ=',
            mime_type: 'application/pdf',
          },
        ],
        tripletex_credentials: TEST_CREDS,
      }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe('completed');
  });

  // Validation tests
  it('returns 400 when prompt is missing', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${validToken()}`,
      },
      body: JSON.stringify({
        tripletex_credentials: TEST_CREDS,
      }),
    });
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error.code).toBe('VALIDATION_ERROR');
  });

  it('returns 400 when tripletex_credentials is missing', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${validToken()}`,
      },
      body: JSON.stringify({
        prompt: 'Create a customer',
      }),
    });
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error.code).toBe('VALIDATION_ERROR');
  });

  it('returns 400 when session_token is missing from credentials', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${validToken()}`,
      },
      body: JSON.stringify({
        prompt: 'Create a customer',
        tripletex_credentials: { base_url: 'https://test.tripletex.dev/v2' },
      }),
    });
    expect(res.status).toBe(400);
  });

  it('accepts GET on /solve as 404 (only POST is defined)', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      headers: { Authorization: `Bearer ${validToken()}` },
    });
    expect(res.status).toBe(404);
  });
});

describe('Auth edge cases', () => {
  it('rejects expired tokens', async () => {
    const expired = jwt.sign({ sub: 'test' }, TEST_SECRET, { expiresIn: '-10s' });
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${expired}`,
      },
      body: JSON.stringify({ prompt: 'test', tripletex_credentials: TEST_CREDS }),
    });
    expect(res.status).toBe(401);
  });

  it('rejects tokens signed with wrong secret', async () => {
    const wrong = jwt.sign({ sub: 'test' }, 'wrong-secret');
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${wrong}`,
      },
      body: JSON.stringify({ prompt: 'test', tripletex_credentials: TEST_CREDS }),
    });
    expect(res.status).toBe(401);
  });

  it('rejects malformed Authorization header', async () => {
    const res = await fetch(`${baseUrl}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Token abc123',
      },
      body: JSON.stringify({ prompt: 'test', tripletex_credentials: TEST_CREDS }),
    });
    expect(res.status).toBe(401);
  });
});
