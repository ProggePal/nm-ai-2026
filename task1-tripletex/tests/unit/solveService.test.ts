import { describe, it, expect, mock, beforeEach } from 'bun:test';

const TEST_CREDS = {
  base_url: 'https://test.tripletex.dev/v2',
  session_token: 'test-token',
};

// Mock the claudeAgent module before importing solveService
const mockRunAgent = mock(async (prompt: string, credentials: any, images: any[], pdfs: any[]) => ({
  toolCallCount: 2,
  errors: [],
  messages: [
    { role: 'user', content: [{ type: 'text', text: prompt }] },
    { role: 'assistant', content: [{ type: 'text', text: 'Done.' }] },
  ],
}));

const mockVerifySandboxResult = mock(async (prompt: string, result: any) => ({
  verified: true,
  summary: 'Completed successfully.',
}));

mock.module('../../src/services/claudeAgent.js', () => ({
  runAgent: mockRunAgent,
  verifySandboxResult: mockVerifySandboxResult,
}));

const { solveService } = await import('../../src/services/solveService.js');

describe('solveService.solve', () => {
  beforeEach(() => {
    mockRunAgent.mockClear();
    mockVerifySandboxResult.mockClear();
  });

  it('returns { status: "completed" } for non-sandbox request', async () => {
    const result = await solveService.solve({
      prompt: 'Create a customer',
      tripletex_credentials: TEST_CREDS,
    });

    expect(result).toEqual({ status: 'completed' });
    expect(mockRunAgent).toHaveBeenCalledTimes(1);
    expect(mockVerifySandboxResult).not.toHaveBeenCalled();
  });

  it('passes credentials to runAgent', async () => {
    await solveService.solve({
      prompt: 'Opprett en ansatt',
      tripletex_credentials: TEST_CREDS,
    });

    const call = mockRunAgent.mock.calls[0];
    expect(call[0]).toBe('Opprett en ansatt');
    expect(call[1]).toEqual(TEST_CREDS);
    expect(call[2]).toEqual([]); // images
    expect(call[3]).toEqual([]); // pdfs
  });

  it('separates image and PDF files for agent attachments', async () => {
    await solveService.solve({
      prompt: 'Register invoice',
      tripletex_credentials: TEST_CREDS,
      files: [
        { filename: 'photo.png', content_base64: 'base64png', mime_type: 'image/png' },
        { filename: 'doc.pdf', content_base64: 'base64pdf', mime_type: 'application/pdf' },
        { filename: 'scan.jpeg', content_base64: 'base64jpg', mime_type: 'image/jpeg' },
      ],
    });

    const call = mockRunAgent.mock.calls[0];
    const images = call[2];
    const pdfs = call[3];
    expect(images).toHaveLength(2);
    expect(images[0]).toEqual({ mimeType: 'image/png', data: 'base64png' });
    expect(images[1]).toEqual({ mimeType: 'image/jpeg', data: 'base64jpg' });
    expect(pdfs).toHaveLength(1);
    expect(pdfs[0]).toEqual({ filename: 'doc.pdf', data: 'base64pdf' });
  });

  it('returns verification result when use_sandbox=true', async () => {
    const result = await solveService.solve({
      prompt: 'Opprett kunden Test AS',
      tripletex_credentials: TEST_CREDS,
      use_sandbox: true,
    });

    expect(result.status).toBe('completed');
    expect(result.verified).toBe(true);
    expect(result.summary).toBe('Completed successfully.');
    expect(result.tool_calls).toBe(2);
    expect(result.errors).toBeUndefined();
    expect(mockVerifySandboxResult).toHaveBeenCalledTimes(1);
  });

  it('includes errors array when agent has errors in sandbox mode', async () => {
    mockRunAgent.mockImplementationOnce(async () => ({
      toolCallCount: 4,
      errors: [
        { tool: 'tripletex_customer_create', status: 422 },
        { tool: 'tripletex_invoice_create', status: 400 },
      ],
      messages: [],
    }));

    const result = await solveService.solve({
      prompt: 'Create something',
      tripletex_credentials: TEST_CREDS,
      use_sandbox: true,
    });

    expect(result.errors).toHaveLength(2);
    expect(result.errors![0].tool).toBe('tripletex_customer_create');
    expect(result.tool_calls).toBe(4);
  });

  it('does not call verifySandboxResult when use_sandbox is falsy', async () => {
    await solveService.solve({ prompt: 'test', tripletex_credentials: TEST_CREDS, use_sandbox: false });
    expect(mockVerifySandboxResult).not.toHaveBeenCalled();

    await solveService.solve({ prompt: 'test', tripletex_credentials: TEST_CREDS });
    expect(mockVerifySandboxResult).not.toHaveBeenCalled();
  });

  it('handles empty files array', async () => {
    await solveService.solve({ prompt: 'test', tripletex_credentials: TEST_CREDS, files: [] });
    const images = mockRunAgent.mock.calls[0][2];
    const pdfs = mockRunAgent.mock.calls[0][3];
    expect(images).toEqual([]);
    expect(pdfs).toEqual([]);
  });

  it('returns unverified result when verification fails', async () => {
    mockVerifySandboxResult.mockImplementationOnce(async () => ({
      verified: false,
      summary: 'Customer was not created.',
    }));

    const result = await solveService.solve({
      prompt: 'Opprett kunden',
      tripletex_credentials: TEST_CREDS,
      use_sandbox: true,
    });

    expect(result.verified).toBe(false);
    expect(result.summary).toBe('Customer was not created.');
  });
});
