import { describe, it, expect, beforeAll } from 'bun:test';
import jwt from 'jsonwebtoken';

const TEST_SECRET = 'shared-test-secret';

// Set JWT_SECRET before importing the middleware (it reads at module level)
process.env.JWT_SECRET = TEST_SECRET;

// Dynamic import after env is set
const { authMiddleware } = await import('../../src/middleware/authMiddleware.js');

function makeReq(headers: Record<string, string> = {}) {
  return { headers } as any;
}

function makeRes() {
  return {} as any;
}

describe('authMiddleware', () => {
  it('calls next with AuthenticationError when no Authorization header', () => {
    const errors: Error[] = [];
    const next = (err?: any) => { if (err) errors.push(err); };

    authMiddleware(makeReq(), makeRes(), next);

    expect(errors).toHaveLength(1);
    expect(errors[0].message).toBe('No token provided');
    expect((errors[0] as any).httpStatusCode).toBe(401);
  });

  it('calls next with AuthenticationError when Authorization is not Bearer', () => {
    const errors: Error[] = [];
    const next = (err?: any) => { if (err) errors.push(err); };

    authMiddleware(makeReq({ authorization: 'Basic abc123' }), makeRes(), next);

    expect(errors).toHaveLength(1);
    expect(errors[0].message).toBe('Invalid Authorization header');
  });

  it('calls next with AuthenticationError for an invalid JWT token', () => {
    const errors: Error[] = [];
    const next = (err?: any) => { if (err) errors.push(err); };

    authMiddleware(makeReq({ authorization: 'Bearer invalid.token.here' }), makeRes(), next);

    expect(errors).toHaveLength(1);
    expect((errors[0] as any).httpStatusCode).toBe(401);
  });

  it('calls next with AuthenticationError for expired JWT token', () => {
    const expired = jwt.sign({ sub: 'test' }, TEST_SECRET, { expiresIn: '-1s' });
    const errors: Error[] = [];
    const next = (err?: any) => { if (err) errors.push(err); };

    authMiddleware(makeReq({ authorization: `Bearer ${expired}` }), makeRes(), next);

    expect(errors).toHaveLength(1);
    expect(errors[0].message).toContain('expired');
  });

  it('calls next with AuthenticationError for wrong secret', () => {
    const wrongToken = jwt.sign({ sub: 'test' }, 'wrong-secret');
    const errors: Error[] = [];
    const next = (err?: any) => { if (err) errors.push(err); };

    authMiddleware(makeReq({ authorization: `Bearer ${wrongToken}` }), makeRes(), next);

    expect(errors).toHaveLength(1);
    expect((errors[0] as any).httpStatusCode).toBe(401);
  });

  it('sets req.tokenData and calls next() for valid JWT', () => {
    const token = jwt.sign({ sub: 'competition-bot' }, TEST_SECRET);
    const req = makeReq({ authorization: `Bearer ${token}` });
    let nextCalled = false;
    let nextErr: any = undefined;
    const next = (err?: any) => { nextCalled = true; nextErr = err; };

    authMiddleware(req, makeRes(), next);

    expect(nextCalled).toBe(true);
    expect(nextErr).toBeUndefined();
    expect(req.tokenData).toBeDefined();
    expect(req.tokenData.sub).toBe('competition-bot');
  });

  it('handles token with custom claims', () => {
    const token = jwt.sign({ sub: 'team-alpha', role: 'admin' }, TEST_SECRET);
    const req = makeReq({ authorization: `Bearer ${token}` });
    const next = (err?: any) => {};

    authMiddleware(req, makeRes(), next);

    expect(req.tokenData.sub).toBe('team-alpha');
    expect(req.tokenData.role).toBe('admin');
  });
});
