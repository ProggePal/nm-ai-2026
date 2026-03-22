import { describe, it, expect } from 'bun:test';
import { errorHandler } from '../../src/middleware/errorHandler.js';
import { GenericError, NotFoundError, AuthenticationError } from '../../src/errors.js';

function makeRes() {
  let statusCode = 0;
  let jsonBody: any = null;
  const res: any = {
    status(code: number) { statusCode = code; return res; },
    json(body: any) { jsonBody = body; return res; },
    get _status() { return statusCode; },
    get _body() { return jsonBody; },
  };
  return res;
}

const req = {} as any;
const next = () => {};

describe('errorHandler', () => {
  it('handles GenericError with correct status and body', () => {
    const err = new GenericError('Something went wrong', 422, 'VALIDATION');
    const res = makeRes();

    errorHandler(err, req, res, next);

    expect(res._status).toBe(422);
    expect(res._body).toEqual({
      error: {
        name: 'GenericError',
        message: 'Something went wrong',
        code: 'VALIDATION',
        statusCode: 422,
      },
    });
  });

  it('handles NotFoundError', () => {
    const err = new NotFoundError();
    const res = makeRes();

    errorHandler(err, req, res, next);

    expect(res._status).toBe(404);
    expect(res._body.error.code).toBe('NOT_FOUND');
    expect(res._body.error.name).toBe('NotFoundError');
  });

  it('handles AuthenticationError', () => {
    const err = new AuthenticationError('Token expired');
    const res = makeRes();

    errorHandler(err, req, res, next);

    expect(res._status).toBe(401);
    expect(res._body.error.message).toBe('Token expired');
    expect(res._body.error.code).toBe('11401');
  });

  it('handles unknown errors with 500 status', () => {
    const err = new Error('unexpected crash');
    const res = makeRes();

    // Suppress console.error for cleaner test output
    const origConsoleError = console.error;
    console.error = () => {};
    errorHandler(err, req, res, next);
    console.error = origConsoleError;

    expect(res._status).toBe(500);
    expect(res._body).toEqual({
      error: {
        name: 'InternalServerError',
        message: 'An unexpected error occurred',
        code: 'INTERNAL_SERVER_ERROR',
        statusCode: 500,
      },
    });
  });

  it('handles non-Error objects thrown', () => {
    const res = makeRes();

    const origConsoleError = console.error;
    console.error = () => {};
    errorHandler('string error' as any, req, res, next);
    console.error = origConsoleError;

    expect(res._status).toBe(500);
    expect(res._body.error.code).toBe('INTERNAL_SERVER_ERROR');
  });
});
