import { describe, it, expect } from 'bun:test';
import { GenericError, NotFoundError, AuthenticationError } from '../../src/errors.js';

describe('GenericError', () => {
  it('stores httpStatusCode, code, and message', () => {
    const err = new GenericError('Something failed', 422, 'VALIDATION');
    expect(err.message).toBe('Something failed');
    expect(err.httpStatusCode).toBe(422);
    expect(err.code).toBe('VALIDATION');
    expect(err.name).toBe('GenericError');
  });

  it('stores optional metadata', () => {
    const meta = { field: 'email' };
    const err = new GenericError('Bad input', 400, 'BAD_REQUEST', meta);
    expect(err.metadata).toEqual(meta);
  });

  it('is an instance of Error', () => {
    const err = new GenericError('test', 500, 'ERR');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(GenericError);
  });
});

describe('NotFoundError', () => {
  it('defaults to 404 and NOT_FOUND code', () => {
    const err = new NotFoundError();
    expect(err.httpStatusCode).toBe(404);
    expect(err.code).toBe('NOT_FOUND');
    expect(err.message).toBe('Not found');
    expect(err.name).toBe('NotFoundError');
  });

  it('accepts custom message', () => {
    const err = new NotFoundError('Customer not found');
    expect(err.message).toBe('Customer not found');
    expect(err.httpStatusCode).toBe(404);
  });

  it('is an instance of GenericError', () => {
    const err = new NotFoundError();
    expect(err).toBeInstanceOf(GenericError);
    expect(err).toBeInstanceOf(Error);
  });
});

describe('AuthenticationError', () => {
  it('uses 401 status and 11401 code', () => {
    const err = new AuthenticationError('Token expired');
    expect(err.httpStatusCode).toBe(401);
    expect(err.code).toBe('11401');
    expect(err.message).toBe('Token expired');
    expect(err.name).toBe('AuthenticationError');
  });

  it('is an instance of GenericError', () => {
    const err = new AuthenticationError('bad token');
    expect(err).toBeInstanceOf(GenericError);
  });
});
