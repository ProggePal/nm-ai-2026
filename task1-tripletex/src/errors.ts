export class GenericError extends Error {
  httpStatusCode: number;
  code: string;
  metadata?: Record<string, unknown>;

  constructor(
    message: string,
    httpStatusCode: number,
    code: string,
    metadata?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'GenericError';
    this.httpStatusCode = httpStatusCode;
    this.code = code;
    this.metadata = metadata;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class NotFoundError extends GenericError {
  constructor(message = 'Not found') {
    super(message, 404, 'NOT_FOUND');
    this.name = 'NotFoundError';
  }
}

export class AuthenticationError extends GenericError {
  constructor(message: string) {
    super(message, 401, '11401');
    this.name = 'AuthenticationError';
  }
}
