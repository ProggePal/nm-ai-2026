import type { Request, Response, NextFunction } from 'express';
import { isCelebrateError } from 'celebrate';
import { GenericError } from '../errors.js';

export function errorHandler(
  err: unknown,
  _req: Request,
  res: Response,
  _next: NextFunction
): void {
  if (isCelebrateError(err)) {
    const details: Record<string, string> = {};
    for (const [segment, joiError] of err.details) {
      details[segment] = joiError.message;
    }
    res.status(400).json({
      error: {
        name: 'ValidationError',
        message: 'Validation failed',
        code: 'VALIDATION_ERROR',
        statusCode: 400,
        details,
      },
    });
    return;
  }

  if (err instanceof GenericError) {
    res.status(err.httpStatusCode).json({
      error: {
        name: err.name,
        message: err.message,
        code: err.code,
        statusCode: err.httpStatusCode,
      },
    });
    return;
  }

  console.error(err);
  res.status(500).json({
    error: {
      name: 'InternalServerError',
      message: 'An unexpected error occurred',
      code: 'INTERNAL_SERVER_ERROR',
      statusCode: 500,
    },
  });
}
