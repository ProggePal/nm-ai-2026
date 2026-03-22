import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { AuthenticationError } from '../errors.js';

declare global {
  namespace Express {
    interface Request {
      tokenData?: jwt.JwtPayload;
    }
  }
}

const JWT_SECRET = process.env.JWT_SECRET;
const API_KEY = process.env.API_KEY;

export function authMiddleware(req: Request, _res: Response, next: NextFunction): void {
  // If no auth is configured, allow all requests
  if (!JWT_SECRET && !API_KEY) {
    return next();
  }

  const authHeader = req.headers.authorization;

  if (!authHeader) {
    return next(new AuthenticationError('No token provided'));
  }

  if (!authHeader.startsWith('Bearer ')) {
    return next(new AuthenticationError('Invalid Authorization header'));
  }

  const token = authHeader.slice(7);

  // If API_KEY is set, accept it as a simple Bearer token
  if (API_KEY && token === API_KEY) {
    return next();
  }

  // Otherwise try JWT verification
  if (JWT_SECRET) {
    try {
      const decoded = jwt.verify(token, JWT_SECRET);
      req.tokenData = decoded as jwt.JwtPayload;
      return next();
    } catch (err) {
      return next(new AuthenticationError((err as Error).message));
    }
  }

  next(new AuthenticationError('Invalid token'));
}
