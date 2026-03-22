import { tripletexRequest } from './tripletexClient.js';
import type { TripletexCredentials } from '../types.js';

export class ApiError extends Error {
  constructor(
    public statusCode: number,
    public path: string,
    public body: unknown,
  ) {
    const bodyStr = typeof body === 'string' ? body : JSON.stringify(body, null, 2);
    super(`Tripletex ${statusCode} on ${path}: ${bodyStr}`);
    this.name = 'ApiError';
  }
}

export interface ApiCall {
  method: string;
  path: string;
  status: number;
  elapsed: number;
}

export class TripletexApi {
  public callLog: ApiCall[] = [];

  constructor(private credentials: TripletexCredentials) {}

  async get(path: string, params?: Record<string, unknown>): Promise<any> {
    return this._call('GET', path, params);
  }

  async post(path: string, body?: unknown, params?: Record<string, unknown>): Promise<any> {
    return this._call('POST', path, params, body);
  }

  async put(path: string, body?: unknown, params?: Record<string, unknown>): Promise<any> {
    return this._call('PUT', path, params, body);
  }

  async del(path: string): Promise<any> {
    return this._call('DELETE', path);
  }

  async postList(path: string, items: unknown[]): Promise<any> {
    return this._call('POST', path, undefined, items);
  }

  getCallLog(): string {
    return this.callLog
      .map((c) => `${c.method} ${c.path} → ${c.status} (${c.elapsed}ms)`)
      .join('\n');
  }

  getErrorCount(): number {
    return this.callLog.filter((c) => c.status >= 400).length;
  }

  private async _call(
    method: string,
    path: string,
    params?: Record<string, unknown>,
    body?: unknown,
  ): Promise<any> {
    const start = Date.now();
    const res = await tripletexRequest(method, this.credentials, path, params, body);
    const elapsed = Date.now() - start;

    this.callLog.push({ method, path, status: res.status_code, elapsed });

    if (res.status_code >= 400) {
      throw new ApiError(res.status_code, path, res.data);
    }

    return res.data;
  }
}
