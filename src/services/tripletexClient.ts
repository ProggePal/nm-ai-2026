import type { TripletexCredentials } from '../types.js';

export interface TripletexResponse {
  status_code: number;
  data?: unknown;
  error?: string;
}

export async function tripletexRequest(
  method: string,
  credentials: TripletexCredentials,
  path: string,
  params?: Record<string, unknown>,
  body?: unknown
): Promise<TripletexResponse> {
  const base = credentials.proxy_url.replace(/\/$/, '');
  const url = new URL(`${base}${path.startsWith('/') ? path : `/${path}`}`);

  if (params) {
    for (const [key, val] of Object.entries(params)) {
      if (val !== undefined && val !== null) {
        url.searchParams.set(key, String(val));
      }
    }
  }

  const authHeader = `Basic ${btoa(`0:${credentials.session_token}`)}`;

  try {
    const res = await fetch(url.toString(), {
      method: method.toUpperCase(),
      headers: {
        Authorization: authHeader,
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });

    let data: unknown;
    try {
      data = await res.json();
    } catch {
      data = { _raw: await res.text() };
    }

    return { status_code: res.status, data };
  } catch (err) {
    return { status_code: 0, error: String(err) };
  }
}
