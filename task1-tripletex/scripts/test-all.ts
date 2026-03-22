#!/usr/bin/env bun
/**
 * Test runner — runs all unique prompts from test-requests/ against the sandbox.
 * Sets up sandbox preconditions before each test, then calls /solve with use_sandbox=true.
 *
 * Usage: bun scripts/test-all.ts
 */

import { readdirSync, readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { tripletexRequest } from '../src/services/tripletexClient.js';
import type { TripletexCredentials } from '../src/types.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Env
// ---------------------------------------------------------------------------

const BASE_URL       = process.env.TRIPLETEX_BASE_URL!;
const SESSION_TOKEN  = process.env.TRIPLETEX_SESSION_TOKEN!;
const SERVICE_URL    = (process.env.SERVICE_URL ?? '').replace(/\/$/, '');
const JWT_TOKEN      = process.env.JWT_TOKEN!;

if (!BASE_URL || !SESSION_TOKEN || !SERVICE_URL || !JWT_TOKEN) {
  console.error('Missing env vars — check .env file');
  process.exit(1);
}

const CREDS: TripletexCredentials = { base_url: BASE_URL, session_token: SESSION_TOKEN };

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RequestFile {
  prompt: string;
}

interface TestResult {
  prompt: string;
  verified: boolean | null;
  elapsed: number;
  toolCalls: number;
  errorCount: number;
  summary: string;
  httpStatus: number;
}

// ---------------------------------------------------------------------------
// Tripletex setup helpers
// ---------------------------------------------------------------------------

async function findCustomerByOrgNr(orgNr: string): Promise<number | null> {
  const res = await tripletexRequest('GET', CREDS, '/customer', {
    organizationNumber: orgNr,
    fields: 'id,name',
    count: 10,
  });
  const values = (res.data as any)?.values ?? [];
  return values[0]?.id ?? null;
}

async function ensureCustomer(name: string, orgNr: string, logs: string[]): Promise<number> {
  const existing = await findCustomerByOrgNr(orgNr);
  if (existing) {
    logs.push(`Customer ${name} already exists (id=${existing})`);
    return existing;
  }
  const res = await tripletexRequest('POST', CREDS, '/customer', undefined, { name, organizationNumber: orgNr });
  const id = (res.data as any)?.value?.id;
  if (!id) throw new Error(`Failed to create customer: ${JSON.stringify(res)}`);
  logs.push(`Created customer ${name} (id=${id})`);
  return id;
}

async function deleteCustomerByOrgNr(orgNr: string, logs: string[]): Promise<void> {
  const res = await tripletexRequest('GET', CREDS, '/customer', {
    organizationNumber: orgNr,
    fields: 'id,name',
    count: 10,
  });
  const values = (res.data as any)?.values ?? [];
  if (values.length === 0) { logs.push(`No customer found with org.nr ${orgNr} — already clean`); return; }
  for (const c of values) {
    const del = await tripletexRequest('DELETE', CREDS, `/customer/${c.id}`);
    if (del.status_code === 204 || del.status_code === 200) {
      logs.push(`Deleted customer ${c.name} (id=${c.id})`);
    } else if (del.status_code === 409) {
      logs.push(`Cannot delete ${c.name} (id=${c.id}) — has linked records (409), skipping`);
    } else {
      logs.push(`DELETE /customer/${c.id} → HTTP ${del.status_code}, skipping`);
    }
  }
}

async function ensureUnpaidInvoice(customerId: number, amount: number, description: string, logs: string[]): Promise<void> {
  const today = new Date().toISOString().slice(0, 10);
  const invoices = await tripletexRequest('GET', CREDS, '/invoice', {
    customerId,
    invoiceDateFrom: '2020-01-01',
    invoiceDateTo: '2030-01-01',
    fields: 'id,amountOutstanding',
    count: 20,
  });
  const unpaid = ((invoices.data as any)?.values ?? []).find((i: any) => i.amountOutstanding > 0);
  if (unpaid) { logs.push(`Unpaid invoice already exists (id=${unpaid.id}, outstanding=${unpaid.amountOutstanding})`); return; }

  const order = await tripletexRequest('POST', CREDS, '/order', undefined, {
    customer: { id: customerId },
    orderDate: today,
    deliveryDate: today,
  });
  const orderId = (order.data as any)?.value?.id;
  if (!orderId) throw new Error(`Failed to create order: ${JSON.stringify(order)}`);
  logs.push(`Created order id=${orderId}`);

  await tripletexRequest('POST', CREDS, '/order/orderline', undefined, {
    order: { id: orderId },
    description,
    count: 1,
    unitPriceExcludingVatCurrency: amount,
  });

  const invoice = await tripletexRequest('PUT', CREDS, `/order/${orderId}/:invoice`, { invoiceDate: today }, {});
  const invoiceId = (invoice.data as any)?.value?.id;
  logs.push(`Created invoice id=${invoiceId} for ${amount} kr`);
}

async function deleteDepartmentsByNames(names: string[], logs: string[]): Promise<void> {
  const res = await tripletexRequest('GET', CREDS, '/department', { fields: 'id,name', count: 100 });
  const all = (res.data as any)?.values ?? [];
  for (const name of names) {
    const matches = all.filter((d: any) => d.name.toLowerCase() === name.toLowerCase());
    if (matches.length === 0) { logs.push(`Department "${name}" not found — already clean`); continue; }
    for (const dept of matches) {
      const del = await tripletexRequest('DELETE', CREDS, `/department/${dept.id}`);
      if (del.status_code === 204 || del.status_code === 200) {
        logs.push(`Deleted department "${dept.name}" (id=${dept.id})`);
      } else {
        logs.push(`Cannot delete "${dept.name}" (id=${dept.id}): HTTP ${del.status_code}`);
      }
    }
  }
}

async function ensureOpenOrder(customerId: number, amount: number, description: string, logs: string[]): Promise<void> {
  const today = new Date().toISOString().slice(0, 10);
  const order = await tripletexRequest('POST', CREDS, '/order', undefined, {
    customer: { id: customerId },
    orderDate: today,
    deliveryDate: today,
  });
  const orderId = (order.data as any)?.value?.id;
  if (!orderId) throw new Error(`Failed to create order: ${JSON.stringify(order)}`);
  await tripletexRequest('POST', CREDS, '/order/orderline', undefined, {
    order: { id: orderId },
    description,
    count: 1,
    unitPriceExcludingVatCurrency: amount,
  });
  logs.push(`Created open order id=${orderId} with "${description}" @ ${amount} NOK`);
}

// ---------------------------------------------------------------------------
// Prompt parsing
// ---------------------------------------------------------------------------

function parseOrgNr(prompt: string): string {
  const match = prompt.match(/\b(\d{9})\b/);
  if (!match) throw new Error(`No 9-digit org.nr found in: ${prompt.slice(0, 80)}`);
  return match[1];
}

function parseCustomerName(prompt: string): string {
  const norwegian = prompt.match(/kunden\s+(.+?)\s+(?:med\s+organisasjonsnummer|\(org\.nr)/i);
  if (norwegian) return norwegian[1].trim();
  const french = prompt.match(/client\s+(.+?)\s*\(n[oº°]\s*org/i);
  if (french) return french[1].trim();
  const english = prompt.match(/customer\s+(.+?)\s*(?:\(org|with org)/i);
  if (english) return english[1].trim();
  return 'Unknown Customer';
}

function parseAmount(prompt: string): number {
  const match = prompt.match(/(\d[\d\s]*\d|\d+)\s*(?:kr|NOK|nok)/i);
  if (!match) return 31300; // fallback
  return parseInt(match[1].replace(/\s/g, ''), 10);
}

function parseDescription(prompt: string): string {
  const forQuoted = prompt.match(/for\s+"([^"]+)"/i);
  if (forQuoted) return forQuoted[1];
  const concernant = prompt.match(/(?:concernant|concerne)\s+(.+?)[\.,\n]/i);
  if (concernant) return concernant[1].trim();
  return 'Konsulenttimer';
}

function parseDepartmentNames(prompt: string): string[] {
  return [...prompt.matchAll(/"([^"]+)"/g)].map(m => m[1]);
}

// ---------------------------------------------------------------------------
// Setup dispatcher
// ---------------------------------------------------------------------------

async function runSetup(prompt: string): Promise<string[]> {
  const p = prompt.toLowerCase();
  const logs: string[] = [];

  // 1. Invoice + send (check before plain create customer to avoid French prompt misclassification)
  if ((p.includes('faktura') || p.includes('facture') || p.includes('invoice')) &&
      (p.includes('send') || p.includes('envoy') || p.includes('sende'))) {
    const orgNr = parseOrgNr(prompt);
    const name = parseCustomerName(prompt);
    const amount = parseAmount(prompt);
    const description = parseDescription(prompt);
    logs.push(`[create+send invoice] Ensure ${name} (${orgNr}) exists with open order for ${amount} NOK`);
    const customerId = await ensureCustomer(name, orgNr, logs);
    await ensureOpenOrder(customerId, amount, description, logs);
    return logs;
  }

  // 2. Register payment on invoice
  if ((p.includes('betaling') || p.includes('payment') || p.includes('paiement')) &&
      (p.includes('faktura') || p.includes('invoice') || p.includes('facture'))) {
    const orgNr = parseOrgNr(prompt);
    const name = parseCustomerName(prompt);
    const amount = parseAmount(prompt);
    logs.push(`[register payment] Ensure ${name} (${orgNr}) has unpaid invoice for ${amount} kr`);
    const customerId = await ensureCustomer(name, orgNr, logs);
    await ensureUnpaidInvoice(customerId, amount, parseDescription(prompt), logs);
    return logs;
  }

  // 3. Create customer — delete existing for clean slate
  if (p.includes('opprett kunden') || p.includes('create customer') ||
      (p.includes('créez') && p.includes('client') && !p.includes('facture'))) {
    const orgNr = parseOrgNr(prompt);
    const name = parseCustomerName(prompt);
    logs.push(`[create customer] Delete ${name} (${orgNr}) if exists`);
    await deleteCustomerByOrgNr(orgNr, logs);
    return logs;
  }

  // 4. Create departments
  if (p.includes('avdeling') || p.includes('department') || p.includes('avdelingar')) {
    const names = parseDepartmentNames(prompt);
    if (names.length > 0) {
      logs.push(`[create departments] Delete existing: ${names.join(', ')}`);
      await deleteDepartmentsByNames(names, logs);
    }
    return logs;
  }

  logs.push('[read only] No setup needed');
  return logs;
}

// ---------------------------------------------------------------------------
// Solve call
// ---------------------------------------------------------------------------

async function callSolve(prompt: string): Promise<{ status: number; body: any; elapsed: number }> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3 * 60 * 1000);
  const start = Date.now();
  try {
    const res = await fetch(`${SERVICE_URL}/solve`, {
      method: 'POST',
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${JWT_TOKEN}`,
      },
      body: JSON.stringify({
        prompt,
        files: [],
        tripletex_credentials: { base_url: BASE_URL, session_token: SESSION_TOKEN },
        use_sandbox: true,
      }),
    });
    const elapsed = (Date.now() - start) / 1000;
    const body = await res.json();
    return { status: res.status, body, elapsed };
  } finally {
    clearTimeout(timeout);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const TEST_DIR = join(__dirname, '../test-requests');
const files = readdirSync(TEST_DIR).filter(f => f.endsWith('.json')).sort();

// Deduplicate by prompt
const seen = new Set<string>();
const tests: RequestFile[] = [];
for (const f of files) {
  const body = JSON.parse(readFileSync(join(TEST_DIR, f), 'utf-8')) as RequestFile;
  if (body.prompt && !seen.has(body.prompt.trim())) {
    seen.add(body.prompt.trim());
    tests.push(body);
  }
}

console.log(`\n🧪 ${files.length} files → ${tests.length} unique prompts\n${'─'.repeat(70)}`);

const results: TestResult[] = [];

for (let i = 0; i < tests.length; i++) {
  const { prompt } = tests[i];
  const short = prompt.length > 65 ? prompt.slice(0, 62) + '...' : prompt;

  console.log(`\n[${i + 1}/${tests.length}] ${short}`);

  // Setup
  try {
    const logs = await runSetup(prompt);
    for (const log of logs) console.log(`  ↳ ${log}`);
  } catch (err) {
    console.log(`  ⚠️  Setup error: ${err}`);
  }

  // Solve
  console.log(`  🚀 Solving...`);
  let result: TestResult;
  try {
    const { status, body, elapsed } = await callSolve(prompt);
    const verified = body.verified ?? null;
    const toolCalls = body.tool_calls ?? 0;
    const errorCount = body.errors?.length ?? 0;
    const summary = body.summary ?? body.error?.message ?? '';
    const icon = status !== 200 ? '❌' : verified === true ? '✅' : '⚠️';
    console.log(`  ${icon} HTTP ${status} | ${elapsed.toFixed(1)}s | ${toolCalls} calls | ${errorCount} errors`);
    if (summary) console.log(`  💬 ${summary}`);
    result = { prompt, verified, elapsed, toolCalls, errorCount, summary, httpStatus: status };
  } catch (err) {
    console.log(`  ❌ Network error: ${err}`);
    result = { prompt, verified: null, elapsed: 0, toolCalls: 0, errorCount: 1, summary: String(err), httpStatus: 0 };
  }

  results.push(result);
}

// Summary
const passed = results.filter(r => r.verified === true).length;
console.log(`\n${'─'.repeat(70)}`);
console.log(`RESULTS: ${passed}/${results.length} verified\n`);
for (const r of results) {
  const icon = r.verified === true ? '✅' : r.verified === false ? '❌' : '⚠️';
  const p = r.prompt.length > 45 ? r.prompt.slice(0, 42) + '...' : r.prompt;
  console.log(`${icon}  ${r.elapsed.toFixed(1)}s  ${String(r.toolCalls).padStart(2)} calls  ${r.errorCount} err  — ${p}`);
}
console.log();
