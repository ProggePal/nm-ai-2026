#!/usr/bin/env bun
/**
 * Comprehensive competition-style tests against a real Tripletex sandbox.
 * Covers all task categories: employees, customers, products, invoicing,
 * travel expenses, projects, corrections, departments, and advanced workflows.
 *
 * Usage:
 *   bun scripts/test-model.ts              # run all tests
 *   bun scripts/test-model.ts --filter inv  # run only tests matching "inv"
 *
 * Requires TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN in env (or .env file).
 */

import { runAgent, verifySandboxResult } from '../src/services/claudeAgent.js';
import { mkdirSync, writeFileSync } from 'fs';
import { resolve, join } from 'path';

const credentials = {
  base_url: process.env.TRIPLETEX_BASE_URL!,
  session_token: process.env.TRIPLETEX_SESSION_TOKEN!,
};

if (!credentials.base_url || !credentials.session_token) {
  console.error('Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN in .env');
  process.exit(1);
}

// Parse --filter flag
const filterArg = process.argv.find((a) => a.startsWith('--filter'));
const filter = filterArg ? process.argv[process.argv.indexOf(filterArg) + 1]?.toLowerCase() : null;

// ---------------------------------------------------------------------------
// Test cases — organized by competition task category
// ---------------------------------------------------------------------------

interface TestCase {
  name: string;
  category: string;
  prompt: string;
  images?: Array<{ mimeType: string; data: string }>;
  pdfs?: Array<{ filename: string; data: string }>;
}

const testCases: TestCase[] = [
  // ── EMPLOYEES ──────────────────────────────────────────────────────────
  {
    name: 'Create employee (Norwegian)',
    category: 'Employees',
    prompt: 'Opprett en ansatt med navn Lars Hansen, e-post lars.hansen@bedrift.no. Han jobber i en avdeling.',
  },
  {
    name: 'Create employee with admin role (English)',
    category: 'Employees',
    prompt: 'Create an employee named Maria Silva with email maria.silva@company.no. She should be an account administrator (kontoadministrator).',
  },
  {
    name: 'Update employee contact info (Norwegian)',
    category: 'Employees',
    prompt: 'Finn den ansatte Lars Hansen og oppdater mobilnummeret til 99887766.',
  },

  // ── CUSTOMERS & PRODUCTS ──────────────────────────────────────────────
  {
    name: 'Create customer with address (Norwegian)',
    category: 'Customers & Products',
    prompt: 'Opprett kunden "Nordic Solutions AS" med org.nr 987654321, e-post post@nordic.no, og postadresse Storgata 15, 0184 Oslo.',
  },
  {
    name: 'Create product with price and VAT (English)',
    category: 'Customers & Products',
    prompt: 'Create a product called "Consulting Hours" with product number 1001, price excluding VAT of 1500 NOK, and the standard high-rate outgoing VAT.',
  },
  {
    name: 'Create customer (Spanish)',
    category: 'Customers & Products',
    prompt: 'Cree un cliente llamado "Barcelona Tech SL" con correo electrónico info@barcelonatech.es.',
  },

  // ── INVOICING ─────────────────────────────────────────────────────────
  {
    name: 'Create invoice for customer (Norwegian)',
    category: 'Invoicing',
    prompt: 'Opprett en faktura for kunden "Nordic Solutions AS". Fakturaen skal inneholde 10 timer konsulenttjenester til 1500 kr per time ekskl. mva. Bruk dagens dato som fakturadato og leveringsdato.',
  },
  {
    name: 'Register payment on invoice (English)',
    category: 'Invoicing',
    prompt: 'Find the most recent unpaid invoice for customer "Nordic Solutions AS" and register a full payment on it with today\'s date.',
  },
  {
    name: 'Create credit note (French)',
    category: 'Invoicing',
    prompt: 'Trouvez la dernière facture pour le client "Nordic Solutions AS" et créez une note de crédit pour cette facture.',
  },

  // ── TRAVEL EXPENSES ───────────────────────────────────────────────────
  {
    name: 'Register travel expense (Norwegian)',
    category: 'Travel Expenses',
    prompt: 'Opprett en reiseregning for den ansatte Lars Hansen. Reisen gikk fra Oslo til Bergen den 2026-03-15. Legg til kjøregodtgjørelse for 460 km.',
  },
  {
    name: 'Delete travel expense (English)',
    category: 'Travel Expenses',
    prompt: 'Find and delete the most recent travel expense report for employee Lars Hansen.',
  },

  // ── PROJECTS ──────────────────────────────────────────────────────────
  {
    name: 'Create project linked to customer (Norwegian)',
    category: 'Projects',
    prompt: 'Opprett et prosjekt med navn "Digitalisering 2026" for kunden "Nordic Solutions AS". Sett prosjektleder til Lars Hansen. Startdato 2026-04-01, sluttdato 2026-12-31.',
  },
  {
    name: 'Create project (Portuguese)',
    category: 'Projects',
    prompt: 'Crie um projeto chamado "Implementação ERP" com data de início 2026-05-01 e data de término 2026-11-30. Use qualquer funcionário como gerente de projeto.',
  },

  // ── CORRECTIONS ───────────────────────────────────────────────────────
  {
    name: 'Delete incorrect entry (English)',
    category: 'Corrections',
    prompt: 'Find and delete the customer named "Barcelona Tech SL" — it was created by mistake.',
  },

  // ── DEPARTMENTS ───────────────────────────────────────────────────────
  {
    name: 'Create department (Nynorsk)',
    category: 'Departments',
    prompt: 'Opprett ei avdeling med namnet "Forsking og utvikling" med avdelingsnummer 200.',
  },

  // ── DOCUMENTS (PDF / IMAGE) ────────────────────────────────────────────
  {
    name: 'Process PDF invoice (Norwegian)',
    category: 'Documents',
    prompt: 'Du har mottatt en faktura som PDF-vedlegg. Les fakturaen og opprett kunden "Testfirma AS" med org.nr 912345678 i Tripletex basert på informasjonen i dokumentet.',
    pdfs: [
      {
        filename: 'faktura-2026-001.pdf',
        data: 'JVBERi0xLjQKMSAwIG9iajw8L1R5cGUvQ2F0YWxvZy9QYWdlcyAyIDAgUj4+ZW5kb2JqCjIgMCBvYmo8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PmVuZG9iagozIDAgb2JqPDwvVHlwZS9QYWdlL01lZGlhQm94WzAgMCA2MTIgNzkyXS9QYXJlbnQgMiAwIFIvUmVzb3VyY2VzPDwvRm9udDw8L0YxIDQgMCBSPj4+Pi9Db250ZW50cyA1IDAgUj4+ZW5kb2JqCjQgMCBvYmo8PC9UeXBlL0ZvbnQvU3VidHlwZS9UeXBlMS9CYXNlRm9udC9IZWx2ZXRpY2E+PmVuZG9iago1IDAgb2JqPDwvTGVuZ3RoIDE3OD4+c3RyZWFtCkJUIC9GMSAxNiBUZiA1MCA3MDAgVGQgKEZBS1RVUkEpIFRqIDAgLTMwIFRkIC9GMSAxMiBUZiAoRmFrdHVyYW5yOiAyMDI2LTAwMSkgVGogMCAtMjAgVGQgKEt1bmRlOiBUZXN0ZmlybWEgQVMpIFRqIDAgLTIwIFRkIChPcmcubnI6IDkxMjM0NTY3OCkgVGogMCAtMjAgVGQgKEJlbG9wOiAyNTAwMCBOT0sgaW5rbC4gbXZhKSBUaiBFVAplbmRzdHJlYW0KZW5kb2JqCnhyZWYKMCA2CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAwOSAwMDAwMCBuIAowMDAwMDAwMDU4IDAwMDAwIG4gCjAwMDAwMDAxMTUgMDAwMDAgbiAKMDAwMDAwMDI2NiAwMDAwMCBuIAowMDAwMDAwMzQwIDAwMDAwIG4gCnRyYWlsZXI8PC9TaXplIDYvUm9vdCAxIDAgUj4+CnN0YXJ0eHJlZgo1NzIKJSVFT0Y=',
      },
    ],
  },

  // ── ADVANCED / MULTI-STEP ─────────────────────────────────────────────
  {
    name: 'Full invoice workflow (Norwegian)',
    category: 'Advanced',
    prompt: `Gjennomfør følgende:
1. Opprett kunden "Fjord Consulting AS" med e-post faktura@fjord.no
2. Opprett produktet "Prosjektledelse" med pris 2000 kr ekskl. mva og standard utgående mva høy sats
3. Opprett en ordre for kunden med dagens dato, legg til 5 stk "Prosjektledelse"
4. Fakturer ordren med dagens dato`,
  },
  {
    name: 'Employee + project + timesheet (English)',
    category: 'Advanced',
    prompt: `Complete this multi-step workflow:
1. Create an employee named "Test Worker" with email test.worker@company.no
2. Create a project named "Q2 Delivery" with the new employee as project manager, start date 2026-04-01
3. The project should be linked to an existing customer if one exists, otherwise create customer "Delivery Corp AS"`,
  },
];

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

const selected = filter
  ? testCases.filter((tc) => tc.name.toLowerCase().includes(filter) || tc.category.toLowerCase().includes(filter))
  : testCases;

console.log(`=== Comprehensive Agent Test (claude-sonnet-4-6) ===`);
console.log(`Running ${selected.length} of ${testCases.length} tests${filter ? ` (filter: "${filter}")` : ''}\n`);

interface Result {
  name: string;
  category: string;
  toolCalls: number;
  errors: number;
  elapsed: string;
  verified: boolean;
  summary: string;
  status: 'PASS' | 'FAIL' | 'ERROR';
}

// Create output directory for this run
const runTimestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
const runDir = join('test-results', runTimestamp);
mkdirSync(runDir, { recursive: true });

const results: Result[] = [];
const fullResults: any[] = [];

for (const tc of selected) {
  console.log(`━━━ [${tc.category}] ${tc.name} ━━━`);
  console.log(`Prompt: ${tc.prompt.slice(0, 120)}${tc.prompt.length > 120 ? '...' : ''}\n`);

  const start = Date.now();
  try {
    const result = await runAgent(tc.prompt, credentials, tc.images ?? [], tc.pdfs ?? []);
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    const verification = await verifySandboxResult(tc.prompt, result);

    console.log(`  Tool calls: ${result.toolCallCount}`);
    console.log(`  Errors:     ${result.errors.length}`);
    console.log(`  Elapsed:    ${elapsed}s`);
    console.log(`  Verified:   ${verification.verified}`);
    console.log(`  Summary:    ${verification.summary}`);

    const pass = result.toolCallCount > 0 && parseFloat(elapsed) < 270;
    console.log(`  ${pass ? 'PASS' : 'FAIL'}\n`);

    const resultEntry = {
      name: tc.name,
      category: tc.category,
      toolCalls: result.toolCallCount,
      errors: result.errors.length,
      elapsed,
      verified: verification.verified,
      summary: verification.summary,
      status: pass ? 'PASS' : 'FAIL',
    };
    results.push(resultEntry);

    // Save full result per test
    const slug = tc.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 50);
    fullResults.push({
      ...resultEntry,
      prompt: tc.prompt,
      systemPrompt: result.systemPrompt.map((b: any) => b.text),
      messages: result.messages,
      agentErrors: result.errors,
      verification,
    });
  } catch (err) {
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    console.log(`  ERROR after ${elapsed}s: ${err}\n`);
    results.push({
      name: tc.name,
      category: tc.category,
      toolCalls: 0,
      errors: 1,
      elapsed,
      verified: false,
      summary: String(err),
      status: 'ERROR',
    });
    fullResults.push({
      name: tc.name,
      category: tc.category,
      prompt: tc.prompt,
      error: String(err),
      status: 'ERROR',
      elapsed,
    });
  }
}

// ---------------------------------------------------------------------------
// Summary table
// ---------------------------------------------------------------------------

console.log(`\n${'═'.repeat(100)}`);
console.log('RESULTS SUMMARY');
console.log(`${'═'.repeat(100)}`);
console.log(
  `${'Category'.padEnd(22)} ${'Test'.padEnd(42)} ${'Calls'.padStart(5)} ${'Errs'.padStart(5)} ${'Time'.padStart(7)} ${'Verified'.padStart(9)} Status`
);
console.log(`${'─'.repeat(100)}`);

for (const r of results) {
  const icon = r.status === 'PASS' ? 'OK' : r.status === 'FAIL' ? 'FAIL' : 'ERR';
  console.log(
    `${r.category.padEnd(22)} ${r.name.slice(0, 40).padEnd(42)} ${String(r.toolCalls).padStart(5)} ${String(r.errors).padStart(5)} ${(r.elapsed + 's').padStart(7)} ${String(r.verified).padStart(9)} ${icon}`
  );
}

console.log(`${'─'.repeat(100)}`);

const passed = results.filter((r) => r.status === 'PASS').length;
const failed = results.filter((r) => r.status !== 'PASS').length;
const totalCalls = results.reduce((sum, r) => sum + r.toolCalls, 0);
const totalErrors = results.reduce((sum, r) => sum + r.errors, 0);
const verifiedCount = results.filter((r) => r.verified).length;

console.log(`\nTotal: ${passed} passed, ${failed} failed | ${totalCalls} API calls, ${totalErrors} errors | ${verifiedCount}/${results.length} verified`);

// ---------------------------------------------------------------------------
// Write full results to disk
// ---------------------------------------------------------------------------

const runSummary = {
  timestamp: new Date().toISOString(),
  model: 'claude-sonnet-4-6',
  testsRun: results.length,
  passed,
  failed,
  totalCalls,
  totalErrors,
  verified: verifiedCount,
  results: fullResults,
};

const outPath = join(runDir, 'results.json');
writeFileSync(outPath, JSON.stringify(runSummary, null, 2));
console.log(`\nFull results saved to: ${outPath}`);

process.exit(failed > 0 ? 1 : 0);
