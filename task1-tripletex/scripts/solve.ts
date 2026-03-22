#!/usr/bin/env bun
/**
 * Test script — send a prompt to the solve service using playground credentials.
 *
 * Usage:
 *   bun run scripts/solve.ts "Opprett kunden Acme AS med org.nr 123456789"
 *   bun run scripts/solve.ts --local "Your prompt here"   (runs against localhost:5000)
 */

const args = process.argv.slice(2);
const localMode = args.includes('--local');
const sandboxMode = args.includes('--sandbox');
const prompt = args.filter((a) => !a.startsWith('--')).join(' ');

if (!prompt) {
  console.error('Usage: bun scripts/solve.ts [--local] [--sandbox] "your prompt here"');
  process.exit(1);
}

const baseUrl = process.env.TRIPLETEX_BASE_URL;
const sessionToken = process.env.TRIPLETEX_SESSION_TOKEN;
const jwtToken = process.env.JWT_TOKEN;
const serviceUrl = localMode ? 'http://localhost:5000' : process.env.SERVICE_URL;

if (!baseUrl || !sessionToken || !jwtToken || !serviceUrl) {
  console.error('Missing env vars — check your .env file');
  process.exit(1);
}

console.log(`\n📤 Sending to: ${serviceUrl}/solve`);
console.log(`📝 Prompt: ${prompt}`);
if (sandboxMode) console.log(`🧪 Sandbox mode: ON\n`);
else console.log();

const start = Date.now();

const res = await fetch(`${serviceUrl}/solve`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${jwtToken}`,
  },
  body: JSON.stringify({
    prompt,
    files: [],
    tripletex_credentials: {
      base_url: baseUrl,
      session_token: sessionToken,
    },
    use_sandbox: sandboxMode || undefined,
  }),
});

const elapsed = ((Date.now() - start) / 1000).toFixed(2);
const body = await res.json();

console.log(`✅ Status: ${res.status} (${elapsed}s)`);
console.log(`📦 Response:`, body);
