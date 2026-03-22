#!/usr/bin/env bun
/**
 * Calls runAgent directly and prints the full AI conversation.
 *
 * Usage: bun scripts/test-agent.ts "Your prompt here"
 */

import { runAgent } from '../src/services/claudeAgent.js';

const prompt = process.argv.slice(2).join(' ');

if (!prompt) {
  console.error('Usage: bun scripts/test-agent.ts "your prompt here"');
  process.exit(1);
}

const credentials = {
  base_url: process.env.TRIPLETEX_BASE_URL!,
  session_token: process.env.TRIPLETEX_SESSION_TOKEN!,
};

if (!credentials.base_url || !credentials.session_token) {
  console.error('Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN in .env');
  process.exit(1);
}

console.log(`\n--- Prompt ---\n${prompt}\n`);

const start = Date.now();
const result = await runAgent(prompt, credentials);
const elapsed = ((Date.now() - start) / 1000).toFixed(1);

console.log(`\n--- Agent Result ---`);
console.log(`Tool calls: ${result.toolCallCount}`);
console.log(`Errors: ${result.errors.length}`);
console.log(`Elapsed: ${elapsed}s`);
console.log(`Turns: ${result.messages.length}`);

console.log(`\n--- Conversation ---`);
for (const msg of result.messages) {
  console.log(`\n[${msg.role.toUpperCase()}]`);
  if (Array.isArray(msg.content)) {
    for (const block of msg.content) {
      if (block.type === 'text') {
        console.log(block.text);
      } else if (block.type === 'mcp_tool_use') {
        console.log(`  🔧 ${block.name}(${JSON.stringify(block.input)})`);
      } else if (block.type === 'mcp_tool_result') {
        const preview = JSON.stringify(block.content).slice(0, 300);
        console.log(`  ${block.is_error ? '❌' : '✅'} ${preview}`);
      } else if (block.type === 'server_tool_use') {
        console.log(`  🔍 tool_search(${JSON.stringify(block.input)})`);
      } else if (block.type === 'server_tool_result') {
        const preview = JSON.stringify(block.content).slice(0, 300);
        console.log(`  📋 ${preview}`);
      } else {
        console.log(`  [${block.type}]`, JSON.stringify(block).slice(0, 200));
      }
    }
  } else if (typeof msg.content === 'string') {
    console.log(msg.content);
  }
}
