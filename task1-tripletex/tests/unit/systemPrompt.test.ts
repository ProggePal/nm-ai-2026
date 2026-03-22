import { describe, it, expect } from 'bun:test';
import { buildSystemPrompt } from '../../src/services/systemPrompt.js';

describe('buildSystemPrompt', () => {
  it('always returns the base block as first element', () => {
    const blocks = buildSystemPrompt('Create a customer');
    expect(blocks.length).toBeGreaterThanOrEqual(1);
    expect(blocks[0].type).toBe('text');
    expect((blocks[0] as any).text).toContain('You are an expert accounting AI agent');
  });

  it('returns only the base block for generic prompts', () => {
    const blocks = buildSystemPrompt('Opprett en ansatt med navn Ola Nordmann');
    expect(blocks).toHaveLength(1);
  });

  // Bank reconciliation module
  it('loads bank reconciliation module for "bank" keyword', () => {
    const blocks = buildSystemPrompt('Gjør en bankavstemming for konto 1920');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Bank Reconciliation');
  });

  it('loads bank reconciliation module for "avstemming" keyword', () => {
    const blocks = buildSystemPrompt('Utfør avstemming av hovedboken');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Bank Reconciliation');
  });

  it('loads bank reconciliation module for English "reconciliation"', () => {
    const blocks = buildSystemPrompt('Perform reconciliation of bank account');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Bank Reconciliation');
  });

  // Asset management module
  it('loads asset module for "eiendel" keyword', () => {
    const blocks = buildSystemPrompt('Registrer en ny eiendel');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Asset Management');
  });

  it('loads asset module for "depreciation" keyword', () => {
    const blocks = buildSystemPrompt('Calculate depreciation for the office equipment');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Asset Management');
  });

  it('loads asset module for "avskrivning" keyword', () => {
    const blocks = buildSystemPrompt('Beregn avskrivning for kontorutstyr');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Asset Management');
  });

  // Balance sheet module
  it('loads balance sheet module for "balanse" keyword', () => {
    const blocks = buildSystemPrompt('Vis balanse for 2025');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Balance Sheet');
  });

  it('loads balance sheet module for "profit and loss" keyword', () => {
    const blocks = buildSystemPrompt('Generate a profit and loss report');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Balance Sheet');
  });

  it('loads balance sheet module for "resultatregnskap" keyword', () => {
    const blocks = buildSystemPrompt('Opprett resultatregnskap for perioden');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Balance Sheet');
  });

  // Salary reconciliation module
  // Note: "lønnsavstemming" contains "avstemming" which also matches bank reconciliation.
  // We check that the salary module is loaded somewhere in the blocks.
  it('loads salary module for "lønnsavstemming" keyword', () => {
    const blocks = buildSystemPrompt('Gjennomfør lønnsavstemming for mars');
    const texts = blocks.map((b: any) => b.text);
    expect(texts.some((t: string) => t.includes('Salary Reconciliation'))).toBe(true);
  });

  it('loads salary module for "feriepenger" keyword', () => {
    const blocks = buildSystemPrompt('Avstem feriepenger for 2025');
    const texts = blocks.map((b: any) => b.text);
    expect(texts.some((t: string) => t.includes('Salary Reconciliation'))).toBe(true);
  });

  it('loads salary module for "payroll reconciliation" English keyword', () => {
    // "payroll reconciliation" matches both bank ("reconciliation") and salary ("payroll reconciliation")
    const blocks = buildSystemPrompt('Run payroll reconciliation for Q1');
    const texts = blocks.map((b: any) => b.text);
    expect(texts.some((t: string) => t.includes('Salary Reconciliation'))).toBe(true);
  });

  // Multiple modules
  it('loads multiple modules when prompt matches several', () => {
    const blocks = buildSystemPrompt('Perform bank reconciliation and calculate depreciation for assets');
    // Should match both bank reconciliation and asset management
    expect(blocks.length).toBeGreaterThanOrEqual(3);
    const texts = blocks.map((b: any) => b.text);
    expect(texts.some((t: string) => t.includes('Bank Reconciliation'))).toBe(true);
    expect(texts.some((t: string) => t.includes('Asset Management'))).toBe(true);
  });

  // Case insensitivity
  it('matches keywords case-insensitively', () => {
    const blocks = buildSystemPrompt('BANK RECONCILIATION needed');
    expect(blocks.length).toBeGreaterThan(1);
    expect((blocks[1] as any).text).toContain('Bank Reconciliation');
  });

  // Cache control
  it('all blocks have cache_control set to ephemeral', () => {
    const blocks = buildSystemPrompt('Perform bank reconciliation and view balanse');
    for (const block of blocks) {
      expect((block as any).cache_control).toEqual({ type: 'ephemeral' });
    }
  });

  // Base block content checks
  it('base block contains API conventions', () => {
    const blocks = buildSystemPrompt('anything');
    const base = (blocks[0] as any).text;
    expect(base).toContain('API conventions');
    expect(base).toContain('fields param');
  });

  it('base block contains field schemas', () => {
    const blocks = buildSystemPrompt('anything');
    const base = (blocks[0] as any).text;
    expect(base).toContain('Customer');
    expect(base).toContain('Employee');
    expect(base).toContain('Invoice');
    expect(base).toContain('Travel expense');
  });

  it('base block contains efficiency rules', () => {
    const blocks = buildSystemPrompt('anything');
    const base = (blocks[0] as any).text;
    expect(base).toContain('Efficiency rules');
    expect(base).toContain('Minimize total API calls');
  });

  it('base block mentions multilingual support', () => {
    const blocks = buildSystemPrompt('anything');
    const base = (blocks[0] as any).text;
    expect(base).toContain('Norwegian');
    expect(base).toContain('English');
  });
});
