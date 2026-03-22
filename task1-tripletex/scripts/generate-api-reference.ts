#!/usr/bin/env bun
/**
 * Generates a compact API reference from the Tripletex OpenAPI spec.
 * Only includes fields that are useful for creating/updating entities.
 * Output is designed to be embedded in the system prompt.
 *
 * Usage: bun scripts/generate-api-reference.ts [path-to-openapi.json]
 */

import { readFileSync, mkdirSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';

const specPath = process.argv[2] || '/Users/hansjorgentorp/nmiai/trippeltex-mcp/openapi.json';
const spec = JSON.parse(readFileSync(specPath, 'utf-8'));

const schemas = spec.components?.schemas || {};

// Fields to always skip (read-only, system-generated)
const SKIP_FIELDS = new Set([
  'id', 'version', 'changes', 'url', 'displayName', 'displayNumber',
  'nameAndNumber', 'numberAsString', 'hierarchyNameAndNumber',
  'customerName', 'projectManagerNameAndNumber',
]);

// Fields that are read-only or rarely used for creation
const SKIP_READONLY = new Set([
  'isDeletable', 'hasSupplierProductConnected', 'canCreateBackorder',
  'sendMethodDescription', 'isAuthProjectOverviewURL', 'pictureId',
  'companyId', 'vismaConnect2FAactive', 'accountingPeriodClosed',
  'accountingPeriodVATClosed', 'isSalaryAdmin', 'showPayslip',
  'stateName', 'actions', 'attachmentCount', 'state', 'type',
  'contributionMarginPercent', 'numberOfSubProjects', 'numberOfProjectParticipants',
  'hierarchyLevel', 'accessType', 'amount', 'chargeableAmount',
  'chargeableAmountCurrency', 'paymentAmount', 'paymentAmountCurrency',
  'lowRateVAT', 'mediumRateVAT', 'highRateVAT',
]);

// Entities for the competition — organized by importance
const TARGETS: Array<{ name: string; comment?: string }> = [
  { name: 'Employee' },
  { name: 'Customer' },
  { name: 'Product' },
  { name: 'Department' },
  { name: 'Order' },
  { name: 'OrderLine' },
  { name: 'Project' },
  { name: 'Invoice', comment: 'Read-only — invoices are created via PUT /order/{id}/:invoice' },
  { name: 'Posting', comment: 'Used inside Voucher.postings array' },
  { name: 'Voucher' },
  { name: 'TravelExpense' },
  { name: 'Cost', comment: 'TravelExpense cost line' },
  { name: 'MileageAllowance' },
  { name: 'PerDiemCompensation' },
  { name: 'TimesheetEntry' },
  { name: 'SalaryTransaction' },
  { name: 'Payslip', comment: 'Inside SalaryTransaction.payslips' },
  { name: 'SalarySpecification', comment: 'Inside Payslip.specifications' },
  { name: 'SalaryType', comment: 'Reference — look up with GET /salary/type' },
  { name: 'SupplierInvoice', comment: 'Read-only — use /incomingInvoice to create' },
  { name: 'IncomingInvoiceHeaderWrite', comment: 'For POST /incomingInvoice' },
  { name: 'IncomingOrderLineWrite', comment: 'Order lines inside incoming invoice' },
  { name: 'AccountingDimensionName' },
  { name: 'AccountingDimensionValue' },
  { name: 'Account', comment: 'Ledger account — GET /ledger/account' },
  { name: 'VatType', comment: 'GET /ledger/vatType' },
];

function getType(prop: any): string {
  if (prop.$ref) return prop.$ref.split('/').pop()!;
  if (prop.type === 'array' && prop.items?.$ref) return `[${prop.items.$ref.split('/').pop()}]`;
  if (prop.type === 'array') return `[${prop.items?.type || 'any'}]`;
  if (prop.enum) return prop.enum.slice(0, 6).join('|') + (prop.enum.length > 6 ? '|...' : '');
  return prop.type || 'unknown';
}

const output: string[] = [];

for (const target of TARGETS) {
  const schema = schemas[target.name];
  if (!schema?.properties) continue;

  const required = new Set(schema.required || []);
  const props = Object.entries(schema.properties as Record<string, any>)
    .filter(([name]) => !SKIP_FIELDS.has(name) && !SKIP_READONLY.has(name));

  if (props.length === 0) continue;

  const header = target.comment ? `${target.name} — ${target.comment}` : target.name;
  output.push(`### ${header}`);

  for (const [name, prop] of props) {
    const type = getType(prop);
    const req = required.has(name) ? ' (R)' : '';
    // Keep descriptions short — just the first sentence or 60 chars
    let desc = '';
    if (prop.description) {
      const firstSentence = prop.description.split(/[.\n<]/)[0].trim();
      if (firstSentence.length > 5 && firstSentence.length < 80) {
        desc = ` — ${firstSentence}`;
      }
    }
    output.push(`  ${name}${req}: ${type}${desc}`);
  }
  output.push('');
}

const result = output.join('\n');

const outDir = resolve(dirname(new URL(import.meta.url).pathname), '../src/generated');
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, 'api-reference.txt');
writeFileSync(outPath, result);

console.log(`Generated: ${result.length} chars, ${output.length} lines`);
console.log(`Written to: ${outPath}`);
console.log(`\nSchemas included: ${TARGETS.filter(t => schemas[t.name]?.properties).length}/${TARGETS.length}`);
