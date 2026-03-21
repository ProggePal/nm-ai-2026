import type { TripletexApi } from './tripletexApi.js';

const EXEC_TIMEOUT_MS = 3 * 60 * 1000; // 3 minutes

export interface ExecResult {
  success: boolean;
  error?: string;
  logs: string[];
}

export function extractCode(responseText: string): string {
  // Try fenced typescript/ts/js block
  const fenced = responseText.match(/```(?:typescript|ts|js|javascript)\n([\s\S]*?)```/);
  if (fenced) return fenced[1].trim();

  // Try generic fenced block
  const generic = responseText.match(/```\n([\s\S]*?)```/);
  if (generic) return generic[1].trim();

  // Fallback: entire text (Claude may skip fences when told to output only code)
  return responseText.trim();
}

/**
 * Strip TypeScript type annotations so code can run in new Function() (which is JS, not TS).
 * Handles: (x: any) => ..., (x: string, y: number), function(x: Type), etc.
 */
function stripTypeAnnotations(code: string): string {
  // Remove type annotations from arrow function and function parameters: (x: Type) → (x)
  // Matches `: SomeType` after parameter names inside parentheses
  let result = code;

  // Remove `: type` annotations in function params — handles any, string, number, Type, Type[], etc.
  // Pattern: word followed by colon and type, within param context
  result = result.replace(/(\w)\s*:\s*(?:any|string|number|boolean|void|unknown|never|null|undefined|Record<[^>]*>|Array<[^>]*>|\w+(?:\[\])?)\s*(?=[,)\]=])/g, '$1');

  // Remove `as Type` casts
  result = result.replace(/\bas\s+(?:any|string|number|boolean|\w+(?:\[\])?)\b/g, '');

  // Remove interface/type declarations (full lines)
  result = result.replace(/^(?:interface|type)\s+\w+\s*(?:=\s*)?{[^}]*}\s*;?\s*$/gm, '');

  return result;
}

export async function executeCode(
  code: string,
  api: TripletexApi,
): Promise<ExecResult> {
  const logs: string[] = [];

  const sandboxConsole = {
    log: (...args: unknown[]) => logs.push(args.map(String).join(' ')),
    error: (...args: unknown[]) => logs.push('[ERROR] ' + args.map(String).join(' ')),
    warn: (...args: unknown[]) => logs.push('[WARN] ' + args.map(String).join(' ')),
    info: (...args: unknown[]) => logs.push(args.map(String).join(' ')),
  };

  // Strip TypeScript type annotations (new Function() runs JS, not TS)
  const jsCode = stripTypeAnnotations(code);

  try {
    const fn = new Function(
      'api',
      'console',
      `return (async () => {\n${jsCode}\n})();`,
    );

    const result = fn(api, sandboxConsole);

    // Race against timeout
    await Promise.race([
      result,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Code execution timed out after 3 minutes')), EXEC_TIMEOUT_MS),
      ),
    ]);

    return { success: true, logs };
  } catch (err) {
    const message = err instanceof Error
      ? `${err.message}${err.stack ? '\n' + err.stack : ''}`
      : String(err);
    return { success: false, error: message, logs };
  }
}
