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
  // Conservative: only strip known TS type annotations in function/arrow params
  // Pattern: after ( or , + identifier + : type — NOT object properties like { number: 1920 }
  return code
    .replace(/([(,]\s*\w+)\s*:\s*(?:any|string|number|boolean|void|unknown|never|null|undefined)(\s*[,)=])/g, '$1$2')
    .replace(/\s+as\s+any\b/g, '');
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
