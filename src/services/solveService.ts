import { runAgent } from './geminiAgent.js';
import type { SolveRequestBody } from '../types.js';

export const solveService = {
  async solve(body: SolveRequestBody): Promise<void> {
    const imageAttachments = (body.files ?? [])
      .filter((f) => f.mime_type.startsWith('image/'))
      .map((f) => ({ mimeType: f.mime_type, data: f.content }));

    await runAgent(body.prompt, body.tripletex_credentials, imageAttachments);
  },
};
