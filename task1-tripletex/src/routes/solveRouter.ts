import { Router, type Request, type Response, type NextFunction } from 'express';
import { celebrate, Joi, Segments } from 'celebrate';
import { logRequest } from '../services/requestLogger.js';
import { solveService } from '../services/solveService.js';
import type { SolveRequestBody } from '../types.js';

export const solveRouter = Router();

const validateSolve = celebrate({
  [Segments.BODY]: Joi.object({
    prompt: Joi.string().required(),
    tripletex_credentials: Joi.object({
      base_url: Joi.string().uri().required(),
      session_token: Joi.string().required(),
    }).required(),
    files: Joi.array().items(
      Joi.object({
        filename: Joi.string().required(),
        content_base64: Joi.string().required(),
        mime_type: Joi.string().required(),
      })
    ).default([]),
    use_sandbox: Joi.boolean(),
  }),
});

solveRouter.post(
  '/',
  validateSolve,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      logRequest(req.body).catch((err) => console.error('[LOG ERROR]', err));
      const result = await solveService.solve(req.body as SolveRequestBody);
      res.json(result);
    } catch (err) {
      next(err);
    }
  }
);
