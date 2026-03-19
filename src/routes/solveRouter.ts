import { Router, type Request, type Response, type NextFunction } from 'express';
import { celebrate, Joi, Segments } from 'celebrate';
import { logRequest } from '../services/requestLogger.js';
import { solveService } from '../services/solveService.js';
import type { SolveRequestBody } from '../types.js';

export const solveRouter = Router();

const solveSchema = Joi.object({
  prompt: Joi.string().required(),
  files: Joi.array()
    .items(
      Joi.object({
        filename: Joi.string().required(),
        content: Joi.string().required(),
        mime_type: Joi.string().required(),
      })
    )
    .optional(),
  tripletex_credentials: Joi.object({
    base_url: Joi.string().uri().required(),
    session_token: Joi.string().required(),
  }).required(),
});

solveRouter.post(
  '/',
  celebrate({ [Segments.BODY]: solveSchema }),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      await logRequest(req.body);
      await solveService.solve(req.body as SolveRequestBody);
      res.json({ status: 'completed' });
    } catch (err) {
      next(err);
    }
  }
);
