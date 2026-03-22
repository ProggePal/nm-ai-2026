import express from 'express';
import { solveRouter } from './routes/solveRouter.js';
import { dashboardRouter } from './routes/dashboardRouter.js';
import { errorHandler } from './middleware/errorHandler.js';
import { authMiddleware } from './middleware/authMiddleware.js';
import { NotFoundError } from './errors.js';

export const app = express();

app.set('trust proxy', true);
app.use(express.json());

// Dashboard routes — no auth required
app.use(dashboardRouter);

// Auth for API routes
app.use(authMiddleware);

app.use('/solve', solveRouter);

app.get('/', (_req, res) => {
  res.json({ message: 'Hello AI-world' });
});

app.use((_req, _res, next) => next(new NotFoundError()));

app.use(errorHandler);
