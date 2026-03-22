import jwt from 'jsonwebtoken';

const secret = process.env.JWT_SECRET;
if (!secret) {
  console.error('JWT_SECRET environment variable is required');
  process.exit(1);
}

const sub = process.argv[2];
if (!sub) {
  console.error('Usage: bun run scripts/generate-token.ts <third-party-name> [--exp <duration>]');
  process.exit(1);
}

const expIndex = process.argv.indexOf('--exp');
const expValue = expIndex !== -1 ? process.argv[expIndex + 1] : undefined;

const payload = { sub };
const options: jwt.SignOptions = { algorithm: 'HS256' };
if (expValue) {
  options.expiresIn = expValue as jwt.SignOptions['expiresIn'];
}

const token = jwt.sign(payload, secret, options);
console.log(token);
