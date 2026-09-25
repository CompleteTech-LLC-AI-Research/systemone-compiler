import { experimental_evaluate as evaluate } from 'ai';

if (!process.env.AI_GATEWAY_API_KEY) {
  console.error('AI_GATEWAY_API_KEY is missing from the local environment.');
  process.exit(1);
}

try {
  if (process.argv.includes('--models')) {
    const response = await fetch('https://ai-gateway.vercel.sh/v1/models', {
      headers: { Authorization: `Bearer ${process.env.AI_GATEWAY_API_KEY}` },
      signal: AbortSignal.timeout(30000),
    });
    if (!response.ok) {
      console.error(JSON.stringify({ ok: false, status: response.status }));
      process.exit(1);
    }
    const body = await response.json();
    console.log(JSON.stringify({ models: body.data.map((m: { id: string }) => m.id)
      .filter((id: string) => /jev|typesafe/.test(id)) }));
  } else {
    const model = 'typesafe-ai/jev';
    const result = await evaluate({
      model,
      state: 'The support agent issued a full refund to the customer.',
      questions: { refunded: { type: 'boolean', instructions: 'Was a refund issued?' } },
      maxRetries: 0,
      abortSignal: AbortSignal.timeout(90000),
    });
    console.log(JSON.stringify({ ok: true, requestedModel: model,
      answers: result.answers,
      usage: result.usage }));
  }
} catch (error) {
  // Do not print SDK errors: they can contain request headers or bodies.
  const e = error as { statusCode?: number };
  console.error(JSON.stringify({ ok: false, status: e.statusCode ?? null,
    message: 'Gateway request failed; no automatic retry or model fallback was attempted.' }));
  process.exitCode = 1;
}
