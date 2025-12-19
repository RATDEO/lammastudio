import { NextRequest } from 'next/server';
import { mergeToolCallArguments } from '@/lib/tool-parsing';

const LITELLM_URL = process.env.LITELLM_URL || process.env.NEXT_PUBLIC_LITELLM_URL || 'http://localhost:4000';
const LITELLM_API_KEY = process.env.LITELLM_MASTER_KEY || process.env.LITELLM_API_KEY || process.env.API_KEY || 'sk-master';

interface StreamEvent {
  type: 'text' | 'tool_calls' | 'done' | 'error';
  content?: string;
  tool_calls?: ToolCall[];
  error?: string;
}

interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface OpenAIDelta {
  role?: string;
  content?: string | null;
  reasoning?: string | null;
  reasoning_content?: string | null;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: string;
    function?: { name?: string; arguments?: string };
  }>;
}

type OpenAIToolCallDelta = NonNullable<OpenAIDelta['tool_calls']>[number];

interface OpenAIChunk {
  id: string;
  choices: Array<{
    index: number;
    delta: OpenAIDelta;
    finish_reason: string | null;
  }>;
}

const mergeStreamingText = (prevFull: string, incoming: string): { nextFull: string; emit: string } => {
  const prev = prevFull || '';
  const next = incoming || '';
  if (!next) return { nextFull: prev, emit: '' };
  if (!prev) return { nextFull: next, emit: next };

  if (next === prev) return { nextFull: prev, emit: '' };
  if (next.startsWith(prev)) return { nextFull: next, emit: next.slice(prev.length) };
  if (prev.startsWith(next)) return { nextFull: prev, emit: '' };
  if (prev.endsWith(next)) return { nextFull: prev, emit: '' };

  const maxOverlap = Math.min(prev.length, next.length);
  for (let k = maxOverlap; k > 0; k--) {
    const prefix = next.slice(0, k);
    if (prev.endsWith(prefix)) {
      const suffix = next.slice(k);
      return { nextFull: prev + suffix, emit: suffix };
    }
  }

  return { nextFull: prev + next, emit: next };
};

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { messages, model, tools, ...rest } = body;

    console.log('[Chat API] Request:', {
      model,
      messageCount: messages?.length,
      toolCount: tools?.length,
    });

    if (!messages || !Array.isArray(messages)) {
      return new Response(JSON.stringify({ error: 'Messages required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Build an OpenAI-compatible request body (forwarding extra params through unchanged).
    const requestBody: Record<string, unknown> = {
      model: model || 'default',
      messages,
      ...rest,
      stream: true,
    };

    // Add tools if provided
    if (tools && tools.length > 0) {
      requestBody.tools = tools;
      requestBody.tool_choice = 'auto';
    }

    const incomingAuth = req.headers.get('authorization');
    const outgoingAuth = incomingAuth || (LITELLM_API_KEY ? `Bearer ${LITELLM_API_KEY}` : undefined);

    // Route all model calls through LiteLLM (provider translation + tool-call handling).
    const response = await fetch(`${LITELLM_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(outgoingAuth ? { Authorization: outgoingAuth } : {}),
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`LiteLLM error ${response.status}: ${errorText}`);
    }

    const encoder = new TextEncoder();
    const sendEvent = (event: StreamEvent): Uint8Array => {
      return encoder.encode(`data: ${JSON.stringify(event)}\n\n`);
    };

    // Track tool calls being assembled (streaming deltas can split id/name/args)
    const toolCallsInProgress: Map<number, ToolCall> = new Map();
    let toolsEmitted = false;
    let assistantContentFull = '';

    const upsertToolCallDelta = (tc: OpenAIToolCallDelta) => {
      const idx = tc.index;
      const existing =
        toolCallsInProgress.get(idx) ||
        ({
          id: tc.id || `call_${idx}`,
          type: 'function' as const,
          function: { name: tc.function?.name || '', arguments: tc.function?.arguments || '' },
        } satisfies ToolCall);

      if (tc.id && (existing.id.startsWith('call_') || !existing.id)) {
        existing.id = tc.id;
      }
      if (tc.function?.name) {
        existing.function.name = tc.function.name;
      }
      if (tc.function?.arguments) {
        existing.function.arguments = mergeToolCallArguments(
          existing.function.arguments,
          tc.function.arguments
        );
      }

      toolCallsInProgress.set(idx, existing);
    };

    const emitCompletedToolsIfAny = (controller: ReadableStreamDefaultController) => {
      if (toolsEmitted || toolCallsInProgress.size === 0) return;
      const completedTools = Array.from(toolCallsInProgress.entries())
        .sort(([a], [b]) => a - b)
        .map(([, v]) => v);
      controller.enqueue(sendEvent({ type: 'tool_calls', tool_calls: completedTools }));
      toolsEmitted = true;
    };

    const stream = new ReadableStream({
      async start(controller) {
        try {
          const reader = response.body?.getReader();
          if (!reader) throw new Error('No response body');

          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue;
              const data = line.slice(6).trim();
              if (!data) continue;
              if (data === '[DONE]') {
                continue;
              }

              try {
                const chunk: OpenAIChunk = JSON.parse(data);
                const delta = chunk.choices[0]?.delta;
                if (!delta) continue;

                // Handle regular content
                if (delta.content) {
                  const merged = mergeStreamingText(assistantContentFull, delta.content);
                  assistantContentFull = merged.nextFull;
                  if (merged.emit) controller.enqueue(sendEvent({ type: 'text', content: merged.emit }));
                }

                // Handle tool calls
                if (delta.tool_calls) {
                  for (const tc of delta.tool_calls) {
                    upsertToolCallDelta(tc);
                  }
                }

                // Check for finish
                if (chunk.choices[0]?.finish_reason) {
                  emitCompletedToolsIfAny(controller);
                }
              } catch {
                // Skip malformed JSON
              }
            }
          }

          emitCompletedToolsIfAny(controller);
          controller.enqueue(sendEvent({ type: 'done' }));
        } catch (error) {
          console.error('[Chat API] Stream error:', error);
          controller.enqueue(sendEvent({
            type: 'error',
            error: error instanceof Error ? error.message : String(error)
          }));
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('[Chat API] Error:', error);
    return new Response(JSON.stringify({ error: String(error) }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
