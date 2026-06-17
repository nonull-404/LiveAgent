import type {
  AssistantMessage,
  AssistantMessageEvent,
  AssistantMessageEventStream,
  ToolCall,
} from "@earendil-works/pi-ai";
import { parseDsmlToolCallMarkup } from "../chat/runner/seedToolCalls";

const DSML_TAG_PREFIX = String.raw`(?:\uFF5C{2}|\|{2})\s*DSML\s*(?:\uFF5C{2}|\|{2})`;
const DSML_TOOL_CALLS_OPEN_PATTERN = new RegExp(
  String.raw`<\s*${DSML_TAG_PREFIX}\s*tool_calls\s*>`,
  "i",
);
const DSML_TOOL_CALLS_CLOSE_PATTERN = new RegExp(
  String.raw`<\/\s*${DSML_TAG_PREFIX}\s*tool_calls\s*>`,
  "i",
);
const DSML_OPEN_HOLD_LIMIT = 96;
const DSML_SWALLOW_BUFFER_LIMIT = 64 * 1024;

type IndexedAssistantEvent = Extract<AssistantMessageEvent, { contentIndex: number }>;

function cloneBlock<T>(block: T): T {
  if (!block || typeof block !== "object") return block;
  return { ...(block as Record<string, unknown>) } as T;
}

function snapshotAssistant(message: AssistantMessage): AssistantMessage {
  return {
    ...message,
    content: message.content.map(cloneBlock),
  };
}

function isTerminalEvent(event: AssistantMessageEvent) {
  return event.type === "done" || event.type === "error";
}

function terminalMessage(event: AssistantMessageEvent) {
  if (event.type === "done") return event.message;
  if (event.type === "error") return event.error;
  return null;
}

function createFallbackAssistant(message?: AssistantMessage): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: message?.api ?? "anthropic-messages",
    provider: message?.provider ?? "anthropic",
    model: message?.model ?? "unknown",
    usage: message?.usage ?? {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: message?.stopReason ?? "stop",
    timestamp: message?.timestamp ?? Date.now(),
  };
}

function findPattern(pattern: RegExp, value: string) {
  const match = pattern.exec(value);
  if (!match || match.index === undefined) return null;
  return {
    index: match.index,
    text: match[0],
  };
}

function findPotentialDsmlOpenStart(value: string) {
  const index = value.lastIndexOf("<");
  if (index < 0) return -1;
  return value.length - index <= DSML_OPEN_HOLD_LIMIT ? index : -1;
}

function normalizeDoneReason(stopReason: AssistantMessage["stopReason"]) {
  return stopReason === "toolUse" || stopReason === "length" ? stopReason : "stop";
}

function readErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function isRecoverableAnthropicStreamEndError(error: unknown) {
  const message = readErrorMessage(error);
  return (
    message.includes("Anthropic stream ended before message_stop") ||
    message.includes('before receiving "message_stop"')
  );
}

export function wrapDeepSeekDsmlToolCallStream(
  source: AssistantMessageEventStream,
): AssistantMessageEventStream {
  const queue: AssistantMessageEvent[] = [];
  const waiting: Array<(result: IteratorResult<AssistantMessageEvent>) => void> = [];
  let closed = false;
  let finalResolved = false;
  let resolveFinal!: (message: AssistantMessage) => void;
  const finalResult = new Promise<AssistantMessage>((resolve) => {
    resolveFinal = resolve;
  });

  let output: AssistantMessage | null = null;
  let extractedToolCalls = false;
  let activeTextSourceIndex: number | null = null;
  let activeTextOutputIndex: number | null = null;
  let textBuffer = "";
  let dsmlBuffer = "";
  let inDsml = false;
  let activeDsmlOpenTag = "";
  const sourceToOutputIndex = new Map<number, number>();

  const ensureOutput = (partial?: AssistantMessage) => {
    if (!output) {
      output = createFallbackAssistant(partial);
      return output;
    }
    if (partial) {
      output = {
        ...partial,
        content: output.content,
        stopReason:
          extractedToolCalls && partial.stopReason === "stop" ? "toolUse" : partial.stopReason,
      };
    }
    return output;
  };

  const buildPartial = (partial?: AssistantMessage) => snapshotAssistant(ensureOutput(partial));

  const buildFinalMessage = (sourceMessage: AssistantMessage) => {
    if (!output) {
      return sourceMessage;
    }
    return {
      ...sourceMessage,
      content: output.content.map(cloneBlock),
      stopReason:
        extractedToolCalls && sourceMessage.stopReason === "stop"
          ? "toolUse"
          : sourceMessage.stopReason,
    } satisfies AssistantMessage;
  };

  const settleFinal = (message: AssistantMessage) => {
    if (finalResolved) return;
    finalResolved = true;
    resolveFinal(message);
  };

  const notifyDone = () => {
    while (waiting.length > 0) {
      waiting.shift()?.({ value: undefined, done: true });
    }
  };

  const enqueue = (event: AssistantMessageEvent) => {
    if (closed) return;
    const terminal = isTerminalEvent(event);
    const message = terminalMessage(event);
    if (message) {
      settleFinal(message);
    }

    const waiter = waiting.shift();
    if (waiter) {
      waiter({ value: event, done: false });
    } else {
      queue.push(event);
    }

    if (terminal) {
      closed = true;
      if (queue.length === 0) notifyDone();
    }
  };

  const close = () => {
    if (closed) return;
    closed = true;
    notifyDone();
  };

  const emitError = (error: unknown) => {
    const errorMessage = readErrorMessage(error);
    const message = {
      ...ensureOutput(),
      stopReason: "error",
      errorMessage,
    } satisfies AssistantMessage;
    enqueue({ type: "error", reason: "error", error: snapshotAssistant(message) });
  };

  const hasRecoverableOutput = () =>
    Boolean(output?.content.length) || Boolean(textBuffer || dsmlBuffer || activeDsmlOpenTag);

  const ensureTextBlock = (sourceIndex: number, partial?: AssistantMessage) => {
    const nextOutput = ensureOutput(partial);
    if (activeTextSourceIndex !== sourceIndex) {
      activeTextSourceIndex = sourceIndex;
      activeTextOutputIndex = null;
    }
    if (activeTextOutputIndex !== null) return activeTextOutputIndex;

    const sourceBlock = partial?.content[sourceIndex];
    const textBlock = {
      type: "text",
      text: "",
      ...(sourceBlock?.type === "text" && sourceBlock.textSignature
        ? { textSignature: sourceBlock.textSignature }
        : {}),
    } as const;
    const outputIndex = nextOutput.content.length;
    nextOutput.content.push(textBlock);
    activeTextOutputIndex = outputIndex;
    enqueue({
      type: "text_start",
      contentIndex: outputIndex,
      partial: buildPartial(partial),
    });
    return outputIndex;
  };

  const emitTextDelta = (sourceIndex: number, delta: string, partial?: AssistantMessage) => {
    if (!delta) return;
    const outputIndex = ensureTextBlock(sourceIndex, partial);
    const block = ensureOutput(partial).content[outputIndex];
    if (block?.type !== "text") return;
    block.text += delta;
    enqueue({
      type: "text_delta",
      contentIndex: outputIndex,
      delta,
      partial: buildPartial(partial),
    });
  };

  const endActiveTextBlock = (partial?: AssistantMessage) => {
    if (activeTextOutputIndex === null) return;
    const outputIndex = activeTextOutputIndex;
    const block = ensureOutput(partial).content[outputIndex];
    if (block?.type === "text") {
      enqueue({
        type: "text_end",
        contentIndex: outputIndex,
        content: block.text,
        partial: buildPartial(partial),
      });
    }
    activeTextOutputIndex = null;
  };

  const emitToolCall = (toolCall: ToolCall, partial?: AssistantMessage) => {
    const nextOutput = ensureOutput(partial);
    const outputIndex = nextOutput.content.length;
    const normalizedToolCall = cloneBlock(toolCall);
    nextOutput.content.push(normalizedToolCall);
    const delta = JSON.stringify(normalizedToolCall.arguments ?? {});
    enqueue({
      type: "toolcall_start",
      contentIndex: outputIndex,
      partial: buildPartial(partial),
    });
    enqueue({
      type: "toolcall_delta",
      contentIndex: outputIndex,
      delta,
      partial: buildPartial(partial),
    });
    enqueue({
      type: "toolcall_end",
      contentIndex: outputIndex,
      toolCall: normalizedToolCall,
      partial: buildPartial(partial),
    });
  };

  const drainDsmlBuffer = (sourceIndex: number, partial?: AssistantMessage) => {
    const closeTag = findPattern(DSML_TOOL_CALLS_CLOSE_PATTERN, dsmlBuffer);
    if (!closeTag) {
      if (dsmlBuffer.length > DSML_SWALLOW_BUFFER_LIMIT) {
        emitTextDelta(sourceIndex, `${activeDsmlOpenTag}${dsmlBuffer}`, partial);
        dsmlBuffer = "";
        activeDsmlOpenTag = "";
        inDsml = false;
      }
      return;
    }

    const blockContent = dsmlBuffer.slice(0, closeTag.index);
    const remainder = dsmlBuffer.slice(closeTag.index + closeTag.text.length);
    const markup = `${activeDsmlOpenTag}${blockContent}${closeTag.text}`;
    const toolCalls = parseDsmlToolCallMarkup(markup);

    if (toolCalls.length === 0) {
      emitTextDelta(sourceIndex, markup, partial);
    } else {
      extractedToolCalls = true;
      for (const toolCall of toolCalls) {
        emitToolCall(toolCall, partial);
      }
    }

    dsmlBuffer = "";
    activeDsmlOpenTag = "";
    inDsml = false;
    textBuffer = remainder;
  };

  const drainTextBuffer = (sourceIndex: number, partial?: AssistantMessage) => {
    while (textBuffer.length > 0) {
      const openTag = findPattern(DSML_TOOL_CALLS_OPEN_PATTERN, textBuffer);
      if (!openTag) {
        const holdIndex = findPotentialDsmlOpenStart(textBuffer);
        if (holdIndex >= 0) {
          emitTextDelta(sourceIndex, textBuffer.slice(0, holdIndex), partial);
          textBuffer = textBuffer.slice(holdIndex);
          return;
        }
        emitTextDelta(sourceIndex, textBuffer, partial);
        textBuffer = "";
        return;
      }

      emitTextDelta(sourceIndex, textBuffer.slice(0, openTag.index), partial);
      endActiveTextBlock(partial);
      activeDsmlOpenTag = openTag.text;
      dsmlBuffer = textBuffer.slice(openTag.index + openTag.text.length);
      textBuffer = "";
      inDsml = true;
      drainDsmlBuffer(sourceIndex, partial);
      if (inDsml) return;
    }
  };

  const flushTextState = (sourceIndex: number, partial?: AssistantMessage) => {
    if (inDsml) {
      emitTextDelta(sourceIndex, `${activeDsmlOpenTag}${dsmlBuffer}`, partial);
      dsmlBuffer = "";
      activeDsmlOpenTag = "";
      inDsml = false;
    }
    if (textBuffer) {
      emitTextDelta(sourceIndex, textBuffer, partial);
      textBuffer = "";
    }
    endActiveTextBlock(partial);
  };

  const mirrorIndexedEvent = (event: IndexedAssistantEvent) => {
    const nextOutput = ensureOutput(event.partial);
    const sourceBlock = event.partial.content[event.contentIndex];
    let outputIndex = sourceToOutputIndex.get(event.contentIndex);
    if (outputIndex === undefined) {
      outputIndex = nextOutput.content.length;
      sourceToOutputIndex.set(event.contentIndex, outputIndex);
    }
    nextOutput.content[outputIndex] = cloneBlock(sourceBlock);
    const partial = buildPartial(event.partial);
    if (event.type === "toolcall_end") {
      const toolCall = nextOutput.content[outputIndex] as ToolCall;
      enqueue({ ...event, contentIndex: outputIndex, toolCall, partial });
      return;
    }
    enqueue({ ...event, contentIndex: outputIndex, partial } as AssistantMessageEvent);
  };

  void (async () => {
    try {
      for await (const event of source) {
        switch (event.type) {
          case "start": {
            output = createFallbackAssistant(event.partial);
            enqueue({ type: "start", partial: buildPartial(event.partial) });
            break;
          }
          case "text_start": {
            activeTextSourceIndex = event.contentIndex;
            activeTextOutputIndex = null;
            textBuffer = "";
            dsmlBuffer = "";
            inDsml = false;
            activeDsmlOpenTag = "";
            ensureOutput(event.partial);
            break;
          }
          case "text_delta": {
            activeTextSourceIndex = event.contentIndex;
            if (inDsml) {
              dsmlBuffer += event.delta;
              drainDsmlBuffer(event.contentIndex, event.partial);
            } else {
              textBuffer += event.delta;
              drainTextBuffer(event.contentIndex, event.partial);
            }
            break;
          }
          case "text_end": {
            flushTextState(event.contentIndex, event.partial);
            activeTextSourceIndex = null;
            break;
          }
          case "done": {
            if (activeTextSourceIndex !== null) {
              flushTextState(activeTextSourceIndex, event.message);
              activeTextSourceIndex = null;
            }
            const message = buildFinalMessage(event.message);
            enqueue({
              type: "done",
              reason: normalizeDoneReason(message.stopReason),
              message,
            });
            break;
          }
          case "error": {
            if (activeTextSourceIndex !== null) {
              flushTextState(activeTextSourceIndex, event.error);
              activeTextSourceIndex = null;
            }
            enqueue({
              type: "error",
              reason: event.reason,
              error: buildFinalMessage(event.error),
            });
            break;
          }
          case "thinking_start":
          case "thinking_delta":
          case "thinking_end":
          case "toolcall_start":
          case "toolcall_delta":
          case "toolcall_end": {
            mirrorIndexedEvent(event);
            break;
          }
        }
      }
      if (!closed) {
        settleFinal(snapshotAssistant(ensureOutput()));
        close();
      }
    } catch (error) {
      if (isRecoverableAnthropicStreamEndError(error) && hasRecoverableOutput()) {
        const sourceIndex = activeTextSourceIndex ?? 0;
        flushTextState(sourceIndex);
        activeTextSourceIndex = null;
        const message = buildFinalMessage(snapshotAssistant(ensureOutput()));
        enqueue({
          type: "done",
          reason: normalizeDoneReason(message.stopReason),
          message,
        });
        return;
      }
      emitError(error);
    }
  })();

  return {
    async *[Symbol.asyncIterator]() {
      while (true) {
        if (queue.length > 0) {
          const event = queue.shift();
          if (event) {
            yield event;
          }
          if (closed && queue.length === 0) notifyDone();
          continue;
        }
        if (closed) return;
        const result = await new Promise<IteratorResult<AssistantMessageEvent>>((resolve) =>
          waiting.push(resolve),
        );
        if (result.done) return;
        yield result.value;
      }
    },
    result() {
      return finalResult;
    },
  } as unknown as AssistantMessageEventStream;
}
