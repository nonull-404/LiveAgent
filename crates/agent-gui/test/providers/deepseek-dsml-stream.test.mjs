import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createTsModuleLoader } from "../helpers/load-ts-module.mjs";

const rootDir = path.resolve(fileURLToPath(new URL("../..", import.meta.url)));
const proxyModulePath = path.join(rootDir, "src/lib/providers/proxy.ts");
const powerActivityModulePath = path.join(rootDir, "src/lib/system/powerActivity.ts");

const loader = createTsModuleLoader();
const { wrapDeepSeekDsmlToolCallStream } = loader.loadModule(
  "src/lib/providers/deepSeekDsmlToolCallStream.ts",
);

function createUsage() {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function createAssistant(text, stopReason = "stop") {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    api: "anthropic-messages",
    provider: "anthropic",
    model: "deepseek-chat",
    usage: createUsage(),
    stopReason,
    timestamp: Date.now(),
  };
}

function createAssistantWithContent(content, stopReason = "toolUse") {
  return {
    role: "assistant",
    content,
    api: "anthropic-messages",
    provider: "anthropic",
    model: "deepseek-chat",
    usage: createUsage(),
    stopReason,
    timestamp: Date.now(),
  };
}

function createToolCall(id, name, args = {}) {
  return {
    type: "toolCall",
    id,
    name,
    arguments: args,
  };
}

function createToolResult(toolCall, text = "ok") {
  return {
    role: "toolResult",
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    content: [{ type: "text", text }],
    isError: false,
    timestamp: Date.now(),
  };
}

function createSourceStream(deltas, stopReason = "stop") {
  const text = deltas.join("");
  const assistant = createAssistant(text, stopReason);
  const partial = {
    ...assistant,
    content: [{ type: "text", text: "" }],
  };
  const events = [
    { type: "start", partial: { ...assistant, content: [] } },
    { type: "text_start", contentIndex: 0, partial },
  ];

  for (const delta of deltas) {
    partial.content[0].text += delta;
    events.push({
      type: "text_delta",
      contentIndex: 0,
      delta,
      partial: { ...partial, content: [{ ...partial.content[0] }] },
    });
  }

  events.push({
    type: "text_end",
    contentIndex: 0,
    content: text,
    partial: assistant,
  });
  events.push({ type: "done", reason: stopReason, message: assistant });

  return {
    async *[Symbol.asyncIterator]() {
      for (const event of events) {
        yield event;
      }
    },
    async result() {
      return assistant;
    },
  };
}

const dsml = "\uFF5C\uFF5CDSML\uFF5C\uFF5C";

test("DeepSeek DSML stream wrapper converts split builtin_web_search markup into tool calls", async () => {
  const deltas = [
    "prefix ",
    "<",
    dsml,
    "tool",
    "_calls>\n",
    "<",
    dsml,
    'invoke name="builtin_web_search">\n',
    "<",
    dsml,
    'parameter name="additionalContext" string="true">',
    "企",
    "查",
    "查 funding rounds",
    "</",
    dsml,
    "parameter>\n",
    "</",
    dsml,
    "invoke>\n",
    "<",
    dsml,
    'invoke name="builtin_web_search">\n',
    "<",
    dsml,
    'parameter name="additionalContext" string="true">',
    "DeepSeek Anthropic web search DSML",
    "</",
    dsml,
    "parameter>\n",
    "</",
    dsml,
    "invoke>\n",
    "</",
    dsml,
    "tool_calls>",
    " suffix",
  ];

  const wrapped = wrapDeepSeekDsmlToolCallStream(createSourceStream(deltas));
  const events = [];
  for await (const event of wrapped) {
    events.push(event);
  }

  const text = events
    .filter((event) => event.type === "text_delta")
    .map((event) => event.delta)
    .join("");
  assert.equal(text, "prefix  suffix");

  const toolCallEvents = events.filter((event) => event.type === "toolcall_end");
  assert.equal(toolCallEvents.length, 2);
  assert.deepEqual(
    toolCallEvents.map((event) => event.toolCall.name),
    ["builtin_web_search", "builtin_web_search"],
  );
  assert.deepEqual(toolCallEvents[0].toolCall.arguments, {
    additionalContext: "企查查 funding rounds",
  });

  const final = await wrapped.result();
  assert.equal(final.stopReason, "toolUse");
  assert.deepEqual(
    final.content.map((block) => block.type),
    ["text", "toolCall", "toolCall", "text"],
  );
  assert.equal(
    final.content
      .filter((block) => block.type === "text")
      .map((block) => block.text)
      .join(""),
    "prefix  suffix",
  );
});

test("DeepSeek DSML stream wrapper resolves transformed result without iteration", async () => {
  const wrapped = wrapDeepSeekDsmlToolCallStream(
    createSourceStream([
      `<${dsml}tool_calls>`,
      `<${dsml}invoke name="builtin_web_search">`,
      `<${dsml}parameter name="additionalContext" string="true">latest docs</${dsml}parameter>`,
      `</${dsml}invoke>`,
      `</${dsml}tool_calls>`,
    ]),
  );

  const final = await wrapped.result();
  assert.equal(final.stopReason, "toolUse");
  assert.equal(final.content.length, 1);
  assert.equal(final.content[0].type, "toolCall");
  assert.equal(final.content[0].name, "builtin_web_search");
});

test("DeepSeek DSML stream wrapper settles result when source closes without terminal event", async () => {
  const assistant = createAssistant("partial answer");
  const partial = {
    ...assistant,
    content: [{ type: "text", text: "" }],
  };
  const wrapped = wrapDeepSeekDsmlToolCallStream({
    async *[Symbol.asyncIterator]() {
      yield { type: "start", partial: { ...assistant, content: [] } };
      yield { type: "text_start", contentIndex: 0, partial };
      partial.content[0].text = "partial answer";
      yield {
        type: "text_delta",
        contentIndex: 0,
        delta: "partial answer",
        partial,
      };
      yield {
        type: "text_end",
        contentIndex: 0,
        content: "partial answer",
        partial: assistant,
      };
    },
    async result() {
      return assistant;
    },
  });

  const final = await wrapped.result();
  assert.equal(final.stopReason, "stop");
  assert.deepEqual(final.content, [{ type: "text", text: "partial answer" }]);
});

test("DeepSeek DSML stream wrapper recovers Anthropic streams missing message_stop after content", async () => {
  const assistant = createAssistant("partial answer");
  const partial = {
    ...assistant,
    content: [{ type: "text", text: "" }],
  };
  const wrapped = wrapDeepSeekDsmlToolCallStream({
    async *[Symbol.asyncIterator]() {
      yield { type: "start", partial: { ...assistant, content: [] } };
      yield { type: "text_start", contentIndex: 0, partial };
      partial.content[0].text = "partial ";
      yield {
        type: "text_delta",
        contentIndex: 0,
        delta: "partial ",
        partial: { ...partial, content: [{ ...partial.content[0] }] },
      };
      partial.content[0].text = "partial answer";
      yield {
        type: "text_delta",
        contentIndex: 0,
        delta: "answer",
        partial: { ...partial, content: [{ ...partial.content[0] }] },
      };
      throw new Error("Anthropic stream ended before message_stop");
    },
    async result() {
      return assistant;
    },
  });

  const events = [];
  for await (const event of wrapped) {
    events.push(event);
  }

  assert.equal(events.at(-1)?.type, "done");
  assert.equal(events.some((event) => event.type === "error"), false);
  assert.equal(
    events
      .filter((event) => event.type === "text_delta")
      .map((event) => event.delta)
      .join(""),
    "partial answer",
  );

  const final = await wrapped.result();
  assert.equal(final.stopReason, "stop");
  assert.deepEqual(final.content, [{ type: "text", text: "partial answer" }]);
});

test("DeepSeek DSML stream wrapper keeps empty premature message_stop streams as errors", async () => {
  const assistant = createAssistant("");
  const wrapped = wrapDeepSeekDsmlToolCallStream({
    async *[Symbol.asyncIterator]() {
      yield { type: "start", partial: { ...assistant, content: [] } };
      throw new Error("Anthropic stream ended before message_stop");
    },
    async result() {
      return assistant;
    },
  });

  const events = [];
  for await (const event of wrapped) {
    events.push(event);
  }

  assert.equal(events.at(-1)?.type, "error");
  const final = await wrapped.result();
  assert.equal(final.stopReason, "error");
  assert.match(final.errorMessage, /Anthropic stream ended before message_stop/);
});

test("streamAssistantMessage replies to recovered DeepSeek DSML tool calls before continuing", async () => {
  const streamQueue = [
    createSourceStream([
      "Searching ",
      `<${dsml}tool_calls>`,
      `<${dsml}invoke name="builtin_web_search">`,
      `<${dsml}parameter name="additionalContext" string="true">latest DeepSeek DSML fix</${dsml}parameter>`,
      `</${dsml}invoke>`,
      `</${dsml}tool_calls>`,
    ]),
    createSourceStream(["final answer with recovered context"]),
  ];
  const capturedContexts = [];
  const localLoader = createTsModuleLoader({
    mocks: {
      "@earendil-works/pi-ai": {
        getModel() {
          return undefined;
        },
      },
      "@earendil-works/pi-ai/anthropic": {
        streamAnthropic(_model, context) {
          capturedContexts.push(context);
          const stream = streamQueue.shift();
          if (!stream) throw new Error("No mocked Anthropic stream queued");
          return stream;
        },
      },
      [proxyModulePath]: {
        async prepareProxyRequest(_providerId, baseUrl) {
          return { baseUrl, headers: { "x-liveagent-test": "1" } };
        },
      },
      [powerActivityModulePath]: {
        async withPowerActivity(_scope, _reason, run) {
          return run();
        },
      },
    },
  });
  const providers = localLoader.loadModule("src/lib/providers/llm.ts");
  const textDeltas = [];

  const final = await providers.streamAssistantMessage({
    providerId: "claude_code",
    model: "deepseek-chat",
    runtime: {
      baseUrl: "https://api.deepseek.com/anthropic",
      apiKey: "test-key",
      requestFormat: "anthropic-messages",
      nativeWebSearchEnabled: true,
    },
    context: {
      messages: [{ role: "user", content: "Search latest DeepSeek DSML fix", timestamp: 1 }],
    },
    nativeWebSearch: true,
    onTextDelta(delta) {
      textDeltas.push(delta);
    },
  });

  assert.equal(final.stopReason, "stop");
  assert.deepEqual(
    final.content.map((block) => block.type),
    ["text"],
  );
  assert.equal(final.content[0].text, "final answer with recovered context");
  assert.equal(textDeltas.join(""), "Searching final answer with recovered context");
  assert.equal(capturedContexts.length, 2);

  const secondMessages = capturedContexts[1].messages;
  assert.equal(secondMessages.at(-1).role, "assistant");
  assert.deepEqual(
    secondMessages.at(-1).content.map((block) => block.type),
    ["text"],
  );
  assert.equal(
    secondMessages.some((message) =>
      message.content?.some?.((block) => block.type === "toolCall" || block.type === "toolResult"),
    ),
    false,
  );
});

test("streamAssistantMessage normalizes recovered DeepSeek DSML tool calls from history", async () => {
  const pairedSearch = createToolCall("dsml-tool-call-paired", "builtin_web_search", {
    additionalContext: "already paired",
  });
  const missingSearch = createToolCall("dsml-tool-call-missing-search", "builtin_web_search", {
    additionalContext: "DeepSeek missing search result",
  });
  const missingLocalTool = createToolCall("dsml-tool-call-missing-read", "Read", {
    path: "README.md",
  });
  const incompleteAssistant = createAssistantWithContent(
    [{ type: "text", text: "Searching" }, pairedSearch, missingSearch, missingLocalTool],
    "toolUse",
  );
  const capturedContexts = [];
  const localLoader = createTsModuleLoader({
    mocks: {
      "@earendil-works/pi-ai": {
        getModel() {
          return undefined;
        },
      },
      "@earendil-works/pi-ai/anthropic": {
        streamAnthropic(_model, context) {
          capturedContexts.push(context);
          return createSourceStream(["answer after repaired history"]);
        },
      },
      [proxyModulePath]: {
        async prepareProxyRequest(_providerId, baseUrl) {
          return { baseUrl, headers: { "x-liveagent-test": "1" } };
        },
      },
      [powerActivityModulePath]: {
        async withPowerActivity(_scope, _reason, run) {
          return run();
        },
      },
    },
  });
  const providers = localLoader.loadModule("src/lib/providers/llm.ts");

  await providers.streamAssistantMessage({
    providerId: "claude_code",
    model: "deepseek-chat",
    runtime: {
      baseUrl: "https://api.deepseek.com/anthropic",
      apiKey: "test-key",
      requestFormat: "anthropic-messages",
    },
    context: {
      messages: [
        { role: "user", content: "previous search", timestamp: 1 },
        incompleteAssistant,
        createToolResult(pairedSearch, "already done"),
        { role: "user", content: "continue", timestamp: 4 },
      ],
    },
    onTextDelta() {},
  });

  assert.equal(capturedContexts.length, 1);
  const messages = capturedContexts[0].messages;
  const assistantIndex = messages.findIndex(
    (message) =>
      message.role === "assistant" &&
      message.content.some((block) => block.type === "text" && block.text === "Searching") &&
      message.content.some(
        (block) => block.type === "text" && block.text.includes(missingLocalTool.id),
      ),
  );
  assert.ok(assistantIndex >= 0);
  assert.equal(
    messages.some((message) => message.role === "toolResult"),
    false,
  );
  assert.equal(
    messages.some(
      (message) =>
        message.role === "assistant" &&
        message.content.some((block) => block.type === "toolCall"),
    ),
    false,
  );
  assert.equal(messages[assistantIndex + 1].role, "user");
  assert.equal(
    messages[assistantIndex + 1].content.some(
      (block) =>
        block.type === "text" &&
        block.text.includes(missingLocalTool.id) &&
        block.text.includes("is_error: true"),
    ),
    true,
  );
});

test("completeAssistantMessage normalizes recovered DeepSeek DSML tool calls from history", async () => {
  const missingSearch = createToolCall("dsml-tool-call-complete-search", "builtin_web_search", {
    additionalContext: "DeepSeek complete missing result",
  });
  const incompleteAssistant = createAssistantWithContent([missingSearch], "toolUse");
  const capturedContexts = [];
  const localLoader = createTsModuleLoader({
    mocks: {
      "@earendil-works/pi-ai": {
        getModel() {
          return undefined;
        },
      },
      "@earendil-works/pi-ai/anthropic": {
        streamAnthropic(_model, context) {
          capturedContexts.push(context);
          return createSourceStream(["completed answer"]);
        },
      },
      [proxyModulePath]: {
        async prepareProxyRequest(_providerId, baseUrl) {
          return { baseUrl, headers: { "x-liveagent-test": "1" } };
        },
      },
      [powerActivityModulePath]: {
        async withPowerActivity(_scope, _reason, run) {
          return run();
        },
      },
    },
  });
  const providers = localLoader.loadModule("src/lib/providers/llm.ts");

  await providers.completeAssistantMessage({
    providerId: "claude_code",
    model: "deepseek-chat",
    runtime: {
      baseUrl: "https://api.deepseek.com/anthropic",
      apiKey: "test-key",
      requestFormat: "anthropic-messages",
    },
    context: {
      messages: [
        { role: "user", content: "previous search", timestamp: 1 },
        incompleteAssistant,
        { role: "user", content: "finish", timestamp: 3 },
      ],
    },
  });

  assert.equal(capturedContexts.length, 1);
  const messages = capturedContexts[0].messages;
  const assistantIndex = messages.findIndex((message) => message === incompleteAssistant);
  assert.equal(assistantIndex, -1);
  assert.equal(
    messages.some(
      (message) =>
        message.role === "toolResult" && message.toolCallId === missingSearch.id,
    ),
    false,
  );
  assert.equal(
    messages.some(
      (message) =>
        message.role === "assistant" &&
        message.content.some(
          (block) => block.type === "toolCall" && block.id === missingSearch.id,
        ),
    ),
    false,
  );
  assert.deepEqual(
    messages.map((message) => message.role),
    ["user", "user"],
  );
});
