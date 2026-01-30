// CRITICAL
"use client";
import { useEffect, useRef, useCallback, useMemo, useState } from "react";
import { flushSync } from "react-dom";
import { useSearchParams, useRouter } from "next/navigation";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, lastAssistantMessageIsCompleteWithToolCalls } from "ai";
import { api } from "@/lib/api";
import { extractArtifacts } from "../artifacts/artifact-renderer";
import { ToolBelt } from "../input/tool-belt";
import { ChatSidePanel } from "./chat-side-panel";
import { ChatConversation } from "./chat-conversation";
import { ChatTopControls } from "./chat-top-controls";
import { ChatActionButtons } from "./chat-action-buttons";
import { ChatToolbeltDock } from "./chat-toolbelt-dock";
import { ChatModals } from "./chat-modals";
import { useChatSessions } from "../../hooks/use-chat-sessions";
import { useChatTools } from "../../hooks/use-chat-tools";
import { useChatUsage } from "../../hooks/use-chat-usage";
import { useChatDerived } from "../../hooks/use-chat-derived";
import { useChatTransport } from "../../hooks/use-chat-transport";
import { useRealtimeStatus } from "@/hooks/use-realtime-status";
import type { UIMessage } from "@ai-sdk/react";
import type { Artifact, StoredMessage, StoredToolCall } from "@/lib/types";
import { useContextManagement, type CompactionEvent } from "@/lib/services/context-management";
import { useMessageParsing } from "@/lib/services/message-parsing";
import { useAppStore } from "@/store";
import type { Attachment } from "../../types";
import { stripThinkingForModelContext, tryParseNestedJsonString } from "../../utils";
export function ChatPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const sessionFromUrl = searchParams.get("session");
  const newChatFromUrl = searchParams.get("new") === "1";
  // Local UI state (sourced from Zustand)
  const input = useAppStore((state) => state.input);
  const setInput = useAppStore((state) => state.setInput);
  const selectedModel = useAppStore((state) => state.selectedModel);
  const setSelectedModel = useAppStore((state) => state.setSelectedModel);
  const systemPrompt = useAppStore((state) => state.systemPrompt);
  const setSystemPrompt = useAppStore((state) => state.setSystemPrompt);
  const toolPanelOpen = useAppStore((state) => state.toolPanelOpen);
  const setToolPanelOpen = useAppStore((state) => state.setToolPanelOpen);
  const activePanel = useAppStore((state) => state.activePanel);
  const setActivePanel = useAppStore((state) => state.setActivePanel);
  const mcpEnabled = useAppStore((state) => state.mcpEnabled);
  const setMcpEnabled = useAppStore((state) => state.setMcpEnabled);
  const artifactsEnabled = useAppStore((state) => state.artifactsEnabled);
  const setArtifactsEnabled = useAppStore((state) => state.setArtifactsEnabled);
  const deepResearch = useAppStore((state) => state.deepResearch);
  const setDeepResearch = useAppStore((state) => state.setDeepResearch);
  const elapsedSeconds = useAppStore((state) => state.elapsedSeconds);
  const setElapsedSeconds = useAppStore((state) => state.setElapsedSeconds);
  const streamingStartTime = useAppStore((state) => state.streamingStartTime);
  const setStreamingStartTime = useAppStore((state) => state.setStreamingStartTime);
  const queuedContext = useAppStore((state) => state.queuedContext);
  const setQueuedContext = useAppStore((state) => state.setQueuedContext);
  const userScrolledUp = useAppStore((state) => state.userScrolledUp);
  const setUserScrolledUp = useAppStore((state) => state.setUserScrolledUp);
  // Modal state (Zustand)
  const settingsOpen = useAppStore((state) => state.chatSettingsOpen);
  const setSettingsOpen = useAppStore((state) => state.setChatSettingsOpen);
  const mcpSettingsOpen = useAppStore((state) => state.mcpSettingsOpen);
  const setMcpSettingsOpen = useAppStore((state) => state.setMcpSettingsOpen);
  const usageOpen = useAppStore((state) => state.usageDetailsOpen);
  const setUsageOpen = useAppStore((state) => state.setUsageDetailsOpen);
  const exportOpen = useAppStore((state) => state.exportOpen);
  const setExportOpen = useAppStore((state) => state.setExportOpen);
  const availableModels = useAppStore((state) => state.availableModels);
  const { status: realtimeStatus } = useRealtimeStatus();
  const setAvailableModels = useAppStore((state) => state.setAvailableModels);
  const sessionUsage = useAppStore((state) => state.sessionUsage);
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  // Sessions hook
  const {
    sessions,
    currentSessionId,
    currentSessionTitle,
    loadSessions,
    loadSession,
    startNewSession,
    createSession,
    setCurrentSessionId,
    setCurrentSessionTitle,
  } = useChatSessions();
  // Tools hook
  const {
    mcpServers,
    mcpTools,
    loadMCPServers,
    loadMCPTools,
    getToolDefinitions,
    executeTool,
    executingTools,
    toolResultsMap,
    addMcpServer,
    updateMcpServer,
    removeMcpServer,
  } = useChatTools({ mcpEnabled });
  // Usage hook
  const { refreshUsage } = useChatUsage();
  // Transport hook for persistence
  const { persistMessage, createSessionWithMessage, generateTitle, sessionIdRef } =
    useChatTransport({
      currentSessionId,
      setCurrentSessionId,
      setCurrentSessionTitle,
      selectedModel,
    });
  const {
    calculateStats,
    formatTokenCount,
    calculateMessageTokens,
    estimateTokens,
    config: contextConfig,
  } = useContextManagement();
  const { parseThinking } = useMessageParsing();
  const updateSessions = useAppStore((state) => state.updateSessions);
  const updateToolResultsMap = useAppStore((state) => state.updateToolResultsMap);
  const setMessageSources = useAppStore((state) => state.setMessageSources);
  const messageSourceMap = useAppStore((state) => state.messageSourceMap);
  // Track the last user input for title generation
  const lastUserInputRef = useRef<string>("");
  const deepResearchToolRef = useRef<string | null>(null);
  const autoArtifactSwitchRef = useRef(false);
  const [compactionHistory, setCompactionHistory] = useState<CompactionEvent[]>([]);
  const [hasPendingUserMessage, setHasPendingUserMessage] = useState(false);
  const [pendingUserText, setPendingUserText] = useState<string | null>(null);
  const [pendingStatus, setPendingStatus] = useState<string | null>(null);
  const [isRestoringSession, setIsRestoringSession] = useState(false);
  const [runtimeContextByModel, setRuntimeContextByModel] = useState<Record<string, number>>({});
  const [compacting, setCompacting] = useState(false);
  const [compactionError, setCompactionError] = useState<string | null>(null);
  const lastCompactionSignatureRef = useRef<string | null>(null);
  const lastSearchSourcesRef = useRef<Array<{ title: string; url: string }>>([]);
  const extractSourcesFromToolContent = (content: string) => {
    const sources: Array<{ title: string; url: string }> = [];
    const seen = new Set<string>();
    const addSource = (title: string, url: string) => {
      if (!url) return;
      if (seen.has(url)) return;
      seen.add(url);
      sources.push({ title: title || url, url });
    };
    const parsed = tryParseNestedJsonString(content) as Record<string, unknown> | null;
    if (parsed && typeof parsed === "object") {
      const candidate =
        (parsed as { results?: unknown }).results ??
        (parsed as { data?: unknown }).data ??
        (parsed as { sources?: unknown }).sources ??
        (parsed as { items?: unknown }).items;
      if (Array.isArray(candidate)) {
        for (const entry of candidate) {
          if (!entry || typeof entry !== "object") continue;
          const record = entry as Record<string, unknown>;
          const title = typeof record.title === "string" ? record.title : "";
          const url = typeof record.url === "string" ? record.url : "";
          addSource(title, url);
        }
      }
    }
    const pairRegex = new RegExp("Title:\\s*(.*)\\n.*?URL:\\s*(\\\\S+)", "gi");
    let match: RegExpExecArray | null;
    while ((match = pairRegex.exec(content)) !== null) {
      const title = (match[1] || "").trim();
      const url = (match[2] || "").trim();
      addSource(title, url);
    }
    const urlRegex = /https?:\/\/[^\s)"]+/g;
    const urlMatches = content.match(urlRegex) || [];
    for (const url of urlMatches) {
      addSource("", url);
    }
    return sources;
  };
  const mapStoredToolCalls = useCallback((toolCalls?: StoredToolCall[]) => {
    if (!toolCalls || toolCalls.length === 0) return [];
    return toolCalls.map((toolCall) => {
      const name = toolCall.function?.name || "tool";
      const args = toolCall.function?.arguments;
      const parsedArgs =
        typeof args === "string" ? tryParseNestedJsonString(args) ?? args : args ?? undefined;
      const hasResult = toolCall.result !== undefined && toolCall.result !== null;
      return {
        type: `tool-${name}`,
        toolCallId: toolCall.id,
        state: hasResult ? "result" : "call",
        input: parsedArgs,
        output: hasResult ? toolCall.result : undefined,
      };
    });
  }, []);
  const mapStoredMessages = useCallback((storedMessages: StoredMessage[]) => {
    return storedMessages.map((message) => {
      const parts: UIMessage["parts"] = [];
      // Build message parts including attachments
      if (message.content) {
        parts.push({ type: "text", text: message.content });
      }
      const toolParts = mapStoredToolCalls(message.tool_calls);
      for (const toolPart of toolParts) {
        parts.push(toolPart as UIMessage["parts"][number]);
      }
      const sources = (message as { sources?: Array<{ title: string; url: string }> }).sources;
      if (Array.isArray(sources) && sources.length > 0) {
        parts.push({
          type: "tool-sources_used",
          toolCallId: `sources_used:${message.id}`,
          state: "result",
        } as UIMessage["parts"][number]);
      }
      const inputTokens = message.prompt_tokens ?? undefined;
      const outputTokens = message.completion_tokens ?? undefined;
      const totalTokens =
        message.total_tokens ??
        (inputTokens != null || outputTokens != null
          ? (inputTokens ?? 0) + (outputTokens ?? 0)
          : undefined);
      return {
        id: message.id,
        role: message.role,
        parts,
        metadata: {
          model: message.model,
          usage:
            inputTokens != null || outputTokens != null || totalTokens != null
              ? {
                  inputTokens,
                  outputTokens,
                  totalTokens,
                }
              : undefined,
        },
      } satisfies UIMessage;
    });
  }, [mapStoredToolCalls]);
  const resolveToolDefinitions = useCallback(async () => {
    if (!mcpEnabled) return [];
    if (mcpTools.length > 0) {
      return getToolDefinitions?.(mcpTools) ?? [];
    }
    const loadedTools = await loadMCPTools();
    const tools = loadedTools.length > 0 ? loadedTools : mcpTools;
    return getToolDefinitions?.(tools) ?? [];
  }, [mcpEnabled, mcpTools, loadMCPTools, getToolDefinitions]);
  // Create transport for useChat (static; request-level body passed on send)
  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
      }),
    [],
  );
  // AI SDK useChat - the source of truth for messages
  const { messages, sendMessage, stop, status, error, setMessages, addToolOutput, clearError } = useChat({
    transport,
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
    onToolCall: async ({ toolCall }) => {
      const toolCallId = toolCall.toolCallId;
      const toolName = toolCall.toolName;
      const result = await executeTool({
        toolCallId,
        toolName,
        args: (toolCall as { input?: unknown }).input as Record<string, unknown>,
      });
      if (result.isError) {
        addToolOutput({
          tool: toolName as never,
          toolCallId,
          state: "output-error",
          errorText: result.content || "Tool execution failed",
        });
      } else {
        addToolOutput({
          tool: toolName as never,
          toolCallId,
          output: result.content as never,
        });
      }
    },
    onFinish: async ({ message }) => {
      setStreamingStartTime(null);
      setElapsedSeconds(0);
      const activeSessionId = sessionIdRef.current ?? currentSessionId;
      // Persist assistant message
      if (activeSessionId && message.role === "assistant") {
        const textContent = message.parts
          .filter((p): p is { type: "text"; text: string } => p.type === "text")
          .map((p) => p.text)
          .join("\n");

        // Extract citations like [1], [2] and surface used sources in Activity
        const citationMatches = [...textContent.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
        if (lastSearchSourcesRef.current.length > 0) {
          const unique = Array.from(new Set(citationMatches)).filter((n) => n > 0);
          const used = unique.length > 0
            ? (unique
                .map((n) => lastSearchSourcesRef.current[n - 1])
                .filter(Boolean) as Array<{ title: string; url: string }>)
            : lastSearchSourcesRef.current;
          if (used.length > 0) {
            const output = used
              .map((s) => `Title: ${s.title}
URL: ${s.url}`)
              .join("\n\n");
            const toolCallId = `sources_used:${Date.now()}`;
            updateToolResultsMap((prev) => new Map(prev).set(toolCallId, {
              tool_call_id: toolCallId,
              content: output,
              isError: false,
            }));
          }
          setMessageSources(message.id, lastSearchSourcesRef.current);
        }

        await persistMessage(activeSessionId, message);

        // Generate title if this is the first exchange
        if (
          (currentSessionTitle === "New Chat" || currentSessionTitle === "Chat") &&
          lastUserInputRef.current
        ) {
          await generateTitle(activeSessionId, lastUserInputRef.current, textContent);
        }
      }
    },
    onError: (err) => {
      console.error("Chat error:", err);
      setStreamingStartTime(null);
      setElapsedSeconds(0);
    },
  });
  const isLoading = status === "streaming" || status === "submitted";
  // Derived state from messages
  const activityMessages = isRestoringSession ? messages.slice(-10) : messages;
  const { thinkingActive, activityGroups } = useChatDerived({
    messages: activityMessages,
    isLoading,
    executingTools,
    toolResultsMap,
  });
  const activityCount = useMemo(() => {
    return activityGroups.reduce(
      (sum, group) => sum + group.toolItems.length + (group.thinkingContent ? 1 : 0),
      0,
    );
  }, [activityGroups]);
  const selectedModelMeta = useMemo(
    () => availableModels.find((model) => model.id === selectedModel),
    [availableModels, selectedModel],
  );
  const maxContext = selectedModelMeta?.runtimeContext ?? selectedModelMeta?.maxModelLen;
  const contextMessages = useMemo(() => {
    return messages
      .map((message) => {
        const textContent = message.parts
          .filter((part): part is { type: "text"; text: string } => part.type === "text")
          .map((part) => part.text)
          .join("\n");
        const cleanedText = stripThinkingForModelContext(textContent);
        const toolContent = message.parts
          .filter(
            (
              part,
            ): part is UIMessage["parts"][number] & {
              input?: unknown;
              output?: unknown;
              errorText?: string;
            } => {
              if (typeof part.type !== "string") return false;
              return part.type === "dynamic-tool" || part.type.startsWith("tool-");
            }
          )
          .map((part) => {
            const input = "input" in part && part.input != null ? JSON.stringify(part.input) : "";
            const output =
              "output" in part && part.output != null ? JSON.stringify(part.output) : "";
            const errorText = "errorText" in part && part.errorText ? part.errorText : "";
            return [input, output, errorText].filter(Boolean).join("\n");
          })
          .filter((value) => value.length > 0)
          .join("\n");
        const combined = [cleanedText, toolContent].filter(Boolean).join("\n");
        return {
          role: message.role,
          content: combined,
        };
      })
      .filter((message) => message.content.trim().length > 0);
  }, [messages]);
  const contextStats = useMemo(() => {
    console.info("[Context] maxContext", { maxContext });
    if (!maxContext) return null;
    const tools = getToolDefinitions?.() ?? [];
    return calculateStats(contextMessages, maxContext, systemPrompt, tools);
  }, [contextMessages, maxContext, systemPrompt, getToolDefinitions, calculateStats]);
  const contextUsageLabel = useMemo(() => {
    if (!contextStats) return null;
    return `${formatTokenCount(contextStats.currentTokens)} / ${formatTokenCount(
      contextStats.maxContext,
    )}`;
  }, [contextStats, formatTokenCount]);
  const contextBreakdown = useMemo(() => {
    if (!contextStats) return null;
    let userTokens = 0;
    let assistantTokens = 0;
    let thinkingTokens = 0;
    let userMessages = 0;
    let assistantMessages = 0;
    let toolCalls = 0;
    messages.forEach((message) => {
      const textContent = message.parts
        .filter((part): part is { type: "text"; text: string } => part.type === "text")
        .map((part) => part.text)
        .join("");
      const cleaned = stripThinkingForModelContext(textContent);
      const tokens = estimateTokens(cleaned);
      if (message.role === "user") {
        userMessages += 1;
        userTokens += tokens;
      } else {
        assistantMessages += 1;
        assistantTokens += tokens;
      }
      const thinking = parseThinking(textContent).thinkingContent;
      if (thinking) {
        thinkingTokens += estimateTokens(thinking);
      }
      toolCalls += message.parts.filter(
        (part) => typeof part.type === "string" && part.type.startsWith("tool-"),
      ).length;
    });
    return {
      messages: messages.length,
      userMessages,
      assistantMessages,
      toolCalls,
      userTokens,
      assistantTokens,
      thinkingTokens,
    };
  }, [contextStats, estimateTokens, messages, parseThinking]);
  const sessionArtifacts = useMemo(() => {
    if (!artifactsEnabled || messages.length === 0) return [];
    const artifacts: Artifact[] = [];
    messages.forEach((msg) => {
      if (msg.role !== "assistant") return;
      const textContent = msg.parts
        .filter((p): p is { type: "text"; text: string } => p.type === "text")
        .map((p) => p.text)
        .join("");
      if (!textContent) return;
      const { artifacts: extracted } = extractArtifacts(textContent, { includeImplicit: true });
      extracted.forEach((artifact, index) => {
        artifacts.push({
          ...artifact,
          id: `${msg.id}-${index}`,
          message_id: msg.id,
          session_id: currentSessionId || undefined,
        });
      });
    });
    return artifacts;
  }, [messages, artifactsEnabled, currentSessionId]);
  const readUiMessageStream = useCallback(async (response: Response) => {
    const reader = response.body?.getReader();
    if (!reader) return "";
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        try {
          const chunk = JSON.parse(payload) as { type?: string; delta?: string; errorText?: string };
          if (chunk.type === "text-delta" && chunk.delta) {
            text += chunk.delta;
          } else if (chunk.type === "error") {
            throw new Error(chunk.errorText || "Compaction stream error");
          }
        } catch (err) {
          console.error("Failed to parse compaction chunk:", err);
        }
      }
    }
    return text.trim();
  }, []);
  const requestCompactionSummary = useCallback(async () => {
    if (!selectedModel || contextMessages.length === 0) return "";
    const summarySystemPrompt = [
      "You are a context-compaction assistant.",
      "Summarize the conversation so it can replace the full history.",
      "The original first user message and the latest message will be preserved separately.",
      "Do not repeat those messages verbatim; focus on key facts, decisions, preferences, and open tasks.",
      "Include important tool outputs, artifacts, and code references when relevant.",
      "Keep the summary under 10k tokens and use concise bullets and short sections.",
      systemPrompt?.trim() ? `Original system prompt:\n${systemPrompt.trim()}` : "",
    ]
      .filter(Boolean)
      .join("\n\n");
    const summaryMessages: UIMessage[] = contextMessages.map((message, index) => ({
      id: `ctx-${index}`,
      role: message.role,
      parts: [{ type: "text", text: message.content }],
    }));
    summaryMessages.push({
      id: `ctx-summary-${Date.now()}`,
      role: "user",
      parts: [{ type: "text", text: "Summarize the conversation above for context compaction." }],
    });
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: selectedModel,
        system: summarySystemPrompt,
        messages: summaryMessages,
      }),
    });
    if (!response.ok) {
      throw new Error(`Compaction request failed (${response.status})`);
    }
    return readUiMessageStream(response);
  }, [contextMessages, readUiMessageStream, selectedModel, systemPrompt]);
  const runAutoCompaction = useCallback(async () => {
    if (!contextStats || !maxContext) return;
    if (!contextConfig.autoCompact) return;
    if (compacting || isLoading) return;
    if (contextStats.utilization < contextConfig.compactionThreshold) return;
    if (!selectedModel || messages.length < 2) return;
    const signature = `${currentSessionId || "new"}-${messages.length}-${contextStats.currentTokens}`;
    if (lastCompactionSignatureRef.current === signature) return;
    lastCompactionSignatureRef.current = signature;
    setCompacting(true);
    setCompactionError(null);
    try {
      const summaryText = await requestCompactionSummary();
      if (!summaryText) {
        throw new Error("Empty compaction summary");
      }
      const firstUser = messages.find((message) => message.role === "user") ?? null;
      const lastMessage = messages[messages.length - 1] ?? null;
      if (!lastMessage) return;
      const summaryMessage: UIMessage = {
        id: `compaction-summary-${Date.now()}`,
        role: "assistant",
        parts: [
          {
            type: "text",
            text: `Context summary (auto-compacted on ${new Date().toLocaleString()}):\n\n${summaryText}`,
          },
        ],
        metadata: { model: selectedModel },
      };
      const cloneMessage = (message: UIMessage, suffix: string): UIMessage => ({
        ...message,
        id: `${message.role}-${Date.now()}-${suffix}`,
        parts: message.parts.map((part) => ({ ...part })),
      });
      const compactedMessages: UIMessage[] = [];
      if (firstUser) {
        compactedMessages.push(cloneMessage(firstUser, "first"));
      }
      compactedMessages.push(summaryMessage);
      if (!firstUser || firstUser.id !== lastMessage.id) {
        compactedMessages.push(cloneMessage(lastMessage, "last"));
      }
      const compactedTitle =
        currentSessionTitle && !["New Chat", "Chat"].includes(currentSessionTitle)
          ? `${currentSessionTitle} (Compacted)`
          : "Compacted Chat";
      const session = await createSession(compactedTitle, selectedModel);
      if (!session) {
        throw new Error("Failed to create compacted session");
      }
      for (const message of compactedMessages) {
        await persistMessage(session.id, message);
      }
      setMessages(compactedMessages);
      const beforeTokens = calculateMessageTokens(contextMessages);
      const afterTokens = calculateMessageTokens(
        compactedMessages.map((message) => {
          const textContent = message.parts
            .filter((part): part is { type: "text"; text: string } => part.type === "text")
            .map((part) => part.text)
            .join("");
          const cleanedText = stripThinkingForModelContext(textContent);
          const toolContent = message.parts
            .filter(
              (
                part,
              ): part is UIMessage["parts"][number] & {
                input?: unknown;
                output?: unknown;
                errorText?: string;
              } => {
                if (typeof part.type !== "string") return false;
                return part.type === "dynamic-tool" || part.type.startsWith("tool-");
              },
            )
            .map((part) => {
              const input = "input" in part && part.input != null ? JSON.stringify(part.input) : "";
              const output =
                "output" in part && part.output != null ? JSON.stringify(part.output) : "";
              const errorText = "errorText" in part && part.errorText ? part.errorText : "";
              return [input, output, errorText].filter(Boolean).join("\n");
            })
            .filter((value) => value.length > 0)
            .join("\n");
          return {
            role: message.role,
            content: [cleanedText, toolContent].filter(Boolean).join("\n"),
          };
        }),
      );
      setCompactionHistory((prev) => [
        ...prev,
        {
          id: `compact-${Date.now()}`,
          timestamp: new Date(),
          beforeTokens,
          afterTokens,
          messagesRemoved: Math.max(0, messages.length - compactedMessages.length),
          messagesKept: compactedMessages.length,
          maxContext,
          utilizationBefore: beforeTokens / maxContext,
          utilizationAfter: afterTokens / maxContext,
          strategy: "summarize",
          summary: summaryText,
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Compaction failed";
      console.error(message);
      setCompactionError(message);
    } finally {
      setCompacting(false);
    }
  }, [
    calculateMessageTokens,
    compacting,
    contextConfig.autoCompact,
    contextConfig.compactionThreshold,
    contextMessages,
    contextStats,
    createSession,
    currentSessionId,
    currentSessionTitle,
    isLoading,
    maxContext,
    messages,
    persistMessage,
    requestCompactionSummary,
    selectedModel,
    setMessages,
  ]);
  useEffect(() => {
    lastCompactionSignatureRef.current = null;
    setCompactionError(null);
    setCompactionHistory([]);
  }, [currentSessionId]);
  useEffect(() => {
    void runAutoCompaction();
  }, [runAutoCompaction]);
  useEffect(() => {
    if (sessionArtifacts.length === 0) {
      autoArtifactSwitchRef.current = false;
      return;
    }
    if (!autoArtifactSwitchRef.current) {
      autoArtifactSwitchRef.current = true;
      const timeoutId = window.setTimeout(() => {
        setActivePanel("artifacts");
      }, 0);
      return () => window.clearTimeout(timeoutId);
    }
  }, [sessionArtifacts.length, setActivePanel]);
  const showEmptyState = messages.length === 0 && !isLoading && !error && !hasPendingUserMessage;
  // Scroll handling
  const handleScroll = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const { scrollTop, scrollHeight, clientHeight } = container;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    setUserScrolledUp(distanceFromBottom >= 160);
  }, [setUserScrolledUp]);
  // Auto-scroll to bottom
  useEffect(() => {
    if (!userScrolledUp) {
      messagesEndRef.current?.scrollIntoView({
        behavior: isLoading ? "auto" : "smooth",
      });
    }
  }, [isLoading, messages, userScrolledUp]);
  // Ensure a start time exists for any streaming session
  useEffect(() => {
    if (isLoading && streamingStartTime == null) {
      setStreamingStartTime(Date.now());
    }
  }, [isLoading, setStreamingStartTime, streamingStartTime]);
  // Elapsed time timer
  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval> | null = null;
    if (isLoading && streamingStartTime != null) {
      intervalId = setInterval(
        () => setElapsedSeconds(Math.floor((Date.now() - streamingStartTime) / 1000)),
        1000,
      );
    } else if (!isLoading) {
      const timeoutId = setTimeout(() => {
        if (!isLoading) {
          setStreamingStartTime(null);
          setElapsedSeconds(0);
        }
      }, 3000);
      return () => clearTimeout(timeoutId);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isLoading, setElapsedSeconds, setStreamingStartTime, streamingStartTime]);
  
  const applyMessagesInBatches = useCallback(
    (storedMessages: StoredMessage[], options?: { immediate?: boolean }) => {
      const full = mapStoredMessages(storedMessages);
      if (options?.immediate || full.length <= 50) {
        setMessages(full);
        return;
      }
      const first = mapStoredMessages(storedMessages.slice(0, 50));
      setMessages(first);
      const schedule = () => setMessages(full);
      if ("requestIdleCallback" in window) {
        (window as Window & { requestIdleCallback?: (cb: () => void) => void }).requestIdleCallback?.(schedule);
      } else {
        window.setTimeout(schedule, 0);
      }
    },
    [mapStoredMessages, setMessages],
  );

  const getSessionCache = useCallback((sessionId: string) => {
    try {
      const raw = localStorage.getItem(`chat-cache:${sessionId}`);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as { messages?: StoredMessage[]; savedAt?: number };
      if (!parsed || !Array.isArray(parsed.messages)) return null;
      return parsed;
    } catch {
      return null;
    }
  }, []);

  const setSessionCache = useCallback((sessionId: string, messages: StoredMessage[]) => {
    try {
      localStorage.setItem(
        `chat-cache:${sessionId}`,
        JSON.stringify({ messages, savedAt: Date.now() }),
      );
    } catch {
      // ignore cache failures
    }
  }, []);

  const restoreSourcesFromSession = useCallback((storedMessages: StoredMessage[]) => {
    if (!storedMessages || storedMessages.length === 0) return;
    const results = new Map<string, { tool_call_id: string; content: string; isError?: boolean }>();
    storedMessages.forEach((msg) => {
      const sources = (msg as { sources?: Array<{ title: string; url: string }> }).sources;
      if (!sources || sources.length === 0) return;
      setMessageSources(msg.id, sources);
      const output = sources
        .map((s) => `Title: ${s.title}\nURL: ${s.url}`)
        .join("\n\n");
      const toolCallId = `sources_used:${msg.id}`;
      results.set(toolCallId, { tool_call_id: toolCallId, content: output, isError: false });
    });
    if (results.size > 0) {
      updateToolResultsMap((prev) => {
        const next = new Map(prev);
        for (const [key, value] of results.entries()) {
          next.set(key, value);
        }
        return next;
      });
    }
  }, [setMessageSources, updateToolResultsMap]);

  const prefetchChatSession = useCallback(async (sessionId: string) => {
    try {
      if (getSessionCache(sessionId)?.messages) return;
      const data = await api.getChatSession(sessionId);
      const storedMessages = data.session?.messages ?? [];
      if (storedMessages.length > 0) {
        setSessionCache(sessionId, storedMessages);
      }
    } catch {
      // ignore prefetch failures
    }
  }, [getSessionCache, setSessionCache]);

  useEffect(() => {
    if (!sessions || sessions.length === 0) return;
    const ids = sessions.slice(0, 3).map((s) => s.id).filter(Boolean);
    if (ids.length === 0) return;
    const run = () => {
      ids.forEach((id) => {
        if (typeof id === "string") {
          void prefetchChatSession(id);
        }
      });
    };
    if ("requestIdleCallback" in window) {
      (window as Window & { requestIdleCallback?: (cb: () => void) => void }).requestIdleCallback?.(run);
    } else {
      window.setTimeout(run, 250);
    }
  }, [sessions, prefetchChatSession]);

// Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    clearError?.();
  }, [currentSessionId, clearError]);

  // Handle PWA resume - reload session when app becomes visible again
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        // Reload current session to restore messages after PWA was backgrounded
        const sessionId = sessionIdRef.current ?? currentSessionId;
        if (sessionId) {
          void (async () => {
            try {
              const cached = getSessionCache(sessionId);
              if (cached?.messages && messages.length === 0) {
                setMessages(mapStoredMessages(cached.messages));
                restoreSourcesFromSession(cached.messages);
              }
              const session = await loadSession(sessionId);
              if (session) {
                const storedMessages = session.messages ?? [];
                // Only restore if we lost messages (PWA was killed)
                if (messages.length === 0 && storedMessages.length > 0) {
                  setIsRestoringSession(true);
                  applyMessagesInBatches(storedMessages);
                  restoreSourcesFromSession(storedMessages);
                  setSessionCache(sessionId, storedMessages);
                  setIsRestoringSession(false);
                }
              }
            } catch (err) {
              console.error("Failed to restore session on resume:", err);
            }
          })();
        }
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, [currentSessionId, loadSession, mapStoredMessages, messages.length, setMessages, restoreSourcesFromSession, getSessionCache, setSessionCache, applyMessagesInBatches]);
// Handle URL session/new params
  useEffect(() => {
    if (newChatFromUrl) {
      clearError?.();
      startNewSession();
      setMessages([]);
      setHasPendingUserMessage(false);
      setPendingUserText(null);
      setPendingStatus(null);
      router.replace("/chat");
      return;
    }
    if (sessionFromUrl) {
      const cached = getSessionCache(sessionFromUrl);
      if (cached?.messages) {
        setIsRestoringSession(true);
        applyMessagesInBatches(cached.messages);
        restoreSourcesFromSession(cached.messages);
      }
      void (async () => {
        const session = await loadSession(sessionFromUrl);
        if (session) {
          if (session.model && session.model !== selectedModel) {
            setSelectedModel(session.model);
          }
          const storedMessages = session.messages ?? [];
          setIsRestoringSession(true);
          applyMessagesInBatches(storedMessages);
          restoreSourcesFromSession(storedMessages);
          setSessionCache(sessionFromUrl, storedMessages);
          setIsRestoringSession(false);
        }
      })();
    }
  }, [
    newChatFromUrl,
    sessionFromUrl,
    startNewSession,
    loadSession,
    setMessages,
    mapStoredMessages,
    selectedModel,
    setSelectedModel,
    restoreSourcesFromSession,
    getSessionCache,
    setSessionCache,
    applyMessagesInBatches,
  ]);
  useEffect(() => {
    if (messages.length > 0) {
      setHasPendingUserMessage(false);
      setPendingUserText(null);
      setPendingStatus(null);
    }
  }, [messages.length]);

  // Load MCP servers/tools when enabled
  useEffect(() => {
    if (mcpEnabled) {
      loadMCPServers();
      loadMCPTools();
    }
  }, [mcpEnabled, loadMCPServers, loadMCPTools]);
  
  const refreshRuntimeContext = useCallback(async (modelId?: string) => {
    const targetModel = modelId || selectedModel;
    if (!targetModel) return;
    try {
      let slotRes = await fetch('/api/proxy/slots');
      if (!slotRes.ok) {
        slotRes = await fetch('/api/proxy/v1/slots');
      }
      if (!slotRes.ok) return;
      const slots = await slotRes.json();
      if (Array.isArray(slots) && slots.length > 0 && typeof slots[0]?.n_ctx === 'number') {
        setRuntimeContextByModel((prev) => ({ ...prev, [targetModel]: slots[0].n_ctx }));
      }
    } catch {
      // ignore runtime context failures
    }
  }, [selectedModel]);

  useEffect(() => {
    void refreshRuntimeContext(selectedModel);
  }, [selectedModel, refreshRuntimeContext]);

// Load available models from OpenAI-compatible endpoint on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const data = await api.getOpenAIModels();
        const dataModels = (data as { data?: unknown[] }).data;
        const modelsField = (data as { models?: unknown[] }).models;
        const rawModels = Array.isArray(data)
          ? data
          : Array.isArray(dataModels)
            ? dataModels
            : Array.isArray(modelsField)
              ? modelsField
              : [];
        const mappedModels = rawModels
          .flatMap((model) => {
            if (!model || typeof model !== "object") return [];
            const record = model as {
              id?: string;
              model?: string;
              name?: string;
              max_model_len?: number;
              meta?: { n_ctx_train?: number };
            };
            const id = record.id ?? record.model ?? record.name;
            if (!id) return [];
            const maxModelLen = record.max_model_len ?? record.meta?.n_ctx_train ?? undefined;
            const runtimeContext = runtimeContextByModel[id];
            const vision = Boolean((record as { vision?: boolean }).vision);
            return [
              {
                id,
                name: id,
                maxModelLen,
                runtimeContext,
                vision,
              },
            ];
          })
          .sort((a, b) => a.id.localeCompare(b.id));
        setAvailableModels(mappedModels);
        const lastModel = localStorage.getItem("vllm-studio-last-model");
        const fallbackModel = mappedModels[0]?.id || "";
        let next = selectedModel;
        if (mappedModels.length === 0) {
          if (lastModel && !next) {
            setSelectedModel(lastModel);
          }
          return;
        }
        if (next && mappedModels.some((model) => model.id === next)) {
          // keep selected model
        } else if (lastModel && mappedModels.some((model) => model.id === lastModel)) {
          next = lastModel;
        } else if (!next || !mappedModels.some((model) => model.id === next)) {
          next = fallbackModel;
        }
        if (next && next !== selectedModel) {
          setSelectedModel(next);
        }
      } catch (err) {
        console.error("Failed to load models:", err);
      }
    };
    loadModels();
  }, [selectedModel, setAvailableModels, setSelectedModel]);
  useEffect(() => {
    if (!realtimeStatus?.running || !realtimeStatus.process) return;
    const served = realtimeStatus.process.served_model_name;
    const path = realtimeStatus.process.model_path;
    const base = path ? path.split("/").pop() : null;
    const candidates = [served, base].filter(Boolean) as string[];
    let next: string | null = null;
    for (const candidate of candidates) {
      if (availableModels.some((model) => model.id === candidate)) {
        next = candidate;
        break;
      }
    }
    if (!next && candidates.length > 0) {
      next = candidates[0] ?? null;
    }
    if (next && next !== selectedModel) {
      setSelectedModel(next);
    }
  }, [realtimeStatus, availableModels, selectedModel, setSelectedModel]);
// Load MCP servers when settings modal opens
  useEffect(() => {
    if (mcpSettingsOpen) {
      loadMCPServers();
    }
  }, [mcpSettingsOpen, loadMCPServers]);
  // Refresh usage when modal opens
  useEffect(() => {
    if (usageOpen && currentSessionId) {
      refreshUsage(currentSessionId);
    }
  }, [usageOpen, currentSessionId, refreshUsage]);
  // Export functions
  const handleExportJson = useCallback(() => {
    const data = {
      title: currentSessionTitle,
      sessionId: currentSessionId,
      model: selectedModel,
      messages: messages.map((m) => ({
        id: m.id,
        role: m.role,
        parts: m.parts,
      })),
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-${currentSessionId || "export"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentSessionId, currentSessionTitle, selectedModel, messages]);
  const handleExportMarkdown = useCallback(() => {
    let md = `# ${currentSessionTitle}\n\n`;
    md += `Model: ${selectedModel}\n`;
    md += `Exported: ${new Date().toLocaleString()}\n\n---\n\n`;
    for (const msg of messages) {
      const role = msg.role === "user" ? "**User**" : "**Assistant**";
      md += `${role}:\n\n`;
      for (const part of msg.parts) {
        if (part.type === "text") {
          md += `${(part as { text: string }).text}\n\n`;
        } else if (part.type.startsWith("tool-") && "toolCallId" in part) {
          md += `> Tool: ${part.type.replace(/^tool-/, "")}\n\n`;
        }
      }
      md += "---\n\n";
    }
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-${currentSessionId || "export"}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentSessionId, currentSessionTitle, selectedModel, messages]);
  const isVisionModel = useCallback(
    (modelId: string | null): boolean => {
      if (!modelId) return false;
      const entry = availableModels.find((model) => model.id === modelId);
      if (entry?.vision) return true;
      const lower = modelId.toLowerCase();
      return lower.includes("vl") || lower.includes("vision") || lower.includes("llava") || lower.includes("minicpm");
    },
    [availableModels],
  );

  const buildFilePartsFromAttachments = (attachments?: Attachment[]) => {
    if (!attachments || attachments.length === 0) return [] as Array<{ type: "file"; mediaType: string; filename?: string; url: string }>;
    const fileParts: Array<{ type: "file"; mediaType: string; filename?: string; url: string }> = [];
    for (const att of attachments) {
      if (att.type === "image" && att.base64) {
        const mediaType = att.file?.type || "image/png";
        const url = `data:${mediaType};base64,${att.base64}`;
        fileParts.push({ type: "file", mediaType, filename: att.name, url });
      }
    }
    return fileParts;
  };

const sendUserMessage = useCallback(
    async (text: string, attachments?: Attachment[], options?: { clearInput?: boolean }) => {
      if (!selectedModel) return;
      if (!text.trim() && (!attachments || attachments.length === 0)) return;
      if (isLoading) return;
      const hasImage = Boolean(attachments?.some((att) => att.type === "image"));
      if (hasImage && !isVisionModel(selectedModel)) {
        clearError?.();
        setPendingStatus("Image input is not supported by the selected model. Switch to a vision model.");
        window.setTimeout(() => setPendingStatus(null), 4000);
        return;
      }
      // Only open side panel on desktop
      if (window.innerWidth >= 768) {
        setToolPanelOpen(true);
        setActivePanel("activity");
      }
      setStreamingStartTime(Date.now());
      if (options?.clearInput) {
        setInput("");
      }
      // Store for title generation
      lastUserInputRef.current = text;
      lastSearchSourcesRef.current = [];
      let finalText = text;
      flushSync(() => {
        setHasPendingUserMessage(true);
        setPendingUserText(finalText);
        setPendingStatus("Preparing request...");
      });
      let deepResearchContext: string | null = null;
      let webSearchContext: string | null = null;
      const finalQuery = text.trim();

      if (mcpEnabled && !deepResearch.enabled && finalQuery) {
        try {
          setPendingStatus("Web search in progress...");
          const toolResult = await executeTool({
            toolCallId: `exa__web_search_exa:${Date.now()}`,
            toolName: "exa__web_search_exa",
            args: {
              query: finalQuery,
              numResults: 8,
              livecrawl: "preferred",
              type: "deep",
              contextMaxCharacters: 20000,
            },
          });
          if (!toolResult.isError && toolResult.content) {
            setPendingStatus("Web search complete. Preparing response...");
            const sources = extractSourcesFromToolContent(toolResult.content);
            if (sources.length > 0) {
              lastSearchSourcesRef.current = sources;
            }
            const sourcesText = sources.length
              ? sources
                  .map((source, idx) => {
                    const label = source.title || source.url || `Source ${idx + 1}`;
                    return `${idx + 1}. ${label}${source.url ? ` (${source.url})` : ""}`;
                  })
                  .join("\n")
              : "";
            webSearchContext = `Use the web search results below to answer the user directly. Cite sources inline in every paragraph or claim (e.g., [1], [2]) and use multiple sources when appropriate. If the results are insufficient, say so.

${sourcesText ? `Sources:
${sourcesText}

` : ""}Search Results:
${toolResult.content}`;
          }
        } catch (err) {
          console.warn("[WebSearch] tool failed", err);
        }
      }
      const resolveDeepResearchToolName = async () => {
        if (deepResearchToolRef.current) return deepResearchToolRef.current;
        const preferredServers = ["gptr-mcp", "gpt-researcher", "gptr"];
        for (const server of preferredServers) {
          try {
            const { tools } = await api.getMCPServerTools(server);
            const tool = tools.find((t) => t.name === "deep_research");
            if (tool) {
              const name = `${server}__${tool.name}`;
              deepResearchToolRef.current = name;
              return name;
            }
          } catch {
            // ignore missing server
          }
        }
        try {
          const { tools } = await api.getMCPTools();
          const tool = tools.find((t) => t.name === "deep_research");
          if (tool) {
            const name = `${tool.server}__${tool.name}`;
            deepResearchToolRef.current = name;
            return name;
          }
        } catch {
          // ignore
        }
        return null;
      };
      if (deepResearch.enabled && finalQuery) {
        try {
          setPendingStatus("Deep research in progress...");
          const toolName = await resolveDeepResearchToolName();
          if (toolName) {
            const toolResult = await executeTool({
              toolCallId: `${toolName}:${Date.now()}`,
              toolName,
              args: {
                query: finalQuery,
                model: selectedModel,
              },
            });
            if (!toolResult.isError && toolResult.content) {
              setPendingStatus("Deep research complete. Preparing response...");
              const maxChars = 12000;
              const parsed = tryParseNestedJsonString(toolResult.content) as
                | Record<string, unknown>
                | null;
              let contextText = toolResult.content;
              let sourcesText = "";
              if (parsed && typeof parsed === "object") {
                if (typeof parsed.context === "string") {
                  contextText = parsed.context;
                }
                const sources = Array.isArray(parsed.sources) ? parsed.sources : [];
                lastSearchSourcesRef.current = sources
                  .map((source) => {
                    if (!source || typeof source !== "object") return null;
                    const record = source as Record<string, unknown>;
                    const title = typeof record.title === "string" ? record.title : "";
                    const url = typeof record.url === "string" ? record.url : "";
                    if (!url) return null;
                    return { title: title || url, url };
                  })
                  .filter(Boolean) as Array<{ title: string; url: string }>;
                if (sources.length > 0) {
                  sourcesText = sources
                    .map((source, idx) => {
                      if (source && typeof source === "object") {
                        const record = source as Record<string, unknown>;
                        const title = typeof record.title === "string" ? record.title : "";
                        const url = typeof record.url === "string" ? record.url : "";
                        const label = title || url || `Source ${idx + 1}`;
                        return `${idx + 1}. ${label}${url ? ` (${url})` : ""}`;
                      }
                      return `${idx + 1}. Source ${idx + 1}`;
                    })
                    .join("\n");
                }
              }
              const combined = sourcesText
                ? `Sources:
${sourcesText}
Context:
${contextText}`
                : contextText;
              deepResearchContext = combined.length > maxChars
                ? combined.slice(0, maxChars) + "\n\n[Truncated]\n"
                : combined;
            }
          } else {
            console.warn("[DeepResearch] tool not found; check MCP server configuration");
          }
        } catch (err) {
          console.warn("[DeepResearch] tool failed", err);
        }
      }
      const parts: UIMessage["parts"] = [];
      if (finalText.trim()) {
        parts.push({ type: "text", text: finalText });
      }
      // Add image attachments as file parts (data URLs) so vision models receive them
      const fileParts = buildFilePartsFromAttachments(attachments);
      for (const filePart of fileParts) {
        parts.push(filePart);
      }
      if (attachments) {
        for (const att of attachments) {
          if (att.type === "file" && att.file) {
            parts.push({
              type: "text",
              text: `[File: ${att.name}]`,
            });
          }
        }
      }
      const userMessage: UIMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        parts,
      };
      // Create session if needed, then persist user message
      let sessionId = currentSessionId;
      if (!sessionId) {
        sessionId = await createSessionWithMessage(userMessage);
      } else {
        await persistMessage(sessionId, userMessage);
      }
      const toolDefinitions = await resolveToolDefinitions();
      const trimmedSystem = systemPrompt?.trim() || undefined;
      const deepResearchSystem = deepResearchContext
        ? `You have a deep research report below. Use it to answer the user, cite sources inline for each paragraph or claim (e.g., [1], [2]), and call out uncertainty if the report is insufficient.
${deepResearchContext}`
        : undefined;
      const webSearchSystem = webSearchContext ? webSearchContext : undefined;
      const mergedSystem = [trimmedSystem, deepResearchSystem, webSearchSystem].filter(Boolean).join("\n\n") || undefined;
      console.info("[CHAT UI] context", {
        model: selectedModel,
        systemLength: trimmedSystem?.length ?? 0,
        messageCount: messages.length + 1,
        tools: toolDefinitions.map((tool) => tool.name),
        mcpEnabled,
        deepResearchEnabled: deepResearch.enabled,
      });
      setPendingStatus("Sending to model...");

      // Send the message via AI SDK - use simple text format
      // Note: For image attachments, the files would need to be passed as FileList
      // but our current attachment handling uses base64. For now, just send text.
      sendMessage(
        {
          text: finalText,
          files: fileParts.length > 0 ? fileParts : undefined,
        },
        {
          body: {
            model: selectedModel || undefined,
            system: mergedSystem,
            tools: webSearchContext ? [] : toolDefinitions,
          },
        },
      );
      setPendingStatus("Awaiting response...");
    },
    [
      isLoading,
      sendMessage,
      currentSessionId,
      createSessionWithMessage,
      persistMessage,
      selectedModel,
      resolveToolDefinitions,
      systemPrompt,
      messages.length,
      mcpEnabled,
      setToolPanelOpen,
      setActivePanel,
      setStreamingStartTime,
      setInput,
      executeTool,
      deepResearch,
    ],
  );
  // Handle send with persistence and attachments
  const handleSend = useCallback(
    async (attachments?: Attachment[]) => {
      await sendUserMessage(input, attachments, { clearInput: true });
    },
    [input, sendUserMessage],
  );
  const handleReprompt = useCallback(
    async (messageId: string) => {
      if (isLoading) return;
      const messageIndex = messages.findIndex((msg) => msg.id === messageId);
      if (messageIndex <= 0) return;
      const previousUser = [...messages.slice(0, messageIndex)]
        .reverse()
        .find((msg) => msg.role === "user");
      if (!previousUser) return;
      const userText = previousUser.parts
        .filter((part): part is { type: "text"; text: string } => part.type === "text")
        .map((part) => part.text)
        .join("");
      if (!userText.trim()) return;
      await sendUserMessage(userText);
    },
    [messages, isLoading, sendUserMessage],
  );
  const handleForkMessage = useCallback(
    async (messageId: string) => {
      if (!currentSessionId) return;
      try {
        const { session } = await api.forkChatSession(currentSessionId, {
          message_id: messageId,
          model: selectedModel || undefined,
          title: "New Chat",
        });
        updateSessions((sessions) => {
          if (sessions.some((existing) => existing.id === session.id)) {
            return sessions.map((existing) => (existing.id === session.id ? session : existing));
          }
          return [session, ...sessions];
        });
        router.push(`/chat?session=${session.id}`);
      } catch (err) {
        console.error("Failed to fork session:", err);
      }
    },
    [currentSessionId, selectedModel, router, updateSessions],
  );
  // Handle stop
  const handleStop = useCallback(() => {
    stop();
    setStreamingStartTime(null);
    setElapsedSeconds(0);
  }, [setElapsedSeconds, setStreamingStartTime, stop]);
  // Handle model change and persist to localStorage
  const handleModelChange = useCallback(
    (modelId: string) => {
      setSelectedModel(modelId);
      localStorage.setItem("vllm-studio-last-model", modelId);
    },
    [setSelectedModel],
  );
  const toolBelt = (
    <ToolBelt
      value={input}
      onChange={setInput}
      onSubmit={handleSend}
      onStop={handleStop}
      disabled={false}
      isLoading={isLoading}
      placeholder={selectedModel ? "Message..." : "Select a model"}
      selectedModel={selectedModel}
      availableModels={availableModels}
      onModelChange={handleModelChange}
      mcpEnabled={mcpEnabled}
      onMcpToggle={() => {
        const nextEnabled = !mcpEnabled || deepResearch.enabled;
        console.log("[ChatPage] MCP toggle:", nextEnabled);
        if (deepResearch.enabled) {
          setDeepResearch({ ...deepResearch, enabled: false });
        }
        setMcpEnabled(nextEnabled);
      }}
      artifactsEnabled={artifactsEnabled}
      onArtifactsToggle={() => {
        console.log("[ChatPage] Artifacts toggle:", !artifactsEnabled);
        setArtifactsEnabled(!artifactsEnabled);
      }}
      deepResearchEnabled={deepResearch.enabled}
      onDeepResearchToggle={() => {
        const nextEnabled = !deepResearch.enabled;
        console.log("[ChatPage] Deep Research toggle:", nextEnabled);
        setDeepResearch({ ...deepResearch, enabled: nextEnabled });
        if (nextEnabled) {
          setMcpEnabled(true);
        }
      }}
      elapsedSeconds={elapsedSeconds}
      queuedContext={queuedContext}
      onQueuedContextChange={setQueuedContext}
      onOpenMcpSettings={() => setMcpSettingsOpen(true)}
      onOpenChatSettings={() => setSettingsOpen(true)}
      hasSystemPrompt={systemPrompt.trim().length > 0}
    />
  );
  return (
    <div className="relative h-full flex overflow-hidden w-full max-w-full">
      <div className="flex-1 flex flex-col min-h-0 min-w-0 overflow-x-hidden">
        <div className="flex-1 flex overflow-hidden relative min-w-0">
          <div className="flex-1 flex flex-col overflow-hidden relative min-w-0">
            <ChatConversation
              messages={messages}
              isLoading={isLoading}
              error={error?.message}
              artifactsEnabled={artifactsEnabled}
              selectedModel={selectedModel}
              contextUsageLabel={contextUsageLabel}
              onFork={handleForkMessage}
              onReprompt={handleReprompt}
              showEmptyState={showEmptyState}
              pendingUserText={pendingUserText}
              pendingStatus={pendingStatus}
              toolBelt={toolBelt}
              onScroll={handleScroll}
              messagesContainerRef={messagesContainerRef}
              messagesEndRef={messagesEndRef}
            />
            <ChatTopControls
              onOpenSidebar={() => {
                window.dispatchEvent(
                  new CustomEvent("vllm:toggle-sidebar", { detail: { open: true } }),
                );
              }}
              onOpenSettings={() => setSettingsOpen(true)}
            />
            <ChatActionButtons
              activityCount={activityCount}
              onOpenActivity={() => {
                setToolPanelOpen(true);
                setActivePanel("activity");
              }}
              onOpenContext={() => {
                setToolPanelOpen(true);
                setActivePanel("context");
              }}
              onOpenSettings={() => setSettingsOpen(true)}
              onOpenMcpSettings={() => setMcpSettingsOpen(true)}
              onOpenUsage={() => setUsageOpen(true)}
              onOpenExport={() => setExportOpen(true)}
            />
            <ChatToolbeltDock toolBelt={toolBelt} showEmptyState={showEmptyState} />
          </div>
          {toolPanelOpen && (
            <ChatSidePanel
              isOpen={toolPanelOpen}
              onClose={() => setToolPanelOpen(false)}
              activePanel={activePanel}
              onSetActivePanel={setActivePanel}
              activityGroups={activityGroups}
              thinkingActive={thinkingActive}
              executingTools={executingTools}
              artifacts={sessionArtifacts}
              contextStats={contextStats}
              contextBreakdown={contextBreakdown}
              compactionHistory={compactionHistory}
              compacting={compacting}
              compactionError={compactionError}
              formatTokenCount={formatTokenCount}
            />
          )}
        </div>
      </div>
      <ChatModals
        settingsOpen={settingsOpen}
        onCloseSettings={() => setSettingsOpen(false)}
        mcpSettingsOpen={mcpSettingsOpen}
        onCloseMcpSettings={() => setMcpSettingsOpen(false)}
        usageOpen={usageOpen}
        onCloseUsage={() => setUsageOpen(false)}
        exportOpen={exportOpen}
        onCloseExport={() => setExportOpen(false)}
        systemPrompt={systemPrompt}
        onSystemPromptChange={setSystemPrompt}
        selectedModel={selectedModel}
        onSelectedModelChange={setSelectedModel}
        availableModels={availableModels}
        deepResearch={deepResearch}
        onDeepResearchChange={setDeepResearch}
        mcpServers={mcpServers}
        onAddServer={addMcpServer}
        onUpdateServer={updateMcpServer}
        onRemoveServer={removeMcpServer}
        onRefreshServers={loadMCPServers}
        sessionUsage={sessionUsage}
        messages={messages}
        onExportJson={handleExportJson}
        onExportMarkdown={handleExportMarkdown}
      />
    </div>
  );
}
