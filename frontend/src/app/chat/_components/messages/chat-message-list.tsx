// CRITICAL
"use client";

import { useCallback } from "react";
import type { UIMessage } from "@ai-sdk/react";
import { Loader2 } from "lucide-react";
import { ChatMessageItem } from "./chat-message-item";
import { useAppStore } from "@/store";

interface ChatMessageListProps {
  messages: UIMessage[];
  isLoading: boolean;
  error?: string | null;
  artifactsEnabled?: boolean;
  selectedModel?: string;
  contextUsageLabel?: string | null;
  onFork?: (messageId: string) => void;
  onReprompt?: (messageId: string) => void;
  pendingUserText?: string | null;
  pendingStatus?: string | null;
}

export function ChatMessageList({
  messages,
  isLoading,
  error,
  artifactsEnabled = false,
  selectedModel,
  contextUsageLabel,
  onFork,
  onReprompt,
  pendingUserText,
  pendingStatus,
}: ChatMessageListProps) {
  const copiedMessageId = useAppStore((state) => state.copiedMessageId);
  const setCopiedMessageId = useAppStore((state) => state.setCopiedMessageId);
  const lastMessage = messages[messages.length - 1];
  const showLoadingIndicator = isLoading && lastMessage?.role === "user";

  const handleCopy = useCallback(async (text: string, messageId: string) => {
    if (!text.trim()) return;
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(textarea);
        if (!ok) {
          throw new Error("Clipboard copy failed");
        }
      }
      setCopiedMessageId(messageId);
      window.setTimeout(() => {
        const current = useAppStore.getState().copiedMessageId;
        if (current === messageId) {
          setCopiedMessageId(null);
        }
      }, 2000);
    } catch (err) {
      console.error("Failed to copy message:", err);
    }
  }, [setCopiedMessageId]);

  const handleExport = useCallback(
    (payload: {
      messageId: string;
      role: "user" | "assistant";
      content: string;
      model?: string;
      totalTokens?: number;
    }) => {
      if (!payload.content.trim()) return;

      const headerLines = [
        `# ${payload.role === "assistant" ? "Assistant" : "User"} Message`,
        payload.model ? `Model: ${payload.model}` : null,
        payload.totalTokens ? `Total tokens: ${payload.totalTokens}` : null,
        `Exported: ${new Date().toLocaleString()}`,
        "",
      ].filter(Boolean);

      const md = [...headerLines, payload.content].join("\n");

      const blob = new Blob([md], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `message-${payload.messageId}.md`;
      a.click();
      URL.revokeObjectURL(url);
    },
    [],
  );

  return (
    <div className="flex flex-col gap-4 px-4 md:px-6 py-4 max-w-4xl mx-auto w-full">
      {messages.map((message, index) => (
        <ChatMessageItem
          key={message.id}
          message={message}
          isStreaming={isLoading && index === messages.length - 1 && message.role === "assistant"}
          artifactsEnabled={artifactsEnabled}
          selectedModel={selectedModel}
          contextUsageLabel={contextUsageLabel}
          copied={copiedMessageId === message.id}
          onCopy={handleCopy}
          onFork={message.role === "assistant" ? onFork : undefined}
          onReprompt={message.role === "assistant" ? onReprompt : undefined}
          onExport={handleExport}
        />
      ))}

      {pendingUserText ? (
        <>
          <ChatMessageItem
            key={"pending-user"}
            message={{ id: "pending-user", role: "user", parts: [{ type: "text", text: pendingUserText }] }}
            isStreaming={false}
            artifactsEnabled={artifactsEnabled}
            selectedModel={selectedModel}
            contextUsageLabel={contextUsageLabel}
            copied={false}
            onCopy={handleCopy}
            onExport={handleExport}
          />
          {pendingStatus ? (
            <div className="flex items-center gap-2 text-[#9a9590] text-xs pl-2">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
              <span>{pendingStatus}</span>
            </div>
          ) : null}
        </>
      ) : null}
      {showLoadingIndicator && (
        <div className="flex items-center gap-2 text-[#9a9590]">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Generating response...</span>
        </div>
      )}
      {error && (
        <div className="px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}
