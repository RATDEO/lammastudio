import { useState, useCallback, useEffect, useRef } from 'react';
import { useSSE } from './useSSE';

interface LogEvent {
  data: {
    line: string;
  };
  timestamp: string;
}

/**
 * Hook for real-time log streaming.
 *
 * Receives initial log content then streams new lines as they're added.
 *
 * @param sessionId - Log session ID to stream
 * @param apiBaseUrl - Base URL for the API (default: empty for relative URLs)
 * @param maxLines - Maximum number of lines to keep in memory (default: 5000)
 */
export function useLogStream(
  sessionId: string | null,
  apiBaseUrl: string = '/api/proxy',
  maxLines: number = 5000
) {
  const [logs, setLogs] = useState<string[]>([]);
  const logsRef = useRef<string[]>([]);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const payload: LogEvent = JSON.parse(event.data);

      if (payload.data?.line !== undefined) {
        // Add new line to logs
        logsRef.current = [...logsRef.current, payload.data.line];

        // Trim if over limit
        if (logsRef.current.length > maxLines) {
          logsRef.current = logsRef.current.slice(-maxLines);
        }

        setLogs([...logsRef.current]);
      }
    } catch (e) {
      console.error('[LogStream] Failed to parse log event:', e);
    }
  }, [maxLines]);

  const url = sessionId ? `${apiBaseUrl}/logs/${sessionId}/stream` : '';

  const { isConnected, error } = useSSE(
    url,
    !!sessionId,
    {
      onMessage: handleMessage,
      reconnectDelay: 2000,
    }
  );

  // Reset logs when session changes
  useEffect(() => {
    logsRef.current = [];
    setLogs([]);
  }, [sessionId]);

  return {
    logs,
    isConnected,
    error,
  };
}
