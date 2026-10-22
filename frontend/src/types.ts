/**
 * NEXA AI - TypeScript Type Definitions
 */

export type MessageRole = 'user' | 'assistant' | 'system' | 'tool';
export type Language = 'en' | 'ur' | 'auto';

export interface ChatMessage {
  role: MessageRole;
  content: string;
  timestamp?: string;
  toolCalls?: ToolCallInfo[];
  isStreaming?: boolean;
}

export interface ToolCallInfo {
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_output?: unknown;
  duration_ms?: number;
  success: boolean;
  error?: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  history: { role: string; content: string }[];
  language: Language;
  enabled_tools?: string[];
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  tool_calls: ToolCallInfo[];
  model: string;
  usage?: { input_tokens: number; output_tokens: number };
  language_detected?: string;
}

export interface ToolDefinition {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  timeout: number;
  permission: string;
  enabled: boolean;
}

export interface AgentStatus {
  api_key_configured: boolean;
  model: string;
  tools_available: string[];
  tools_enabled: string[];
  max_steps: number;
  database_connected: boolean;
}

export interface StreamEventData {
  event: 'session' | 'text_delta' | 'tool_start' | 'tool_end' | 'done' | 'error';
  data: Record<string, unknown>;
}

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  size_bytes: number;
  content_type: string;
  status: string;
}

export interface SessionInfo {
  session_id: string;
  created_at: string;
  last_active: string;
  message_count: number;
  language: Language;
}
