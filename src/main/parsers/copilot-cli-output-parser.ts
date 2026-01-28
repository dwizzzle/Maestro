/**
 * GitHub Copilot CLI Output Parser
 *
 * Parses JSON output from GitHub Copilot CLI (`copilot --acp`).
 * Copilot CLI uses the Agent Client Protocol (ACP) which is based on JSON-RPC.
 *
 * ACP message types (based on the Agent Client Protocol spec):
 * - Session initialization messages with session_id
 * - Streaming delta messages with incremental text
 * - Tool use messages for agent actions
 * - Completion messages when the agent finishes
 *
 * Key schema details:
 * - Output is JSON-RPC based with jsonrpc: "2.0" field
 * - Session IDs are provided in initialization messages
 * - Text content streams via delta objects
 * - Tool calls are reported with name and arguments
 *
 * @see https://github.com/github/copilot-cli
 * @see https://github.com/agentclientprotocol/agent-client-protocol
 */

import type { ToolType, AgentError } from '../../shared/types';
import type { AgentOutputParser, ParsedEvent } from './agent-output-parser';
import { getErrorPatterns, matchErrorPattern } from './error-patterns';

/**
 * Known model context window sizes for Copilot CLI models (in tokens)
 */
const MODEL_CONTEXT_WINDOWS: Record<string, number> = {
	// Claude models
	'claude-sonnet-4.5': 200000,
	'claude-haiku-4.5': 200000,
	'claude-opus-4.5': 200000,
	'claude-sonnet-4': 200000,
	// GPT models
	'gpt-5.2-codex': 400000,
	'gpt-5.2': 400000,
	'gpt-5.1-codex-max': 200000,
	'gpt-5.1-codex': 200000,
	'gpt-5.1': 200000,
	'gpt-5': 200000,
	'gpt-5.1-codex-mini': 128000,
	'gpt-5-mini': 128000,
	'gpt-4.1': 128000,
	// Gemini
	'gemini-3-pro-preview': 1000000,
	// Default fallback
	default: 200000,
};

/**
 * Get the context window size for a given model
 */
function getModelContextWindow(model: string): number {
	return MODEL_CONTEXT_WINDOWS[model] || MODEL_CONTEXT_WINDOWS['default'];
}

/**
 * ACP JSON-RPC message structure
 */
interface AcpMessage {
	jsonrpc?: '2.0';
	id?: number | string;
	method?: string;
	params?: AcpParams;
	result?: AcpResult;
	error?: AcpError;
}

/**
 * ACP params for method calls
 */
interface AcpParams {
	sessionId?: string;
	delta?: {
		text?: string;
		toolCall?: {
			name?: string;
			arguments?: unknown;
		};
	};
	message?: {
		role?: string;
		content?: string;
	};
	tool?: {
		name?: string;
		input?: unknown;
		output?: unknown;
		status?: 'running' | 'completed' | 'failed';
	};
	intent?: string;
}

/**
 * ACP result for completed requests
 */
interface AcpResult {
	sessionId?: string;
	output?: string;
	message?: {
		role?: string;
		content?: string;
	};
	usage?: {
		inputTokens?: number;
		outputTokens?: number;
	};
}

/**
 * ACP error structure
 */
interface AcpError {
	code?: number;
	message?: string;
	data?: unknown;
}

/**
 * GitHub Copilot CLI Output Parser Implementation
 *
 * Transforms Copilot CLI's ACP format into normalized ParsedEvents.
 */
export class CopilotCliOutputParser implements AgentOutputParser {
	readonly agentId: ToolType = 'copilot-cli';

	// Cached context window
	private contextWindow: number;
	private currentSessionId: string | null = null;

	constructor(model: string = 'claude-sonnet-4.5') {
		this.contextWindow = getModelContextWindow(model);
	}

	/**
	 * Parse a single JSON line from Copilot CLI ACP output
	 */
	parseJsonLine(line: string): ParsedEvent | null {
		if (!line.trim()) {
			return null;
		}

		try {
			const msg: AcpMessage = JSON.parse(line);
			return this.transformMessage(msg);
		} catch {
			// Not valid JSON - return as raw text event
			return {
				type: 'text',
				text: line,
				raw: line,
			};
		}
	}

	/**
	 * Transform a parsed ACP message into a normalized ParsedEvent
	 */
	private transformMessage(msg: AcpMessage): ParsedEvent {
		// Handle JSON-RPC error responses
		if (msg.error) {
			return {
				type: 'error',
				text: msg.error.message || 'Unknown error',
				raw: msg,
			};
		}

		// Handle JSON-RPC result (completion)
		if (msg.result) {
			return this.transformResult(msg.result, msg);
		}

		// Handle JSON-RPC method calls (streaming events)
		if (msg.method) {
			return this.transformMethod(msg.method, msg.params, msg);
		}

		// Default: preserve as system event
		return {
			type: 'system',
			raw: msg,
		};
	}

	/**
	 * Transform a JSON-RPC result into a ParsedEvent
	 */
	private transformResult(result: AcpResult, msg: AcpMessage): ParsedEvent {
		// Store session ID if present
		if (result.sessionId) {
			this.currentSessionId = result.sessionId;
		}

		// If result contains message content, it's the final response
		if (result.message?.content || result.output) {
			const event: ParsedEvent = {
				type: 'result',
				sessionId: this.currentSessionId || result.sessionId,
				text: result.message?.content || result.output || '',
				raw: msg,
			};

			// Add usage if present
			if (result.usage) {
				event.usage = {
					inputTokens: result.usage.inputTokens || 0,
					outputTokens: result.usage.outputTokens || 0,
					contextWindow: this.contextWindow,
				};
			}

			return event;
		}

		// Session initialization result
		if (result.sessionId && !result.message && !result.output) {
			return {
				type: 'init',
				sessionId: result.sessionId,
				raw: msg,
			};
		}

		return {
			type: 'system',
			raw: msg,
		};
	}

	/**
	 * Transform a JSON-RPC method call into a ParsedEvent
	 */
	private transformMethod(
		method: string,
		params: AcpParams | undefined,
		msg: AcpMessage
	): ParsedEvent {
		// Store session ID if present
		if (params?.sessionId) {
			this.currentSessionId = params.sessionId;
		}

		// Handle different ACP methods
		switch (method) {
			// Session/agent initialization
			case 'agent/initialize':
			case 'session/start':
				return {
					type: 'init',
					sessionId: params?.sessionId || this.currentSessionId || undefined,
					raw: msg,
				};

			// Streaming text delta
			case 'agent/completion/stream':
			case 'agent/delta':
			case 'textDocument/delta':
				if (params?.delta?.text) {
					return {
						type: 'text',
						text: params.delta.text,
						isPartial: true,
						raw: msg,
					};
				}
				// Tool call within delta
				if (params?.delta?.toolCall) {
					return {
						type: 'tool_use',
						toolName: params.delta.toolCall.name,
						toolState: {
							status: 'running',
							input: params.delta.toolCall.arguments,
						},
						raw: msg,
					};
				}
				break;

			// Tool usage events
			case 'agent/tool/start':
			case 'tool/start':
				return {
					type: 'tool_use',
					toolName: params?.tool?.name,
					toolState: {
						status: 'running',
						input: params?.tool?.input,
					},
					raw: msg,
				};

			case 'agent/tool/complete':
			case 'tool/complete':
				return {
					type: 'tool_use',
					toolName: params?.tool?.name,
					toolState: {
						status: params?.tool?.status || 'completed',
						output: params?.tool?.output,
					},
					raw: msg,
				};

			// Message/content events
			case 'agent/message':
			case 'message':
				if (params?.message?.content) {
					return {
						type: 'text',
						text: params.message.content,
						raw: msg,
					};
				}
				break;

			// Intent updates (what the agent is working on)
			case 'agent/intent':
			case 'intent':
				// Intent is informational, treat as system event
				return {
					type: 'system',
					raw: msg,
				};
		}

		// Default: treat as system event
		return {
			type: 'system',
			raw: msg,
		};
	}

	/**
	 * Check if an event is a final result message
	 */
	isResultMessage(event: ParsedEvent): boolean {
		return event.type === 'result' && !!event.text;
	}

	/**
	 * Extract session ID from an event
	 */
	extractSessionId(event: ParsedEvent): string | null {
		return event.sessionId || this.currentSessionId || null;
	}

	/**
	 * Extract usage statistics from an event
	 */
	extractUsage(event: ParsedEvent): ParsedEvent['usage'] | null {
		return event.usage || null;
	}

	/**
	 * Extract slash commands from an event
	 * Copilot CLI has built-in slash commands but they're not discoverable via output
	 */
	extractSlashCommands(_event: ParsedEvent): string[] | null {
		// Return known Copilot CLI slash commands
		// These are documented in the help output
		return [
			'/help',
			'/model',
			'/login',
			'/logout',
			'/feedback',
			'/agent',
			'/delegate',
			'/experimental',
		];
	}

	/**
	 * Detect an error from a line of agent output
	 *
	 * Only detect errors from structured JSON error events, not from
	 * arbitrary text content to avoid false positives.
	 */
	detectErrorFromLine(line: string): AgentError | null {
		if (!line.trim()) {
			return null;
		}

		let errorText: string | null = null;
		try {
			const parsed: AcpMessage = JSON.parse(line);
			// Check for JSON-RPC error
			if (parsed.error?.message) {
				errorText = parsed.error.message;
			}
		} catch {
			// Not JSON - skip pattern matching
		}

		if (!errorText) {
			return null;
		}

		// Match against error patterns
		const patterns = getErrorPatterns(this.agentId);
		const match = matchErrorPattern(patterns, errorText);

		if (match) {
			return {
				type: match.type,
				message: match.message,
				recoverable: match.recoverable,
				agentId: this.agentId,
				timestamp: Date.now(),
				raw: {
					errorLine: line,
				},
			};
		}

		return null;
	}

	/**
	 * Detect an error from process exit information
	 */
	detectErrorFromExit(exitCode: number, stderr: string, stdout: string): AgentError | null {
		// Exit code 0 is success
		if (exitCode === 0) {
			return null;
		}

		// Check stderr and stdout for error patterns
		const combined = `${stderr}\n${stdout}`;
		const patterns = getErrorPatterns(this.agentId);
		const match = matchErrorPattern(patterns, combined);

		if (match) {
			return {
				type: match.type,
				message: match.message,
				recoverable: match.recoverable,
				agentId: this.agentId,
				timestamp: Date.now(),
				raw: {
					exitCode,
					stderr,
					stdout,
				},
			};
		}

		// Non-zero exit with no recognized pattern - treat as crash
		return {
			type: 'agent_crashed',
			message: `Agent exited with code ${exitCode}`,
			recoverable: true,
			agentId: this.agentId,
			timestamp: Date.now(),
			raw: {
				exitCode,
				stderr,
				stdout,
			},
		};
	}
}
