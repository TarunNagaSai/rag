export type Role = "user" | "assistant";

export interface Citation {
  id: string;
  /** Source document the snippet came from. */
  source: string;
  /** Retrieved text chunk used to ground the answer. */
  snippet: string;
  page?: number;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  /** Grounding sources surfaced beneath an assistant answer. */
  citations?: Citation[];
  /** Streaming / awaiting-backend state. */
  pending?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  updatedAt: number;
  /** Short preview of the last exchange. */
  preview?: string;
}

export type FileStatus = "ready" | "processing" | "error";

export interface RagFile {
  id: string;
  name: string;
  size: number;
  status: FileStatus;
  /** Number of indexed vector chunks. */
  chunks?: number;
  addedAt?: number;
}
