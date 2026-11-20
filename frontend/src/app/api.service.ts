import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

/** Shared envelope used by every backend endpoint. */
export interface ApiResponse<T> {
  success: true;
  data: T;
  meta: { request_id: string; processing_time_ms: number };
}

export type AnswerKind = 'answer' | 'not_found' | 'off_topic' | 'small_talk';

/** An earlier message and who wrote it, so the model does not have to guess. */
export interface HistoryMessage {
  role: 'customer' | 'agent';
  text: string;
}

export interface ChatSource {
  article_id: string;
  title: string;
}

/** One fact in the answer: the article section it comes from and the exact words. */
export interface ChatCitation {
  article_id: string;
  title: string;
  heading: string;
  quote: string;
}

export interface ChatResult {
  question: string;
  answer: string;
  kind: AnswerKind;
  sources: ChatSource[];
  citations: ChatCitation[];
  search_query: string;
  rewrite_fallback: boolean;
  answer_fallback: boolean;
  grounding_failed: boolean;
  guardrail_actions: string[];
}

export interface ArticleSummary {
  id: string;
  title: string;
  category: string | null;
  chunk_count: number;
  updated_at: string;
}

export interface ArticleResult extends ArticleSummary {
  content: string;
  created_at: string;
}

export interface ArticleCreate {
  title: string;
  content: string;
  category?: string;
}

export interface Health {
  status: string;
  environment: string;
  embedding_model_loaded: boolean;
  reranker_loaded: boolean;
  vector_store: string;
  keyword_index_chunks: number;
  rewrite_provider: string;
}

/** Turn a failed request into one readable line, using the backend error envelope when present. */
export function describeError(error: unknown): string {
  if (error instanceof HttpErrorResponse) {
    if (error.status === 0 || error.status === 504) {
      return 'Cannot reach the support service. Check that the backend is running, then try again.';
    }
    const body = error.error?.error;
    if (body?.code === 'KB_EMPTY') {
      return 'There are no help articles yet. Add some on the Articles tab first.';
    }
    if (body?.code) {
      const details = Array.isArray(body.details)
        ? ' ' + body.details.map((d: { loc: string[]; msg: string }) => `${d.loc.slice(1).join('.')}: ${d.msg}`).join('; ')
        : '';
      return `${body.message}${details}`;
    }
    return `The request failed (HTTP ${error.status}).`;
  }
  return 'Something went wrong. Please try again.';
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly base = '/api/v1/support';

  health(): Observable<Health> {
    return this.http.get<ApiResponse<Health>>('/health').pipe(map((r) => r.data));
  }

  /** `history` is the earlier messages of this chat with their speaker, oldest first. */
  chat(question: string, history: HistoryMessage[]): Observable<ChatResult> {
    return this.http
      .post<ApiResponse<ChatResult>>(`${this.base}/chat`, { question, history })
      .pipe(map((r) => r.data));
  }

  listArticles(): Observable<ArticleSummary[]> {
    return this.http.get<ApiResponse<ArticleSummary[]>>(`${this.base}/articles`).pipe(map((r) => r.data));
  }

  createArticle(article: ArticleCreate): Observable<ArticleResult> {
    return this.http.post<ApiResponse<ArticleResult>>(`${this.base}/articles`, article).pipe(map((r) => r.data));
  }

  deleteArticle(id: string): Observable<unknown> {
    return this.http.delete(`${this.base}/articles/${encodeURIComponent(id)}`);
  }
}
