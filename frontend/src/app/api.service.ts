import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

/** Shared envelope used by every backend endpoint. */
export interface ApiResponse<T> {
  success: true;
  data: T;
  meta: { request_id: string; processing_time_ms: number };
}

export type QueryMode = 'rewrite' | 'question_only' | 'full_history';
export type RetrievalMode = 'hybrid' | 'vector' | 'keyword';

export interface SearchRequest {
  question: string;
  history: string[];
  top_k: number;
  query_mode: QueryMode;
  retrieval: RetrievalMode;
  rerank: boolean;
  include_debug: boolean;
}

export interface StageDebug {
  vector_rank: number | null;
  keyword_rank: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
}

export interface RetrievedArticle {
  article_id: string;
  title: string;
  category: string | null;
  score: number;
  snippet: string;
  debug: StageDebug | null;
}

export interface SearchResult {
  question: string;
  search_query: string;
  query_mode: QueryMode;
  retrieval: RetrievalMode;
  rerank: boolean;
  needs_retrieval: boolean;
  rewrite_source: string;
  rewrite_fallback: boolean;
  rerank_fallback: boolean;
  score_type: string;
  results: RetrievedArticle[];
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
    if (error.status === 0) {
      return 'Cannot reach the backend. Is it running on http://127.0.0.1:8000?';
    }
    const body = error.error?.error;
    if (body?.code) {
      const details = Array.isArray(body.details)
        ? ' ' + body.details.map((d: { loc: string[]; msg: string }) => `${d.loc.slice(1).join('.')}: ${d.msg}`).join('; ')
        : '';
      return `${body.code} (HTTP ${error.status}): ${body.message}${details}`;
    }
    return `HTTP ${error.status}: ${error.statusText}`;
  }
  return 'Unexpected error.';
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly base = '/api/v1/support';

  health(): Observable<Health> {
    return this.http.get<ApiResponse<Health>>('/health').pipe(map((r) => r.data));
  }

  search(request: SearchRequest): Observable<ApiResponse<SearchResult>> {
    return this.http.post<ApiResponse<SearchResult>>(`${this.base}/search`, request);
  }

  listArticles(): Observable<ArticleSummary[]> {
    return this.http.get<ApiResponse<ArticleSummary[]>>(`${this.base}/articles`).pipe(map((r) => r.data));
  }

  getArticle(id: string): Observable<ArticleResult> {
    return this.http.get<ApiResponse<ArticleResult>>(`${this.base}/articles/${encodeURIComponent(id)}`).pipe(map((r) => r.data));
  }

  createArticle(article: ArticleCreate): Observable<ArticleResult> {
    return this.http.post<ApiResponse<ArticleResult>>(`${this.base}/articles`, article).pipe(map((r) => r.data));
  }

  deleteArticle(id: string): Observable<unknown> {
    return this.http.delete(`${this.base}/articles/${encodeURIComponent(id)}`);
  }
}
