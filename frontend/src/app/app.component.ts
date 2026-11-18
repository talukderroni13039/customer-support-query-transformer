import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  ApiService,
  ArticleCreate,
  ArticleSummary,
  Health,
  QueryMode,
  RetrievalMode,
  SearchResult,
  describeError,
} from './api.service';
import { ACCEPTED_EXTENSIONS, MAX_CONTENT_CHARS, hasAcceptedExtension, parseArticleFile } from './article-file';

interface Example {
  label: string;
  question: string;
  history: string[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  private readonly api = inject(ApiService);

  readonly tab = signal<'search' | 'articles'>('search');
  readonly health = signal<Health | null>(null);
  readonly healthError = signal('');

  // --- search ---------------------------------------------------------------------------
  question = '';
  historyText = '';
  queryMode: QueryMode = 'rewrite';
  retrieval: RetrievalMode = 'hybrid';
  rerank = true;
  includeDebug = true;
  topK = 3;

  readonly searching = signal(false);
  readonly searchError = signal('');
  readonly result = signal<SearchResult | null>(null);
  readonly searchMs = signal(0);

  readonly examples: Example[] = [
    { label: 'Shipping to Canada', question: 'How long does shipping to Canada take?', history: [] },
    { label: 'Exact product code', question: 'Is TB-PRO-23 machine washable?', history: [] },
    { label: 'Paraphrase', question: "I can't remember my login details.", history: [] },
    { label: 'Warranty', question: 'What is the warranty on backpacks?', history: [] },
    {
      label: 'Follow-up (needs history)',
      question: 'No, I meant the backpack, not the tent.',
      history: ["What's the warranty on your tents?", 'Basecamp tents are covered for five years.'],
    },
  ];

  // --- articles -------------------------------------------------------------------------
  readonly acceptedExtensions = ACCEPTED_EXTENSIONS.join(',');
  readonly maxContentChars = MAX_CONTENT_CHARS;
  readonly articles = signal<ArticleSummary[]>([]);
  readonly articlesError = signal('');
  readonly uploadMessage = signal('');
  readonly uploadError = signal('');
  readonly uploading = signal(false);
  readonly draftFileName = signal('');
  draft: ArticleCreate | null = null;

  ngOnInit(): void {
    this.refreshHealth();
    this.loadArticles();
  }

  refreshHealth(): void {
    this.api.health().subscribe({
      next: (health) => {
        this.health.set(health);
        this.healthError.set('');
      },
      error: (error) => {
        this.health.set(null);
        this.healthError.set(describeError(error));
      },
    });
  }

  // --- search ---------------------------------------------------------------------------

  /** History is typed one message per line, oldest first. Blank lines are ignored. */
  get history(): string[] {
    return this.historyText
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
  }

  /** True when this request will ask Claude to rewrite the question (uses the call budget). */
  get willCallRewriter(): boolean {
    return this.queryMode === 'rewrite' && this.history.length > 0;
  }

  useExample(example: Example): void {
    this.question = example.question;
    this.historyText = example.history.join('\n');
    this.runSearch();
  }

  runSearch(): void {
    const question = this.question.trim();
    if (!question || this.searching()) return;
    this.searching.set(true);
    this.searchError.set('');
    this.api
      .search({
        question,
        history: this.history,
        top_k: this.topK,
        query_mode: this.queryMode,
        retrieval: this.retrieval,
        rerank: this.rerank,
        include_debug: this.includeDebug,
      })
      .subscribe({
        next: (response) => {
          this.result.set(response.data);
          this.searchMs.set(Math.round(response.meta.processing_time_ms));
          this.searching.set(false);
        },
        error: (error) => {
          this.result.set(null);
          this.searchError.set(describeError(error));
          this.searching.set(false);
        },
      });
  }

  // --- articles -------------------------------------------------------------------------

  loadArticles(): void {
    this.api.listArticles().subscribe({
      next: (articles) => {
        this.articles.set(articles);
        this.articlesError.set('');
      },
      error: (error) => this.articlesError.set(describeError(error)),
    });
  }

  async onFileSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = ''; // allow picking the same file again
    this.uploadMessage.set('');
    this.uploadError.set('');
    this.draft = null;
    if (!file) return;

    if (!hasAcceptedExtension(file.name)) {
      this.uploadError.set(
        `"${file.name}" is not supported. Upload a text file (${ACCEPTED_EXTENSIONS.join(', ')}). ` +
          'PDF, Word and Excel files must be converted to text first.',
      );
      return;
    }
    const text = await file.text();
    if (text.includes('\u0000')) {
      this.uploadError.set(`"${file.name}" looks like a binary file, not text.`);
      return;
    }
    this.draft = parseArticleFile(file.name, text);
    this.draftFileName.set(file.name);
  }

  get draftProblem(): string {
    const draft = this.draft;
    if (!draft) return '';
    if (draft.title.trim().length < 3) return 'Title needs at least 3 characters.';
    if (draft.title.trim().length > 200) return 'Title can be at most 200 characters.';
    if (draft.content.trim().length < 20) return 'Content needs at least 20 characters.';
    if (draft.content.length > MAX_CONTENT_CHARS) {
      return `Content is ${draft.content.length} characters; the limit is ${MAX_CONTENT_CHARS}. Split the file.`;
    }
    if (draft.category && !/^[a-z0-9-]{1,50}$/.test(draft.category)) {
      return 'Category may only contain lowercase letters, digits and hyphens (max 50).';
    }
    return '';
  }

  saveDraft(): void {
    const draft = this.draft;
    if (!draft || this.draftProblem || this.uploading()) return;
    this.uploading.set(true);
    this.uploadError.set('');
    const payload: ArticleCreate = { title: draft.title.trim(), content: draft.content };
    if (draft.category) payload.category = draft.category;
    this.api.createArticle(payload).subscribe({
      next: (article) => {
        this.uploadMessage.set(`Stored "${article.title}" as ${article.id} (${article.chunk_count} chunks). It is searchable now.`);
        this.draft = null;
        this.uploading.set(false);
        this.loadArticles();
        this.refreshHealth();
      },
      error: (error) => {
        this.uploadError.set(describeError(error));
        this.uploading.set(false);
      },
    });
  }

  cancelDraft(): void {
    this.draft = null;
  }

  deleteArticle(article: ArticleSummary): void {
    if (!confirm(`Delete "${article.title}" (${article.id})? This removes it from search.`)) return;
    this.api.deleteArticle(article.id).subscribe({
      next: () => {
        this.uploadMessage.set(`Deleted ${article.id}.`);
        this.loadArticles();
        this.refreshHealth();
      },
      error: (error) => this.articlesError.set(describeError(error)),
    });
  }
}
