import { AfterViewChecked, Component, ElementRef, OnInit, ViewChild, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  AnswerKind,
  ApiService,
  ArticleCreate,
  ArticleSummary,
  ChatCitation,
  ChatSource,
  HistoryMessage,
  describeError,
} from './api.service';
import { ACCEPTED_EXTENSIONS, MAX_CONTENT_CHARS, hasAcceptedExtension, parseArticleFile } from './article-file';

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  kind?: AnswerKind | 'error';
  sources?: ChatSource[];
  citations?: ChatCitation[];
}

// Limits of the chat endpoint: each history item up to 2000 characters, 100 items.
const MAX_MESSAGE_CHARS = 2000;
const MAX_HISTORY_ITEMS = 40;

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit, AfterViewChecked {
  private readonly api = inject(ApiService);

  readonly tab = signal<'chat' | 'articles'>('chat');
  readonly online = signal<boolean | null>(null);

  // --- chat -----------------------------------------------------------------------------
  @ViewChild('thread') private thread?: ElementRef<HTMLElement>;
  private scrollPending = false;

  draftMessage = '';
  readonly messages = signal<ChatMessage[]>([]);
  readonly thinking = signal(false);
  readonly suggestions = [
    'How long does shipping to Canada take?',
    'Is TB-PRO-23 machine washable?',
    'What is the warranty on backpacks?',
    'I forgot my password',
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
    this.checkBackend();
    this.loadArticles();
  }

  ngAfterViewChecked(): void {
    if (this.scrollPending && this.thread) {
      const element = this.thread.nativeElement;
      element.scrollTop = element.scrollHeight;
      this.scrollPending = false;
    }
  }

  checkBackend(): void {
    this.api.health().subscribe({
      next: (health) => this.online.set(health.status === 'ok'),
      error: () => this.online.set(false),
    });
  }

  // --- chat -----------------------------------------------------------------------------

  onEnter(event: Event): void {
    const keyboard = event as KeyboardEvent;
    if (keyboard.shiftKey) return; // Shift+Enter makes a new line
    event.preventDefault();
    this.send();
  }

  ask(question: string): void {
    this.draftMessage = question;
    this.send();
  }

  send(): void {
    const question = this.draftMessage.trim().slice(0, MAX_MESSAGE_CHARS);
    if (!question || this.thinking()) return;

    // The conversation so far goes with every request: the backend stores nothing.
    const history: HistoryMessage[] = this.messages()
      .filter((message) => message.kind !== 'error')
      .map((message) => ({
        role: message.role === 'user' ? ('customer' as const) : ('agent' as const),
        text: message.text.slice(0, MAX_MESSAGE_CHARS),
      }))
      .slice(-MAX_HISTORY_ITEMS);

    this.messages.update((list) => [...list, { role: 'user', text: question }]);
    this.draftMessage = '';
    this.thinking.set(true);
    this.scrollPending = true;

    this.api.chat(question, history).subscribe({
      next: (result) => {
        this.messages.update((list) => [
          ...list,
          {
            role: 'assistant',
            text: result.answer,
            kind: result.kind,
            sources: result.sources,
            citations: result.citations,
          },
        ]);
        this.thinking.set(false);
        this.scrollPending = true;
      },
      error: (error) => {
        this.messages.update((list) => [...list, { role: 'assistant', text: describeError(error), kind: 'error' }]);
        this.thinking.set(false);
        this.scrollPending = true;
        this.checkBackend();
      },
    });
  }

  newChat(): void {
    if (this.thinking()) return;
    this.messages.set([]);
    this.draftMessage = '';
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
        this.uploadMessage.set(`Stored "${article.title}" (${article.id}). You can ask about it in the chat now.`);
        this.draft = null;
        this.uploading.set(false);
        this.loadArticles();
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
    if (!confirm(`Delete "${article.title}" (${article.id})? The chat will no longer use it.`)) return;
    this.api.deleteArticle(article.id).subscribe({
      next: () => {
        this.uploadMessage.set(`Deleted "${article.title}".`);
        this.loadArticles();
      },
      error: (error) => this.articlesError.set(describeError(error)),
    });
  }
}
