import { ArticleCreate } from './api.service';

export const ACCEPTED_EXTENSIONS = ['.md', '.markdown', '.txt'];
export const MAX_CONTENT_CHARS = 20000;

/**
 * Turn the text of an uploaded file into an article draft.
 *
 * Optional front matter is honoured (the format of data/kb/*.md):
 *   ---
 *   title: Returns for Outerwear
 *   category: returns
 *   ---
 * Without it, the title comes from the first heading, or else from the file name.
 */
export function parseArticleFile(fileName: string, text: string): ArticleCreate {
  const normalized = text.replace(/\r\n/g, '\n');
  let content = normalized;
  let title = '';
  let category = '';

  const frontMatter = /^---\n([\s\S]*?)\n---\n?/.exec(normalized);
  if (frontMatter) {
    content = normalized.slice(frontMatter[0].length).replace(/^\n+/, '');
    for (const line of frontMatter[1].split('\n')) {
      const separator = line.indexOf(':');
      if (separator === -1) continue;
      const key = line.slice(0, separator).trim().toLowerCase();
      const value = line.slice(separator + 1).trim().replace(/^["']|["']$/g, '');
      if (key === 'title') title = value;
      if (key === 'category') category = value;
    }
  }

  if (!title) {
    const heading = /^#\s+(.+)$/m.exec(content);
    title = heading ? heading[1].trim() : fileName.replace(/\.[^.]+$/, '').replace(/[-_]+/g, ' ').trim();
  }

  return { title, content, category: category.toLowerCase() };
}

export function hasAcceptedExtension(fileName: string): boolean {
  const lower = fileName.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((extension) => lower.endsWith(extension));
}
