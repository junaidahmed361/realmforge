import fs from 'node:fs/promises';
import path from 'node:path';

import type { CodeWorldProvider, RepoRef } from './base.js';

type GitHubTreeNode = { path: string; type: 'blob' | 'tree' };

class UrlProvider implements CodeWorldProvider {
  constructor(private readonly root: string) {}

  async listFiles(repo: RepoRef): Promise<string[]> {
    if (this.root.includes('github.com') && repo.owner) {
      try {
        const ref = repo.ref ?? 'main';
        const apiUrl = `https://api.github.com/repos/${repo.owner}/${repo.repo}/git/trees/${encodeURIComponent(ref)}?recursive=1`;
        const response = await fetch(apiUrl, {
          headers: {
            'Accept': 'application/vnd.github+json',
            ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {})
          }
        });
        if (response.ok) {
          const data = (await response.json()) as { tree?: GitHubTreeNode[] };
          return (data.tree ?? []).filter((n) => n.type === 'blob').map((n) => n.path);
        }
      } catch {
        // fallback below
      }
    }
    return ['README.md'];
  }

  async readFile(repo: RepoRef, filePath: string): Promise<string> {
    if (this.root.includes('github.com') && repo.owner) {
      const ref = repo.ref ?? 'main';
      const rawUrl = `https://raw.githubusercontent.com/${repo.owner}/${repo.repo}/${ref}/${filePath}`;
      const response = await fetch(rawUrl);
      if (response.ok) return response.text();
    }
    return 'placeholder';
  }

  buildEvidenceRef(repo: RepoRef, p: string, s: number, e: number): string {
    const owner = repo.owner ?? 'org';
    return `${this.root}/${owner}/${repo.repo}/blob/${repo.ref ?? 'main'}/${p}#L${s}-L${e}`;
  }
}

class LocalProvider implements CodeWorldProvider {
  async listFiles(repo: RepoRef): Promise<string[]> {
    const base = repo.localPath ?? repo.repo;
    const out: string[] = [];
    async function walk(dir: string, rel = ''): Promise<void> {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.name.startsWith('.git')) continue;
        const abs = path.join(dir, entry.name);
        const r = path.join(rel, entry.name);
        if (entry.isDirectory()) await walk(abs, r);
        else out.push(r.replaceAll('\\', '/'));
      }
    }
    await walk(base);
    return out;
  }

  async readFile(repo: RepoRef, p: string): Promise<string> {
    const base = repo.localPath ?? repo.repo;
    return fs.readFile(path.join(base, p), 'utf8');
  }

  buildEvidenceRef(repo: RepoRef, p: string, s: number, e: number): string {
    const base = repo.localPath ?? repo.repo;
    return `file://${path.join(base, p)}:L${s}-L${e}`;
  }
}

export function getProvider(kind: RepoRef['provider']): CodeWorldProvider {
  if (kind === 'local') return new LocalProvider();
  if (kind === 'gitlab') return new UrlProvider('https://gitlab.com');
  if (kind === 'bitbucket') return new UrlProvider('https://bitbucket.org');
  if (kind === 'gitea') return new UrlProvider('https://gitea.com');
  return new UrlProvider('https://github.com');
}
