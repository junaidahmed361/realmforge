import fs from 'node:fs/promises';
import path from 'node:path';
import type { CodeWorldProvider, RepoRef } from './base.js';

class UrlProvider implements CodeWorldProvider {
  constructor(private readonly root: string) {}
  async listFiles(_repo: RepoRef): Promise<string[]> {
    return ['README.md'];
  }
  async readFile(_repo: RepoRef, _p: string): Promise<string> {
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
