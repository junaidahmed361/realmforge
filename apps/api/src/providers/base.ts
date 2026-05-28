export type RepoRef = {
  provider: 'github' | 'gitlab' | 'bitbucket' | 'gitea' | 'local';
  owner?: string;
  repo: string;
  ref?: string;
  localPath?: string;
};

export interface CodeWorldProvider {
  listFiles(repo: RepoRef): Promise<string[]>;
  readFile(repo: RepoRef, path: string): Promise<string>;
  buildEvidenceRef(repo: RepoRef, path: string, lineStart: number, lineEnd: number): string;
}
