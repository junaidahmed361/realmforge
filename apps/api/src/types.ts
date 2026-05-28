export type World = 'code' | 'runtime' | 'business' | 'user' | 'test' | 'knowledge';

export type EvidenceRef = {
  id: string;
  sourceType: 'code' | 'test' | 'metric' | 'doc' | 'issue' | 'pr' | 'log';
  uri: string;
  label: string;
  excerpt?: string;
  confidence: number;
};

export type Entity = {
  id: string;
  world: World;
  type: string;
  name: string;
  metadata: Record<string, unknown>;
  confidence: number;
  evidenceRefs: EvidenceRef[];
};

export type Trajectory = {
  id: string;
  name: string;
  description: string;
  expected_benefits: string[];
  risks: string[];
};
