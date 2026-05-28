import fs from 'node:fs/promises';
import path from 'node:path';

import express from 'express';
import { z } from 'zod';

import { getProvider } from './providers/providers.js';

const app = express();
app.use(express.json({ limit: '2mb' }));

const IntentRequest = z.object({
  repo: z.object({
    provider: z.enum(['github', 'gitlab', 'bitbucket', 'gitea', 'local']),
    owner: z.string().optional(),
    repo: z.string(),
    ref: z.string().optional(),
    localPath: z.string().optional()
  }),
  intent: z.string().min(8),
  calibration: z
    .object({
      riskTolerance: z.number().min(0).max(1).default(0.4),
      confidenceThreshold: z.number().min(0).max(1).default(0.65)
    })
    .optional()
});

function parseBusinessGoal(intent: string) {
  const text = intent.toLowerCase();
  const isTfToTorch = /tensorflow/.test(text) && /pytorch/.test(text) && /(migrat|churn|switch|adopt)/.test(text);
  const onboarding = /onboard|developer experience|dx|getting started|tutorial/.test(text);
  const reliability = /reliability|uptime|stability|error|crash|regression/.test(text);
  const latency = /latency|p95|p99|inference speed|throughput/.test(text);
  const training = /train|training stability|convergence/.test(text);

  const heuristics = [] as string[];
  if (isTfToTorch) heuristics.push('tf_to_torch_migration_goal');
  if (onboarding) heuristics.push('onboarding_velocity');
  if (reliability) heuristics.push('reliability_protection');
  if (latency) heuristics.push('latency_guardrail');
  if (training) heuristics.push('training_stability_guardrail');

  return {
    isTfToTorch,
    heuristics,
    priority_mode: isTfToTorch ? 'migration_with_reliability' : 'generic_optimization'
  };
}

function filePriorityScore(file: string, parsed: ReturnType<typeof parseBusinessGoal>) {
  let score = 0;
  if (/readme|docs|tutorial|example|quickstart|beginner/i.test(file)) score += 3;
  if (/train|trainer|optim|checkpoint|distributed|fsdp|ddp/i.test(file)) score += 3;
  if (/inference|latency|benchmark|profil|serve|api|model/i.test(file)) score += 2;
  if (/tensorflow|tf/i.test(file)) score += parsed.isTfToTorch ? 5 : 1;
  if (/pytorch|torch/i.test(file)) score += parsed.isTfToTorch ? 4 : 1;
  return score;
}

app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'worldforge-api', byocodeworld: true });
});

app.post('/repos/ingest', async (req, res) => {
  const parsed = IntentRequest.shape.repo.safeParse(req.body?.repo ?? req.body);
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

  const provider = getProvider(parsed.data.provider);
  const files = await provider.listFiles(parsed.data);

  res.json({
    repo: parsed.data,
    file_count: files.length,
    sampled_files: files.slice(0, 50),
    provider: parsed.data.provider,
    backend_mode: 'open-source-byocodeworld'
  });
});

app.post('/intent', async (req, res) => {
  const parsed = IntentRequest.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

  const { repo, intent, calibration } = parsed.data;
  const provider = getProvider(repo.provider);
  const files = await provider.listFiles(repo);
  const goal = parseBusinessGoal(intent);

  const impactFiles = [...files]
    .map((f) => ({ f, score: filePriorityScore(f, goal) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 8)
    .map((x) => x.f);

  const entities = impactFiles.map((f, idx) => ({
    id: `ent_${idx + 1}`,
    world: 'code',
    type: /readme|docs|tutorial|example/i.test(f) ? 'doc_or_example' : 'file',
    name: f,
    confidence: 0.7,
    evidenceRefs: [
      {
        id: `ev_${idx + 1}`,
        sourceType: 'code',
        uri: provider.buildEvidenceRef(repo, f, 1, 250),
        label: f,
        confidence: 0.78
      }
    ]
  }));

  const workUnit = {
    id: `wu_${Date.now()}`,
    title: goal.isTfToTorch
      ? 'Increase TensorFlow→PyTorch migration adoption with reliability guardrails'
      : 'Optimization work unit',
    intent: {
      raw: intent,
      parsed: goal
    },
    impactSurface: { entities },
    trajectories: [
      {
        id: 'traj_onboarding_first',
        name: 'Onboarding-first migration',
        description: 'Prioritize docs/tutorial/examples + compatibility guidance to reduce migration friction.',
        expected_benefits: ['faster developer adoption', 'lower migration confusion'],
        risks: ['slower infra-level wins']
      },
      {
        id: 'traj_perf_reliability_first',
        name: 'Performance/reliability-first migration',
        description: 'Prioritize training/inference stability and benchmark evidence to persuade enterprise teams.',
        expected_benefits: ['stronger production confidence', 'better conversion for performance-sensitive teams'],
        risks: ['higher engineering complexity']
      }
    ],
    simulationReports: [
      {
        trajectory_id: 'traj_onboarding_first',
        risk: 'low',
        confidence: Math.max(0.62, calibration?.confidenceThreshold ?? 0.65),
        expected_impact: {
          migration_adoption: '+8%..+18%',
          time_to_first_success: '-20%..-35%'
        }
      },
      {
        trajectory_id: 'traj_perf_reliability_first',
        risk: 'medium',
        confidence: 0.66,
        expected_impact: {
          enterprise_conversion: '+6%..+14%',
          p95_latency: '-10%..-22%',
          training_failure_rate: '-8%..-15%'
        }
      }
    ],
    calibrationProfile: {
      mode: goal.priority_mode,
      constraints: {
        max_blast_radius: (calibration?.riskTolerance ?? 0.4) > 0.6 ? 'high' : 'medium',
        min_confidence: calibration?.confidenceThreshold ?? 0.65
      }
    },
    approvalStatus: 'needs_review'
  };

  res.json(workUnit);
});

app.post('/demo/export-workunit', async (req, res) => {
  const body = req.body as Record<string, unknown>;
  const workUnit = body.workUnit;
  if (!workUnit || typeof workUnit !== 'object') {
    return res.status(400).json({ error: 'workUnit object is required' });
  }

  const outputDir = path.resolve(process.cwd(), '../..', 'exports');
  await fs.mkdir(outputDir, { recursive: true });
  const filePath = path.join(outputDir, `workunit-${Date.now()}.json`);
  await fs.writeFile(filePath, JSON.stringify(workUnit, null, 2), 'utf8');

  res.json({ ok: true, filePath });
});

const port = Number(process.env.PORT ?? 8787);
app.listen(port, () => {
  console.log(`worldforge api listening on :${port}`);
});
