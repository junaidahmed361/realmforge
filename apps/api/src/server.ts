import express from 'express';
import { z } from 'zod';
import { getProvider } from './providers/providers.js';

const app = express();
app.use(express.json({ limit: '1mb' }));

const IntentRequest = z.object({
  repo: z.object({
    provider: z.enum(['github', 'gitlab', 'bitbucket', 'gitea', 'local']),
    owner: z.string().optional(),
    repo: z.string(),
    ref: z.string().optional(),
    localPath: z.string().optional()
  }),
  intent: z.string().min(8),
  calibration: z.object({
    riskTolerance: z.number().min(0).max(1).default(0.4),
    confidenceThreshold: z.number().min(0).max(1).default(0.65)
  }).optional()
});

app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'worldforge-api', byocodeworld: true });
});

app.post('/repos/ingest', async (req, res) => {
  const parsed = IntentRequest.shape.repo.safeParse(req.body?.repo ?? req.body);
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

  const provider = getProvider(parsed.data.provider);
  const files = await provider.listFiles(parsed.data);
  const sampled = files.slice(0, 50);

  res.json({
    repo: parsed.data,
    file_count: files.length,
    sampled_files: sampled,
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
  const topFiles = files.filter((f) => /router|route|api|model|cache|queue|auth|session/i.test(f)).slice(0, 6);
  const impactFiles = (topFiles.length ? topFiles : files.slice(0, 6));

  const entities = impactFiles.map((f, idx) => ({
    id: `ent_${idx + 1}`,
    world: 'code',
    type: 'file',
    name: f,
    confidence: 0.65,
    evidenceRefs: [{
      id: `ev_${idx + 1}`,
      sourceType: 'code',
      uri: provider.buildEvidenceRef(repo, f, 1, 200),
      label: f,
      confidence: 0.72
    }]
  }));

  const workUnit = {
    id: `wu_${Date.now()}`,
    intent: { raw: intent },
    impactSurface: { entities },
    trajectories: [
      {
        id: 'traj_low_risk',
        name: 'Low-Risk Stabilization',
        description: 'Add observability and targeted refactors first.',
        expected_benefits: ['safer rollout', 'higher confidence'],
        risks: ['slower impact velocity']
      },
      {
        id: 'traj_high_impact',
        name: 'High-Impact Optimization',
        description: 'Aggressive optimization across hot paths.',
        expected_benefits: ['bigger latency/cost gains'],
        risks: ['larger blast radius']
      }
    ],
    simulationReports: [
      {
        trajectory_id: 'traj_low_risk',
        risk: 'low',
        confidence: Math.max(0.6, calibration?.confidenceThreshold ?? 0.65),
        expected_impact: { latency: '-8%..-15%', reliability: '+2%..+5%' }
      },
      {
        trajectory_id: 'traj_high_impact',
        risk: 'medium',
        confidence: 0.62,
        expected_impact: { latency: '-18%..-30%', cost: '-5%..-12%' }
      }
    ],
    calibrationProfile: {
      mode: 'enterprise_byocodeworld',
      constraints: {
        max_blast_radius: (calibration?.riskTolerance ?? 0.4) > 0.6 ? 'high' : 'medium',
        min_confidence: calibration?.confidenceThreshold ?? 0.65
      }
    },
    approvalStatus: 'needs_review'
  };

  res.json(workUnit);
});

const port = Number(process.env.PORT ?? 8787);
app.listen(port, () => {
  console.log(`worldforge api listening on :${port}`);
});
