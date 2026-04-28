# RealmForge

RealmForge is a domain-agnostic framework for building energy-based world models with bring-your-own-concept (BYOC) domain overlays.

Name origin and terminology
- RealmForge = the framework/repo (this project)
- Realm = a domain-specific scaffold (examples: HealthRealm, FinanceRealm)
- World = the simulated environment, rules, entities, and dynamics
- Campaign = a goal-driven journey/problem inside the world
- Scenario = a specific setup or counterfactual within that campaign
- Timeline / Run = one sampled rollout of what could happen

This vocabulary lets multiple domains coexist on one backbone while preserving domain-specific semantics.

Core architecture
- wm_app/: shared backbone (encoding, transition, energy graph, rollout interfaces, serving/eval utilities)
- domains/: realm overlays that define domain schema, mappings, actions, concepts, and configs
- configs/backbone/: default backbone behavior

Current realms
- domains/clinical_hf: first implementation track (education/research use only)
- domains/financial_template: starter scaffold for non-clinical adaptation
- domains/_realm_template: boilerplate to create new realms quickly

Quick start
1) Clone and install
   - python3 -m venv .venv
   - source .venv/bin/activate
   - pip install -e .[dev]

2) Run quality gates
   - pre-commit install
   - pre-commit run --all-files
   - make ci

3) Run tests
   - pytest -q

Create a new realm (boilerplate)
1) Copy template
   - cp -R domains/_realm_template domains/<your_realm>

2) Update realm config
   - edit domains/<your_realm>/configs/domain.yaml
   - set entity key/time key, actions, outcomes, concept tags

3) Add mappings and pipeline stubs
   - edit domains/<your_realm>/mappings/schema.md
   - edit domains/<your_realm>/pipelines/README.md
   - edit domains/<your_realm>/concepts/seed_concepts.yaml

4) Load merged config in code
   - from wm_app.core.config_loader import load_domain_config
   - cfg = load_domain_config("domains/<your_realm>/configs/domain.yaml")

Minimal startup boilerplate for any realm
- Define observed variables (o_t)
- Define latent variables (z_t)
- Define action variables (a_t)
- Define outcomes (y_t)
- Define plausibility constraints/factors (E_i)
- Train: encoder -> JEPA -> transition -> energy -> outcome heads
- Simulate: campaign -> scenario -> timeline/run sampling

Safety note
Clinical realms are for retrospective research and medical education simulation only.
Do not present outputs as treatment recommendations.
