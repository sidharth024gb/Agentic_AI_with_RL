# Agent Framework

This folder contains the final reinforcement-learning framework used for the MSc experiments. It integrates an LLM planner and prerequisite-aware procedure tracker with a custom PPO executor connected to the stateful finance backend.

## Architecture

The final execution loop is:

```text
Task Goal + Initial Finance State
              ↓
         LLM Planner
              ↓
  Procedure + Prerequisites
              ↓
       Procedure Tracker
              ↓
     Current Recommendation
         ↙             ↘
 Guidance Input     Reward Guidance
         ↘             ↙
          PPO Executor
              ↓
       FinanceEnvironment
              ↓
           REST API
              ↓
       Finance Sandbox
              ↓
     State + Base Reward
              ↓
       Reward Processor
              ↓
          PPO Update
```

Important design rule:

> The LLM provides advisory procedural guidance. PPO remains the action-selection mechanism and can always choose any action in the discrete action space.

The LLM is therefore a procedural prior rather than a hard controller.

## Folder Structure

```text
agent_framework/
├── agents/
├── config/
├── environment/
├── llm/
├── memory/
├── models/
├── results/
├── tests/
├── training/
├── utils/
├── main.py
├── README.md
└── requirements.txt
```

## File Reference

### `agents/`

| File | Responsibility |
| --- | --- |
| `base_agent.py` | Common agent interface, shared lifecycle behaviour and random-seed setup. |
| `ppo_agent.py` | Custom PPO implementation: action selection, updates, entropy regularisation, gradient clipping, checkpointing and evaluation behaviour. |
| `llm_rl_agent.py` | Extends PPO with LLM planning and `INPUT`, `REWARD_SHAPING` and `INPUT_AND_REWARD` guidance. |

### `config/`

| File | Responsibility |
| --- | --- |
| `.env` | Local backend credentials, Ollama settings and experiment defaults. |
| `config.py` | Central configuration for backend endpoints, environment, PPO, training, evaluation, seeds and paths. |

### `environment/`

| File | Responsibility |
| --- | --- |
| `action_space.py` | Defines the discrete finance actions and action IDs. |
| `api_client.py` | Authenticated REST client for sandbox, invoice, supplier, approval, payment, report and episode endpoints. |
| `finance_env.py` | Main RL environment: reset, step, action execution, state refresh, completion detection and backend step logging. |
| `procedure_tracker.py` | Maintains the LLM procedure, prerequisite graph, procedural completion state and current eligible recommendation. |
| `reward_processor.py` | Combines environment reward, small local efficiency penalties, completion reward and optional guidance bonus. |
| `state_encoder.py` | Converts finance JSON state into the numeric PPO observation. |

The finance observation contains 13 base values.

### `llm/`

| File | Responsibility |
| --- | --- |
| `cache.py` | Reuses generated plans and tracks cache hits/misses. |
| `parser.py` | Validates and converts LLM output into action IDs and prerequisite mappings. |
| `planner.py` | Calls Ollama/Llama 3 and manages high-level plan generation. |
| `prompts.py` | Structured finance-planning instructions and output format. |

### `memory/`

| File | Responsibility |
| --- | --- |
| `rollout_buffer.py` | Stores on-policy transitions and computes returns/advantages for PPO. |

### `models/`

| File | Responsibility |
| --- | --- |
| `policy_network.py` | PPO actor network that outputs the categorical action distribution. |
| `value_network.py` | PPO critic network that estimates state value. |

### `training/`

| File | Responsibility |
| --- | --- |
| `train.py` | Runs training episodes and captures per-episode/LLM/PPO metadata. |
| `evaluate.py` | Runs deterministic evaluation using the trained policy. |
| `experiment_suite.py` | Implements the complete condition × seed orchestration and combined reporting; it is invoked by `main.py run-all`. |

### `utils/`

| File | Responsibility |
| --- | --- |
| `logger.py` | Console and file logging. |
| `metrics.py` | Per-run episode, action, PPO-update, guidance and configuration metrics. |
| `visualization.py` | Per-run plots. |
| `comparison_metrics.py` | Aggregates experiment outputs across conditions/seeds. |
| `comparison_visualization.py` | Produces per-seed and all-seed comparison plots. |

### `tests/`

| File | Responsibility |
| --- | --- |
| `test_environment.py` | Finance-environment, transition, reward and backend-integration tests. |
| `test_llm.py` | Planner, parser, cache and guidance tests. |

### `results/`

```text
results/
├── graphs/
├── llm_cache/
├── logs/
├── metrics/
└── models/
```

- `graphs/` — individual and combined figures/CSV matrices.
- `llm_cache/` — cached structured LLM plans.
- `logs/` — training logs, manifests and raw run records.
- `metrics/` — Excel workbooks and raw metric exports.
- `models/` — periodic and final `.pt` checkpoints.

## Guidance Modes

| Mode | Policy observation | Guidance reward | Purpose |
| --- | --- | --- | --- |
| `NONE` | 13-value finance state | No | PPO baseline. |
| `INPUT` | 13-value state + 8-value one-hot guidance vector | No | Tests guidance as additional policy information. |
| `REWARD_SHAPING` | 13-value finance state | Yes | Tests procedural guidance only through reward shaping. |
| `INPUT_AND_REWARD` | 21-value guided observation | Yes | Tests both integration mechanisms together. |

Guidance never removes an action.

A procedural step becomes complete in the tracker only when:

1. the environment action succeeds; and
2. the LLM-defined prerequisites for that step are already procedurally complete.

This lets PPO execute actions out of order while ensuring the guidance system does not incorrectly treat an out-of-order action as satisfying a later procedural requirement.

## Reward Semantics

The final Python reward layer distinguishes:

- **base reward** — environment/task progress;
- **guidance bonus** — optional reward for following an eligible LLM recommendation;
- **completion bonus** — terminal task success;
- **valid no-op** — neutral;
- **repeated no-progress action** — very small efficiency penalty;
- **guided repeated no-op** — repeat penalty can be suppressed when the action is the current valid LLM recommendation;
- **environment/system error** — recorded separately and not treated as an ordinary PPO mistake.

This separation prevents infrastructure failures from corrupting the policy-learning signal.

## Dependencies

Install exact versions from:

```text
agent_framework/requirements.txt
```

Main groups include:

- `torch` — custom PPO actor/critic and optimisation;
- `numpy` — observations and numerical operations;
- `requests` — finance backend HTTP client;
- `ollama` — local Llama 3 planner access;
- `python-dotenv` — local environment configuration;
- `pandas` — experiment data processing;
- `openpyxl` — Excel output;
- `matplotlib` / `seaborn` — figures;
- `pytest` — tests.

**Stable-Baselines3 and Gymnasium are used by the earlier `POC/`, not by the final PPO implementation in this folder.**

## Environment Configuration

Create:

```text
agent_framework/config/.env
```

Use the exact variable names consumed by `config/config.py`. The project configuration has used values equivalent to:

```dotenv
BASE_URL=http://localhost:5000/api
EMAIL=<agent-email>
PASSWORD=<agent-password>

MAX_STEPS_PER_EPISODE=20
RANDOM_SEED=42

TOTAL_EPISODES=1000
EVALUATION_EPISODES=100

GAMMA=0.99
GAE_LAMBDA=0.95
LEARNING_RATE=0.0003
BATCH_SIZE=64
UPDATE_INTERVAL=256
PPO_EPOCHS=10
CLIP_EPSILON=0.2
HIDDEN_NEURON_SIZE=256
ENTROPY_COEF=0.01
MAX_GRAD_NORM=0.5

OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3
OLLAMA_TIMEOUT=120
LLM_TEMPERATURE=0.1
LLM_USE_CACHE=true

GUIDANCE_BONUS=1.0
```

Do not commit real credentials.

## Installation

From the repository root:

```bash
python -m venv .venv-agent
```

Activate it, then:

```bash
python -m pip install --upgrade pip
python -m pip install -r agent_framework/requirements.txt
```

## Services Required

Before training:

1. MongoDB must be available.
2. `backend_server` must be running.
3. Ollama must be running for the three LLM-guided conditions.

Backend:

```bash
cd backend_server
npm start
```

Ollama:

```bash
ollama serve
```

## Tests

```bash
cd agent_framework
python -m pytest tests
```

Run tests before an expensive multi-seed experiment.

## Running Individual Experiments

### PPO baseline

```bash
python main.py train --agent ppo --episodes 1000 --eval-episodes 100
```

Runtime behaviour:

```text
AGENT_TYPE = RL
GUIDANCE_MODE = NONE
GUIDANCE_BONUS = 0
```

### LLM Input

```bash
python main.py train --agent llm_rl --guidance-mode input --episodes 1000 --eval-episodes 100
```

### LLM Reward Shaping

```bash
python main.py train --agent llm_rl --guidance-mode reward_shaping --episodes 1000 --eval-episodes 100
```

### LLM Input + Reward

```bash
python main.py train --agent llm_rl --guidance-mode input_and_reward --guidance-bonus 1.0 --episodes 1000 --eval-episodes 100
```

### Single-run seed override

`--seed` changes the base random seed for one run without editing the configuration file:

```bash
python main.py train   --agent ppo   --seed 42   --episodes 1000   --eval-episodes 100
```

For a guided condition:

```bash
python main.py train   --agent llm_rl   --guidance-mode input_and_reward   --guidance-bonus 1.0   --seed 42   --experiment-name llm_input_reward_seed_42
```


## `main.py` Command-Line Interface

`main.py` is the supported command-line entry point for both individual runs and the complete experiment suite.

### `train`

```text
python main.py train --agent {ppo,llm_rl} [options]
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `--agent {ppo,llm_rl}` | Required. Select baseline PPO or LLM-enhanced PPO. |
| `--episodes N` | Override `TOTAL_EPISODES`. Must be greater than zero. |
| `--eval-episodes N` | Override `EVALUATION_EPISODES`. Must be greater than zero. |
| `--experiment-name NAME` | Optional experiment prefix. Timestamp/run suffixes are still generated by the logging layer. |
| `--seed SEED` | Override `config.environment.RANDOM_SEED` for this single run. |
| `--guidance-mode MODE` | `none`, `input`, `reward_shaping`, or `input_and_reward`. Baseline PPO always uses `NONE`. |
| `--guidance-bonus VALUE` | Non-negative bonus for followed procedural guidance. Relevant to reward-guided modes. |

### `run-all`

```text
python main.py run-all [options]
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `--seeds S1 S2 ...` | Override the configured experiment seed list. Values must be non-negative and unique. |
| `--episodes N` | Training episodes for every condition/seed. |
| `--eval-episodes N` | Deterministic evaluation episodes for every condition/seed. |
| `--guidance-bonus VALUE` | Bonus used by `REWARD_SHAPING` and `INPUT_AND_REWARD`. |
| `--suite-name NAME` | Name of the combined result folder. |
| `--continue-on-error` | Continue later experiment runs after one run fails. |

If no subcommand is supplied, `main.py` prints the command help.

A keyboard interrupt exits with the conventional interrupt status after reporting that training was interrupted. Other experiment exceptions are reported and re-raised so failures are not silently hidden.

## Running the Full Final Experiment Suite

The current CLI exposes the suite through `main.py`:

```bash
python main.py run-all
```

`run-all` executes all four conditions for every configured seed and then builds the combined reports.

The final dissertation suite used:

```text
seeds = [10, 24, 33, 42, 50]
```

with 1000 training episodes and 100 deterministic evaluation episodes for each condition/seed combination.

### Use configured defaults

```bash
python main.py run-all
```

### Override the seed list

```bash
python main.py run-all --seeds 10 24 33 42 50
```

### Override episodes

```bash
python main.py run-all   --episodes 1000   --eval-episodes 100
```

### Override the guidance bonus

```bash
python main.py run-all --guidance-bonus 1.0
```

The bonus is used by the `REWARD_SHAPING` and `INPUT_AND_REWARD` conditions.

### Set the suite folder name

```bash
python main.py run-all --suite-name final_dissertation
```

If `--suite-name` is omitted, the configured `EXPERIMENT_SUITE_NAME` is used.

### Continue if one run fails

```bash
python main.py run-all --continue-on-error
```

Without this flag, an experiment failure stops the suite. With it, later conditions/seeds are still attempted and the failed run can be investigated separately.

### Complete explicit command

```bash
python main.py run-all   --seeds 10 24 33 42 50   --episodes 1000   --eval-episodes 100   --guidance-bonus 1.0   --suite-name final_dissertation
```

Before launching the suite, verify:

- backend URL;
- credentials;
- MongoDB;
- Ollama model;
- guidance bonus;
- total/evaluation episodes;
- seed list;
- checkpoint interval;
- result directories.

## Reproducibility

The Python environment derives an episode seed and passes it to the Node.js sandbox reset. The backend uses seeded scenario generation rather than unseeded `Math.random()` for experiment-critical reset behaviour.

For fair comparisons:

- use the same seed set for all conditions;
- keep PPO hyperparameters fixed;
- keep environment/reward rules fixed;
- keep Llama model/configuration fixed;
- do not alter MongoDB during a run;
- start every episode through `FinanceEnvironment.reset()`;
- use separate deterministic evaluation scenarios/seeds;
- preserve run manifests and raw outputs.

## Metrics Captured

The reporting layer preserves:

- training and evaluation success;
- total/base/guidance/completion reward;
- episode length;
- useful, failed and no-op actions;
- termination reason;
- environment errors;
- action frequency;
- procedure attempts/followed/adherence;
- per-procedure-action behaviour;
- LLM plan and prerequisites;
- cache hits/misses;
- planning time and LLM latency;
- policy loss;
- clipped policy loss;
- value loss;
- entropy / normalised entropy;
- approximate KL;
- clip fraction;
- explained variance;
- policy/value gradient norms;
- checkpoints;
- wall-clock time.

Raw backend and local Python records are preserved so additional metrics can be derived without retraining.

## Interpretation

Training uses stochastic PPO action sampling; evaluation uses the configured deterministic policy.

Therefore:

- high training success does not automatically imply reliable deterministic execution;
- success rate is more important than shaped reward alone;
- reward should be interpreted alongside guidance bonus;
- evaluation consistency across independent seeds is a key reliability measure;
- procedure adherence measures whether PPO followed the recommendation, not whether it was forced to do so.
