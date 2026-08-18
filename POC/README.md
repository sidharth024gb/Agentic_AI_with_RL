# GridWorld Proof of Concept

This folder contains the preliminary proof of concept (POC) developed before the final finance-agent framework.

Its purpose was to test the core feasibility question:

> Can LLM-derived guidance positively influence PPO learning, and what engineering cost does frequent LLM inference introduce?

The POC is exploratory design evidence. It should not be combined directly with the final finance multi-seed experiment results.

## POC Design

- Environment: custom **5 × 5 GridWorld**
- Environment API: **Gymnasium**
- RL algorithm: **Stable-Baselines3 PPO**
- LLM: **Llama 3 via Ollama**
- Baseline: PPO using the environment observation/reward only
- Guided variant: PPO receives an auxiliary positive reward when its selected movement matches the LLM movement recommendation
- LLM response cache: avoids repeated inference for previously seen states

The LLM-guided POC used frequent low-level guidance. It produced an encouraging reward trajectory but added substantial wall-clock delay. This motivated the final design in which the LLM generates a reusable high-level finance procedure instead of being called at every PPO state transition.

## Folder Structure

```text
POC/
├── logs/
│   ├── Baseline/
│   ├── LLM/
│   ├── Model_Performance.xlsx
│   └── Training Convergence.png
├── test_module/
│   ├── __init__.py
│   ├── test_env.py
│   └── test_ollama.py
├── .env
├── agent.py
├── config.yaml
├── environment.py
├── main.py
├── planner.py
├── README.md
├── requirements.txt
├── test.py
└── visualize.py
```

## File Reference

| Entry | Responsibility |
| --- | --- |
| `environment.py` | Implements the 5 × 5 GridWorld, Gymnasium observation/action spaces, reset, transition and reward behaviour. |
| `agent.py` | Creates/trains Stable-Baselines3 PPO for baseline and LLM-guided conditions. |
| `planner.py` | Calls Ollama/Llama 3 for movement guidance and parses the recommendation. |
| `main.py` | Main POC training/orchestration entry point. |
| `test.py` | Supporting trained-model evaluation/comparison. |
| `visualize.py` | Generates convergence/performance figures. |
| `config.yaml` | Grid, reward, PPO, training and LLM settings. |
| `.env` | Local model/service configuration. |
| `test_module/test_env.py` | Tests the GridWorld Gymnasium contract and transitions. |
| `test_module/test_ollama.py` | Tests Ollama connectivity and LLM responses. |
| `logs/Baseline/` | Baseline monitor data and saved Stable-Baselines3 model. |
| `logs/LLM/` | Guided monitor data and saved Stable-Baselines3 model. |
| `logs/Model_Performance.xlsx` | Comparison metrics. |
| `logs/Training Convergence.png` | Smoothed baseline/guided training comparison. |
| `requirements.txt` | Exact POC Python dependencies. |

## Dependencies

Use a separate virtual environment from the final `agent_framework`.

```bash
python -m venv .venv-poc
```

Activate it, then:

```bash
python -m pip install --upgrade pip
python -m pip install -r POC/requirements.txt
```

Key dependencies include:

### Reinforcement learning/environment

- `gymnasium` — Gymnasium environment interface and spaces.
- `stable-baselines3` — pre-built PPO implementation used by the POC.
- `shimmy` — Gym/Gymnasium compatibility support.

### LLM

- `ollama` — Python client for the local Llama 3 service.

### Data/configuration

- `numpy`
- `pandas`
- `openpyxl`
- `matplotlib`
- `PyYAML`
- `python-dotenv`
- `pytest`

The Stable-Baselines3 PPO in this folder is **not** the PPO used by the final finance experiments. The final framework implements PPO directly in PyTorch.

## Ollama Setup

Install Ollama and make the configured model available:

```bash
ollama pull llama3
ollama serve
```

Confirm that the model tag in `config.yaml`/`.env` matches the installed Ollama model.

## Configuration

Review:

```text
POC/config.yaml
POC/.env
```

before running.

Important values include:

- random seed;
- grid size;
- start/goal position;
- environment step and goal rewards;
- LLM guidance bonus;
- PPO hyperparameters;
- total training timesteps;
- Ollama model/base URL;
- logging/output paths.

Changing these values changes the experimental condition.

## Tests

From the repository root with `.venv-poc` active:

```bash
python -m pytest POC/test_module
```

`test_env.py` tests the environment itself.

`test_ollama.py` requires the Ollama service and Llama 3 model to be available.

## Running the POC

From the repository root:

```bash
cd POC
python main.py
```

Supporting evaluation/visualisation:

```bash
python test.py
python visualize.py
```

Check the active configuration before running the scripts so an old saved model/result is not accidentally overwritten or compared against a different configuration.

## Expected Outputs

```text
logs/
├── Baseline/
│   ├── monitor.csv
│   └── ppo_gridworld_Baseline.zip
├── LLM/
│   ├── monitor.csv
│   └── ppo_gridworld_LLM.zip
├── Model_Performance.xlsx
└── Training Convergence.png
```

The `.zip` models are Stable-Baselines3 model files.

## Interpretation

The POC should be interpreted as an engineering feasibility study.

It provided two important lessons:

1. **Potential benefit:** LLM-derived guidance showed evidence of maintaining a stronger PPO reward trajectory.
2. **Latency cost:** querying the LLM during low-level interaction added substantial wall-clock overhead.

Those observations motivated the final architecture:

```text
POC
LLM queried repeatedly during execution
              ↓
High inference overhead
              ↓
Final system
LLM generates high-level procedure once
              ↓
PPO performs low-level environment interaction
```

Therefore the POC supports the design rationale of the final framework, but the dissertation's main claims should come from the final finance experiments rather than from this small GridWorld study.
