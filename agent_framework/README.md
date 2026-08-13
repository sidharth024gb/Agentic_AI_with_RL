# Autonomous Finance RL Agent

## MSc Project

An autonomous reinforcement learning agent framework for task execution in a controlled finance sandbox.

The project investigates whether augmenting a Reinforcement Learning (RL) agent with Large Language Model (LLM) reasoning can improve training efficiency, convergence, task execution, or other measurable aspects of agent performance.

The primary comparison is between:

- **PPO** — Proximal Policy Optimization
- **LLM + PPO** — PPO augmented with LLM-based planning

The LLM used in this project is **Llama 3 running locally through Ollama**.

---

## 1. Project Objective

The main research objective is to investigate whether LLM-based reasoning can provide additional benefits to an RL agent operating in a structured environment.

The basic idea is:

```text
                Environment State
                       │
                       ▼
                 ┌───────────┐
                 │    PPO    │
                 │   Agent   │
                 └─────┬─────┘
                       │
                       ▼
                    Action
                       │
                       ▼
              Finance Sandbox
                       │
                       ▼
                New Environment
                     State
```

The LLM-augmented architecture adds an LLM planning component:

```text
                 Environment State
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
         PPO Agent          Llama 3 Planner
                                  │
                                  ▼
                              Plan / Guidance
                                  │
                                  ▼
                         LLM + PPO Agent
                                  │
                                  ▼
                               Action
                                  │
                                  ▼
                         Finance Sandbox
```

The experiment aims to determine whether the LLM augmentation can:

- improve learning speed
- reduce the number of steps required to complete tasks
- improve task success rate
- improve convergence
- improve decision making
- reduce inefficient exploration
- provide other measurable benefits

---

# 2. Architecture

The project is divided into several components.

```text
finance-agent/
│
├── config/
│   ├── config.py
│   ├── api_config.py
│   └── experiment_config.py
│
├── environment/
│   ├── finance_env.py
│   ├── api_client.py
│   ├── state_encoder.py
│   ├── reward_processor.py
│   └── action_space.py
│
├── agents/
│   ├── base_agent.py
│   ├── ppo_agent.py
│   └── llm_rl_agent.py
│
├── llm/
│   ├── planner.py
│   ├── prompts.py
│   └── parser.py
│
├── memory/
│   ├── replay_buffer.py
│   ├── episode_memory.py
│   └── rollout_buffer.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── experiment.py
│   └── callbacks.py
│
├── models/
│   ├── policy_network.py
│   ├── value_network.py
│   └── checkpoints/
│
├── utils/
│   ├── logger.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── constants.py
│   └── helpers.py
│
├── results/
│   ├── logs/
│   ├── models/
│   ├── metrics/
│   └── graphs/
│
├── tests/
│   ├── test_environment.py
│   ├── test_agent.py
│   └── test_api.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 3. Environment

The RL agent interacts with a finance backend through `FinanceEnv`.

The environment provides the standard RL interaction:

```text
State
  │
  ▼
Agent
  │
  ▼
Action
  │
  ▼
Finance Backend
  │
  ├── New State
  └── Reward
```

The environment is responsible for:

- resetting the sandbox
- obtaining the current state
- validating actions
- executing agent actions
- processing rewards
- recording episode steps
- determining whether the episode is complete
- determining whether the episode is done

### `complete` vs `done`

The project uses these concepts separately:

```text
complete = True
    → The agent has achieved the goal.

done = True
    → The episode should terminate.
```

Therefore:

```text
Goal achieved
    ├── complete = True
    └── done = True
```

while:

```text
Maximum episode steps reached
    ├── complete = False
    └── done = True
```

---

# 4. Backend API

The RL environment communicates with the finance backend through `api_client.py`.

The API client exposes only actions permitted to the agent.

### Authentication

- Login
- Get profile

### Sandbox

- Reset environment
- Get state
- Get reward

Randomization is part of the environment reset operation and therefore does not use a separate randomization endpoint.

### Episodes

- Get episode
- Start episode
- Record step
- End episode

### Invoices

- Get invoices
- Get invoice
- Invoice duplicate check

### Invoice approval

- Approve invoice

### Payments

- Pay invoice
- Cancel payment
- Retry payment

### Suppliers

- Get suppliers
- Validate supplier

### Accounts

- Get accounts
- Check budget
- Get cash position

### Reports

- Get transactions
- Generate report

The agent does **not** expose:

- Create invoice
- Blacklist supplier
- Separate randomize endpoint

This keeps the RL action space aligned with the backend permissions.

---

# 5. Action Space

The action space represents the operations available to the agent.

Actions are state-aware so that the agent does not blindly attempt operations that are impossible or inappropriate for the current environment state.

For example, actions can depend on whether:

- an invoice exists
- a supplier is available
- an invoice has been approved
- sufficient budget exists
- sufficient account balance exists
- a payment transaction exists

This allows the agent to explore the finance environment while respecting the available backend operations.

---

# 6. State Representation

The raw backend state is transformed into a numerical representation using:

```text
environment/state_encoder.py
```

The state encoder converts structured finance information into a representation suitable for neural networks.

The general pipeline is:

```text
Backend State
     │
     ▼
State Encoder
     │
     ▼
Numerical State Vector
     │
     ▼
Policy / Value Network
```

---

# 7. Reward System

The environment uses rewards to indicate whether an action moves the agent toward its goal.

Important reward categories include:

| Event                   | Reward |
| ----------------------- | -----: |
| Successful task         |    +10 |
| Invoice found           |     +5 |
| Supplier validated      |    +10 |
| Budget check successful |    +10 |
| Payment successful      |    +30 |
| Report generated        |     +5 |
| Invalid action          |     -5 |
| Invoice not found       |    -10 |
| Duplicate invoice       |    -15 |
| Supplier inactive       |    -20 |
| Supplier high risk      |    -15 |
| Budget exceeded         |    -25 |
| Insufficient balance    |    -30 |
| Payment failed          |    -20 |
| Unauthorized action     |    -50 |
| No reward               |      0 |

The reward processor converts backend responses into RL-compatible rewards.

---

# 8. PPO Agent

The main RL algorithm used in this project is **Proximal Policy Optimization (PPO)**.

The PPO implementation is built directly using PyTorch rather than relying on Stable-Baselines3.

The main components are:

```text
PPO Agent
    │
    ├── Policy Network
    │
    ├── Value Network
    │
    ├── Rollout Buffer
    │
    └── PPO Update
```

### Policy Network

The policy network estimates the probability distribution over available actions.

```text
State
  │
  ▼
Policy Network
  │
  ▼
Action Distribution
  │
  ▼
Selected Action
```

### Value Network

The value network estimates the expected future return from a state.

```text
State
  │
  ▼
Value Network
  │
  ▼
State Value
```

### Rollout Buffer

The rollout buffer stores experience collected during interaction with the environment.

Typical information includes:

- states
- actions
- rewards
- log probabilities
- value estimates
- termination information
- advantages
- returns

---

# 9. LLM + PPO Agent

The LLM-augmented agent extends the PPO approach with an LLM planning component.

The LLM is:

**Llama 3 through Ollama**

The general architecture is:

```text
Environment State
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
   State Encoder      Llama 3
       │                  │
       │                  ▼
       │               Planner
       │                  │
       │                  ▼
       │               Parsed
       │               Guidance
       │                  │
       └────────┬─────────┘
                ▼
           LLM + PPO
                │
                ▼
              Action
```

The LLM is not intended to replace PPO.

Instead, it provides higher-level reasoning or planning information that can assist the RL agent.

This allows the experiment to compare:

```text
PPO
vs
PPO + LLM reasoning
```

under the same finance environment.

---

# 10. LLM Components

The LLM subsystem contains three main components.

### `planner.py`

Responsible for communicating with the local Ollama Llama 3 model and obtaining planning guidance.

### `prompts.py`

Contains the prompts used to provide environment information and task context to the LLM.

### `parser.py`

Converts the LLM response into structured information that can be used by the agent.

The intended pipeline is:

```text
Environment State
       │
       ▼
Prompt
       │
       ▼
Llama 3 / Ollama
       │
       ▼
LLM Response
       │
       ▼
Parser
       │
       ▼
Structured Guidance
       │
       ▼
RL Agent
```

---

# 11. Training

Training functionality is contained in:

```text
training/
├── train.py
├── evaluate.py
├── experiment.py
└── callbacks.py
```

### `train.py`

Responsible for training an agent.

### `evaluate.py`

Evaluates a trained agent without updating its parameters.

### `experiment.py`

Coordinates the main research comparison.

The primary comparison is:

```text
                ┌──────────────┐
                │   Finance    │
                │ Environment  │
                └──────┬───────┘
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
          PPO Agent          LLM + PPO
             │                   │
             ▼                   ▼
         Training            Training
             │                   │
             └─────────┬─────────┘
                       ▼
                  Evaluation
                       │
                       ▼
                  Comparison
```

---

# 12. Research Metrics

The project can evaluate the agents using metrics such as:

- episode reward
- average reward
- episode length
- average number of steps
- task completion rate
- success rate
- convergence behaviour
- training time
- LLM planning time
- number of LLM calls
- reward progression

The objective is not simply to determine which agent obtains the highest reward.

The experiment should also investigate whether LLM augmentation provides useful benefits relative to its additional computational cost.

---

# 13. Installation

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 14. Environment Configuration

Create a `.env` file containing the required backend and authentication configuration.

Example structure:

```text
BACKEND_URL=your_backend_url
USERNAME=your_username
PASSWORD=your_password
```

The exact configuration values should correspond to the variables defined in:

```text
config/config.py
config/api_config.py
```

Do not commit `.env` to version control.

---

# 15. Ollama / Llama 3

The LLM agent uses Ollama to run Llama 3 locally.

Ensure Ollama is installed and the required model is available before running the LLM-augmented agent.

The LLM component is only required when running:

```text
LLM + PPO
```

The baseline PPO agent does not depend on Llama 3.

---

# 16. Running the Project

### Train PPO

```bash
python main.py train --agent ppo
```

### Train LLM + PPO

```bash
python main.py train --agent llm_ppo
```

### Evaluate PPO

```bash
python main.py evaluate --agent ppo
```

### Evaluate LLM + PPO

```bash
python main.py evaluate --agent llm_ppo
```

### Run the main comparison

```bash
python main.py experiment
```

---

# 17. Testing

The project contains separate tests for the main components.

### API tests

```bash
pytest tests/test_api.py
```

These test the API client without requiring real HTTP requests.

### Agent tests

```bash
pytest tests/test_agent.py
```

These test the PPO and LLM + PPO agent interfaces and neural-network components.

### Environment tests

```bash
pytest tests/test_environment.py
```

These test the finance RL environment and its interaction with the backend.

### Run all tests

```bash
pytest
```

---

# 18. Results

Experiment outputs are stored under:

```text
results/
├── logs/
├── models/
├── metrics/
└── graphs/
```

The results should make it possible to compare the baseline and LLM-augmented agents.

A typical research comparison can be represented as:

| Metric              | PPO | LLM + PPO |
| ------------------- | --: | --------: |
| Average reward      |     |           |
| Average steps       |     |           |
| Success rate        |     |           |
| Completion rate     |     |           |
| Training time       |     |           |
| Convergence time    |     |           |
| LLM overhead        | N/A |           |
| Number of LLM calls | N/A |           |

The final values should be generated from the actual experiments rather than manually entered.

---

# 19. Research Experiment

The central experiment compares two conditions:

### Baseline

```text
Finance Environment
        ↓
       PPO
        ↓
     Actions
```

### LLM Augmented

```text
Finance Environment
        ↓
   Llama 3 Planner
        ↓
   PPO + Guidance
        ↓
      Actions
```

The experiments should use comparable training conditions so that differences can be attributed as much as possible to the LLM augmentation.

Important experimental variables include:

- random seed
- number of training episodes
- maximum episode steps
- PPO hyperparameters
- environment configuration
- task goals
- LLM model
- LLM planning frequency

Multiple runs with different seeds can be used to reduce the effect of randomness.

---

# 20. Project Status

Current architecture:

```text
Configuration
      │
      ▼
Finance Environment
      │
      ├── API Client
      ├── State Encoder
      ├── Reward Processor
      └── Action Space
      │
      ▼
Agents
      │
      ├── PPO
      │
      └── LLM + PPO
             │
             └── Llama 3 / Ollama
      │
      ▼
Training
      │
      ├── Training
      ├── Evaluation
      ├── Experiments
      └── Callbacks
      │
      ▼
Metrics / Visualization
      │
      ▼
Research Results
```

The implementation intentionally focuses on **PPO and LLM-augmented PPO**, as the primary research question is whether LLM reasoning can improve RL training or task execution.

Baseline PPO:

    python main.py train --agent ppo

LLM + PPO:

    python main.py train --agent llm_rl

Short test:

    python main.py train --agent llm_rl --episodes 50 --eval-episodes 10


1. invoiceController.js
2. supplierController.js
3. accountController.js
4. paymentController.js
5. api_client.py

         ↓ backend semantics now correct

6. reward_processor.py
7. finance_env.py
8. state_encoder.py

         ↓ environment now correct

9. config.py
10. ppo_agent.py
11. train.py

         ↓ PPO exploration/training now correct

12. Test PPO baseline

         ↓ only after PPO converges

13. Test LLM INPUT
14. Test LLM REWARD_SHAPING
15. Test LLM INPUT_AND_REWARD

# First verify the corrected environment
python main.py train --agent ppo --episodes 50 --eval-episodes 10

# Final baseline
python main.py train --agent ppo --episodes 1000 --eval-episodes 100

# LLM input
python main.py train --agent llm_rl --guidance-mode input --episodes 1000 --eval-episodes 100

# LLM reward shaping
python main.py train --agent llm_rl --guidance-mode reward_shaping --guidance-bonus 1.0 --episodes 1000 --eval-episodes 100

# LLM input + reward
python main.py train --agent llm_rl --guidance-mode input_and_reward --guidance-bonus 1.0 --episodes 1000 --eval-episodes 100