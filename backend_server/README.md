# Finance Sandbox Backend

This folder implements the stateful finance environment used by the final PPO experiments. It is a Node.js/Express API backed by MongoDB.

The backend represents the simulated enterprise finance system. It enforces workflow rules, updates business state, records episodes/audit events and exposes finance actions to the Python RL environment.

It does **not** implement PPO or the LLM planner.

## Folder Structure

```text
backend_server/
├── config/
├── controllers/
├── logs/
├── middleware/
├── models/
├── routes/
├── services/
├── utils/
├── .env
├── index.js
├── package.json
├── package-lock.json
├── README.md
└── test_backend_environment.py
```

## File Reference

### `config/`

| File | Responsibility |
| --- | --- |
| `db.js` | Connects to MongoDB and handles database connection status/errors. |

### `controllers/`

| File | Responsibility |
| --- | --- |
| `accountController.js` | Accounts, balances, budget checks, transfer/cash-position operations. |
| `approvalController.js` | Invoice approval/rejection workflow. |
| `authController.js` | Registration, login and current-user operations. |
| `episodeController.js` | Starts episodes, records steps and closes episodes. |
| `invoiceController.js` | Invoice listing/detail, duplicate detection and invoice operations. |
| `paymentController.js` | Payment, refund, cancellation and retry workflow. |
| `reportController.js` | Finance report generation. |
| `sandboxController.js` | Seeded environment reset, generated scenario state and environment state/reward access. |
| `supplierController.js` | Supplier validation and risk/activity checks. |

### `models/`

| File | Entity |
| --- | --- |
| `Account.js` | Treasury/payment account and balance state. |
| `AuditLog.js` | Audit history of user/agent actions. |
| `Budget.js` | Category/period budget state. |
| `Episode.js` | Agent experiment episode, actions, rewards, state and termination. |
| `Invoice.js` | Invoice workflow state, supplier, amount, duplicate flag and approval/payment fields. |
| `Supplier.js` | Supplier active/risk data. |
| `Transaction.js` | Payment/deposit/refund transaction data. |
| `User.js` | User/agent authentication and permissions. |

### `routes/`

| File | API group |
| --- | --- |
| `accountRoutes.js` | Account/budget operations. |
| `approvalRoutes.js` | Approval/rejection operations. |
| `authRoutes.js` | Authentication. |
| `episodeRoutes.js` | Episode lifecycle. |
| `invoiceRoutes.js` | Invoice operations. |
| `paymentRoutes.js` | Payment operations. |
| `reportRoutes.js` | Reporting. |
| `sandboxRoutes.js` | Environment reset/state/reward. |
| `supplierRoutes.js` | Supplier validation. |

Consult the route source for the exact current HTTP method/path/payload. This avoids stale endpoint documentation.

### `middleware/`

| File | Responsibility |
| --- | --- |
| `authMiddleware.js` | JWT authentication and permission-based access. |
| `actionLogger.js` | Records auditable API actions. |

### `services/`

| File | Responsibility |
| --- | --- |
| `auditService.js` | Central service for creating standardised audit records. |

### `utils/`

| File | Responsibility |
| --- | --- |
| `permissions.js` | Permission constants/rules. |
| `rewards.js` | Canonical backend business-reward constants. |

### Other files

| Entry | Responsibility |
| --- | --- |
| `index.js` | Creates Express app, middleware, routes and server startup. |
| `.env` | Local port, MongoDB, JWT and seed configuration. |
| `package.json` | Node dependencies and scripts. |
| `package-lock.json` | Exact installed dependency versions. |
| `test_backend_environment.py` | End-to-end Python API validation harness. |
| `logs/` | Excel logs from backend/environment tests. |

## Finance Workflow

A representative invoice-payment workflow is:

```text
GET/LIST INVOICES
      ↓
CHECK DUPLICATE
      ↓
VALIDATE SUPPLIER
      ↓
APPROVE if required
      ↓
CHECK BUDGET / BALANCE
      ↓
PAY
      ↓
TRANSACTION + AUDIT + EPISODE RECORD
```

The backend enforces business constraints rather than trusting the agent.

Examples include:

- invoice must exist;
- duplicate invoices are not payable;
- inactive/high-risk suppliers are filtered or rejected according to the workflow;
- pending invoices require approval before payment;
- budget and available balance must be sufficient;
- invalid transitions return a business failure rather than silently changing state.

Validation actions distinguish between:

```text
the API action executed successfully
```

and:

```text
the business object is eligible/valid
```

For example, successfully detecting a duplicate is a successful duplicate-check action even though the invoice itself is invalid for payment.

## Episode Logging

An episode represents one complete agent task, not one API request.

The Python environment:

1. resets the sandbox;
2. starts an episode;
3. executes actions;
4. records each step;
5. ends the episode with final state and termination reason.

Episode records contain information such as:

- agent type;
- algorithm;
- task goal;
- initial/final state;
- action sequence;
- reward;
- success/usefulness;
- total steps;
- completion status;
- termination reason;
- execution time.

System/environment failures are kept separate from ordinary agent mistakes.

## Reward Responsibility

`utils/rewards.js` contains backend business reward constants.

The final learning reward can additionally include logic from the Python `RewardProcessor`, such as:

- completion bonus;
- repeated-action efficiency shaping;
- LLM procedural guidance bonus.

Therefore the backend reward table should not be interpreted as the entire final PPO reward function.

Infrastructure failures return an environment-error signal and should not teach PPO that an otherwise correct finance action was bad.

## Dependencies

Install the exact Node dependencies using:

```bash
cd backend_server
npm install
```

`package-lock.json` is the authoritative tested dependency lock.

The backend uses dependency groups including:

- Express;
- Mongoose / MongoDB access;
- JSON Web Tokens;
- password hashing;
- dotenv;
- CORS;
- development tooling defined in `package.json`.

The Python `test_backend_environment.py` harness uses packages such as `requests`, `pandas` and `openpyxl`; it can be run using the final agent Python environment.

## Configuration

Create:

```text
backend_server/.env
```

Typical settings include:

```dotenv
PORT=5000
MONGO_URI=mongodb://127.0.0.1:27017/finance_rl_agent_env
JWT_SECRET=<long-random-secret>
RANDOM_SEED=42
```

Use the exact names consumed by the current source.

The agent backend URL must point to the same server, e.g.:

```dotenv
BASE_URL=http://localhost:5000/api
```

Never commit real JWT secrets or credentials.

## Seeded Sandbox Reset

Reproducibility is important because the finance scenario is randomised.

The Python environment sends the episode seed during reset. `sandboxController.js` uses deterministic seeded random generation for experiment-critical scenario construction.

This allows:

```text
same episode seed
      ↓
same backend scenario
```

when configuration and code are unchanged.

For fair experiments:

- use the same base/episode seeds across all conditions;
- reset before every episode;
- do not manually edit MongoDB during training;
- do not mix development-test records with experimental runs;
- keep the backend code/version fixed across conditions.

## Running the Backend

### 1. Start MongoDB

Start the local MongoDB service or ensure Atlas is reachable.

### 2. Install dependencies

```bash
cd backend_server
npm install
```

### 3. Start the server

```bash
npm start
```

If the current `package.json` defines a development command:

```bash
npm run dev
```

Keep the backend terminal open while the Python agent is running.

## Backend Verification

From the repository root, with the final agent Python environment active:

```bash
python backend_server/test_backend_environment.py
```

The harness validates major environment/API flows and creates timestamped Excel logs under:

```text
backend_server/logs/
```

Run this before long multi-seed training.

## API Groups

| Group | Typical purpose |
| --- | --- |
| Authentication | Register/login/profile. |
| Sandbox | Reset, observable state and reward/state support. |
| Episodes | Start, append step, retrieve and end an episode. |
| Invoices | List/detail and duplicate-related operations. |
| Suppliers | Validate supplier activity/risk. |
| Approvals | Approve or reject invoices. |
| Payments | Pay/refund/cancel/retry. |
| Accounts | Read accounts, check budget, transfer/cash position. |
| Reports | Generate finance reports. |

Exact routes are defined in the corresponding `routes/*.js` file.

## Permission Model

The backend uses permission-based authorization.

The agent account receives only the permissions required to interact with the finance environment. Administrative operations remain protected.

This separation means the RL agent interacts with a realistic constrained system rather than receiving unrestricted database access.

## Troubleshooting

### MongoDB duplicate-key error on `referenceId`

If MongoDB reports a duplicate key for `referenceId`, verify that generated transaction/invoice reference IDs are non-null and unique.

### Approval endpoint failures

The Python client and Express route must use the same HTTP method and path. The final RL integration uses the corrected approval route/client contract.

### Environment errors during training

Do not convert database/server failures into normal negative PPO rewards. Record the episode as an environment error, fix the backend issue and rerun the affected experiment if required.
