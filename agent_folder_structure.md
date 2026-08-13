# Folder Structure

Root: `MSc_Project`

```text
MSc_Project
├── agent_framework
│   ├── agents
│   │   ├── __pycache__
│   │   │   ├── base_agent.cpython-312.pyc
│   │   │   └── ppo_agent.cpython-312.pyc
│   │   ├── base_agent.py
│   │   ├── llm_rl_agent.py
│   │   └── ppo_agent.py
│   ├── config
│   │   ├── __pycache__
│   │   │   └── config.cpython-312.pyc
│   │   ├── .env
│   │   └── config.py
│   ├── environment
│   │   ├── __pycache__
│   │   │   ├── action_space.cpython-312.pyc
│   │   │   ├── api_client.cpython-312.pyc
│   │   │   ├── finance_env.cpython-312.pyc
│   │   │   ├── reward_processor.cpython-312.pyc
│   │   │   └── state_encoder.cpython-312.pyc
│   │   ├── action_space.py
│   │   ├── api_client.py
│   │   ├── finance_env.py
│   │   ├── procedure_tracker.py
│   │   ├── reward_processor.py
│   │   └── state_encoder.py
│   ├── llm
│   │   ├── cache.py
│   │   ├── parser.py
│   │   ├── planner.py
│   │   └── prompts.py
│   ├── memory
│   │   ├── __pycache__
│   │   │   └── rollout_buffer.cpython-312.pyc
│   │   └── rollout_buffer.py
│   ├── models
│   │   ├── __pycache__
│   │   │   ├── policy_network.cpython-312.pyc
│   │   │   └── value_network.cpython-312.pyc
│   │   ├── checkpoints
│   │   ├── policy_network.py
│   │   └── value_network.py
│   ├── results
│   │   ├── graphs
│   │   │   └── ppo_baseline_20260812_002054
│   │   │       ├── ppo_action_frequency.png
│   │   │       ├── ppo_entropy.png
│   │   │       ├── ppo_evaluation_reward.png
│   │   │       ├── ppo_policy_loss.png
│   │   │       ├── ppo_reward_curve.png
│   │   │       ├── ppo_steps_curve.png
│   │   │       ├── ppo_success_rate.png
│   │   │       ├── ppo_termination_reasons.png
│   │   │       └── ppo_value_loss.png
│   │   ├── logs
│   │   │   ├── ppo_baseline_20260812_002054
│   │   │   │   └── training.log
│   │   │   └── test
│   │   ├── metrics
│   │   │   └── ppo_baseline_20260812_002054
│   │   │       └── ppo_results.xlsx
│   │   └── models
│   │       └── ppo_baseline_20260812_002054
│   │           ├── ppo_episode_100.pt
│   │           ├── ppo_episode_1000.pt
│   │           ├── ppo_episode_200.pt
│   │           ├── ppo_episode_300.pt
│   │           ├── ppo_episode_400.pt
│   │           ├── ppo_episode_500.pt
│   │           ├── ppo_episode_600.pt
│   │           ├── ppo_episode_700.pt
│   │           ├── ppo_episode_800.pt
│   │           ├── ppo_episode_900.pt
│   │           └── ppo_final.pt
│   ├── tests
│   │   ├── test_agent.py
│   │   ├── test_api.py
│   │   ├── test_environment.py
│   │   └── test_llm.py
│   ├── training
│   │   ├── __pycache__
│   │   │   ├── evaluate.cpython-312.pyc
│   │   │   └── train.cpython-312.pyc
│   │   ├── callbacks.py
│   │   ├── evaluate.py
│   │   ├── experiment.py
│   │   └── train.py
│   ├── utils
│   │   ├── __pycache__
│   │   │   ├── logger.cpython-312.pyc
│   │   │   ├── metrics.cpython-312.pyc
│   │   │   └── visualization.cpython-312.pyc
│   │   ├── constants.py
│   │   ├── helpers.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── main.py
│   ├── README.md
│   └── requirements.txt
├── backend_server
│   ├── config
│   │   └── db.js
│   ├── controllers
│   │   ├── accountController.js
│   │   ├── approvalController.js
│   │   ├── authController.js
│   │   ├── episodeController.js
│   │   ├── invoiceController.js
│   │   ├── paymentController.js
│   │   ├── reportController.js
│   │   ├── sandboxController.js
│   │   └── supplierController.js
│   ├── middleware
│   │   ├── actionLogger.js
│   │   └── authMiddleware.js
│   ├── models
│   │   ├── Account.js
│   │   ├── AuditLog.js
│   │   ├── Budget.js
│   │   ├── Episode.js
│   │   ├── Invoice.js
│   │   ├── Supplier.js
│   │   ├── Transaction.js
│   │   └── User.js
│   ├── routes
│   │   ├── accountRoutes.js
│   │   ├── approvalRoutes.js
│   │   ├── authRoutes.js
│   │   ├── episodeRoutes.js
│   │   ├── invoiceRoutes.js
│   │   ├── paymentRoutes.js
│   │   ├── reportRoutes.js
│   │   ├── sandboxRoutes.js
│   │   └── supplierRoutes.js
│   ├── services
│   │   └── auditService.js
│   ├── utils
│   │   ├── permissions.js
│   │   └── rewards.js
│   ├── .env
│   ├── index.js
│   ├── package-lock.json
│   ├── package.json
│   └── test_backend_environment.py
├── POC
│   ├── test_module
│   │   ├── __init__.py
│   │   ├── test_env.py
│   │   └── test_ollama.py
│   ├── .env
│   ├── agent.py
│   ├── config.yaml
│   ├── environment.py
│   ├── main.py
│   ├── planner.py
│   ├── requirements.txt
│   ├── test.py
│   └── visualize.py
├── .gitignore
├── agent_folder_structure.md
├── generate_folder_structure.py
└── MSc_Final_Project.docx - Shortcut.lnk
```