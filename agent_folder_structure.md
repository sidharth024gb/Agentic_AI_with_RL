# Folder Structure

Root: `MSc_Project`

```text
MSc_Project
├── agent_framework
│   ├── agents
│   │   ├── base_agent.py
│   │   ├── llm_rl_agent.py
│   │   └── ppo_agent.py
│   ├── config
│   │   ├── .env
│   │   └── config.py
│   ├── environment
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
│   │   └── rollout_buffer.py
│   ├── models
│   │   ├── policy_network.py
│   │   └── value_network.py
│   ├── results
│   │   ├── graphs
│   │   │   ├── llm_ppo_input_20260814_235722
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_reward_20260814_230254
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_reward_20260815_003001
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   └── ppo_baseline_20260815_010839
│   │   │       ├── ppo_action_frequency.png
│   │   │       ├── ppo_entropy.png
│   │   │       ├── ppo_evaluation_reward.png
│   │   │       ├── ppo_policy_loss.png
│   │   │       ├── ppo_reward_curve.png
│   │   │       ├── ppo_steps_curve.png
│   │   │       ├── ppo_success_rate.png
│   │   │       ├── ppo_termination.png
│   │   │       └── ppo_value_loss.png
│   │   ├── llm_cache
│   │   │   └── db64633ce85d6f529f3672e6720da46a6a8773f3f689315e21a81e7327be5083.json
│   │   ├── logs
│   │   │   ├── llm_ppo_input_20260814_235722
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_reward_20260814_230254
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_reward_20260815_003001
│   │   │   │   └── training.log
│   │   │   └── ppo_baseline_20260815_010839
│   │   │       └── training.log
│   │   ├── metrics
│   │   │   ├── llm_ppo_input_20260814_235722
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_reward_20260814_230254
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_reward_20260815_003001
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   └── ppo_baseline_20260815_010839
│   │   │       └── ppo_results.xlsx
│   │   └── models
│   │       ├── llm_ppo_input_20260814_235722
│   │       │   ├── llm_rl_episode_100.pt
│   │       │   ├── llm_rl_episode_1000.pt
│   │       │   ├── llm_rl_episode_200.pt
│   │       │   ├── llm_rl_episode_300.pt
│   │       │   ├── llm_rl_episode_400.pt
│   │       │   ├── llm_rl_episode_500.pt
│   │       │   ├── llm_rl_episode_600.pt
│   │       │   ├── llm_rl_episode_700.pt
│   │       │   ├── llm_rl_episode_800.pt
│   │       │   ├── llm_rl_episode_900.pt
│   │       │   └── llm_rl_final.pt
│   │       ├── llm_ppo_input_reward_20260814_230254
│   │       │   ├── llm_rl_episode_100.pt
│   │       │   ├── llm_rl_episode_1000.pt
│   │       │   ├── llm_rl_episode_200.pt
│   │       │   ├── llm_rl_episode_300.pt
│   │       │   ├── llm_rl_episode_400.pt
│   │       │   ├── llm_rl_episode_500.pt
│   │       │   ├── llm_rl_episode_600.pt
│   │       │   ├── llm_rl_episode_700.pt
│   │       │   ├── llm_rl_episode_800.pt
│   │       │   ├── llm_rl_episode_900.pt
│   │       │   └── llm_rl_final.pt
│   │       ├── llm_ppo_reward_20260815_003001
│   │       │   ├── llm_rl_episode_100.pt
│   │       │   ├── llm_rl_episode_1000.pt
│   │       │   ├── llm_rl_episode_200.pt
│   │       │   ├── llm_rl_episode_300.pt
│   │       │   ├── llm_rl_episode_400.pt
│   │       │   ├── llm_rl_episode_500.pt
│   │       │   ├── llm_rl_episode_600.pt
│   │       │   ├── llm_rl_episode_700.pt
│   │       │   ├── llm_rl_episode_800.pt
│   │       │   ├── llm_rl_episode_900.pt
│   │       │   └── llm_rl_final.pt
│   │       └── ppo_baseline_20260815_010839
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
│   │   ├── test_environment.py
│   │   └── test_llm.py
│   ├── training
│   │   ├── evaluate.py
│   │   └── train.py
│   ├── utils
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
│   ├── logs
│   │   ├── rl_environment_test_20260802_125555.xlsx
│   │   ├── rl_environment_test_20260802_134953.xlsx
│   │   ├── rl_environment_test_20260802_140522.xlsx
│   │   ├── rl_environment_test_20260802_141157.xlsx
│   │   └── rl_environment_test_20260802_141357.xlsx
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
│   ├── README.md
│   └── test_backend_environment.py
├── POC
│   ├── logs
│   │   ├── Baseline
│   │   │   ├── monitor.csv
│   │   │   └── ppo_gridworld_Baseline.zip
│   │   ├── LLM
│   │   │   ├── monitor.csv
│   │   │   └── ppo_gridworld_LLM.zip
│   │   ├── Model_Performance.xlsx
│   │   └── Training Convergence.png
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
├── MSc_Final_Project.docx - Shortcut.lnk
└── README.md
```