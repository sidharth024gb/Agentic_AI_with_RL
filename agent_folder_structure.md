# Folder Structure

Root: `MSc_Project`

```text
MSc_Project
├── agent
│   ├── models
│   ├── results
│   ├── config.py
│   ├── evaluate.py
│   ├── finance_env.py
│   ├── llm_planner.py
│   ├── plot_results.py
│   └── train_ppo.py
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