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
│   │   │   ├── combined
│   │   │   │   └── final_dissertation_20260815_170644
│   │   │   │       ├── all_seeds
│   │   │   │       │   ├── combined_run_matrix.csv
│   │   │   │       │   ├── convergence_episode_bar.png
│   │   │   │       │   ├── evaluation_action_frequency_comparison.png
│   │   │   │       │   ├── evaluation_average_reward_bar.png
│   │   │   │       │   ├── evaluation_reward_comparison.png
│   │   │   │       │   ├── evaluation_steps_bar.png
│   │   │   │       │   ├── evaluation_success_bar.png
│   │   │   │       │   ├── llm_average_llm_latency_ms.png
│   │   │   │       │   ├── llm_cache_hits.png
│   │   │   │       │   ├── llm_total_llm_latency_ms.png
│   │   │   │       │   ├── ppo_approx_kl.png
│   │   │   │       │   ├── ppo_clip_fraction.png
│   │   │   │       │   ├── ppo_entropy.png
│   │   │   │       │   ├── ppo_explained_variance.png
│   │   │   │       │   ├── ppo_normalized_entropy.png
│   │   │   │       │   ├── ppo_policy_loss.png
│   │   │   │       │   ├── ppo_value_loss.png
│   │   │   │       │   ├── procedure_adherence_comparison.png
│   │   │   │       │   ├── train_action_frequency_comparison.png
│   │   │   │       │   ├── training_average_reward_bar.png
│   │   │   │       │   ├── training_last_100_success_bar.png
│   │   │   │       │   ├── training_reward_comparison.png
│   │   │   │       │   ├── training_steps_comparison.png
│   │   │   │       │   ├── training_success_bar.png
│   │   │   │       │   ├── training_success_comparison.png
│   │   │   │       │   └── training_wall_clock_bar.png
│   │   │   │       ├── seed_10
│   │   │   │       │   ├── combined_run_matrix.csv
│   │   │   │       │   ├── convergence_episode_bar.png
│   │   │   │       │   ├── evaluation_action_frequency_comparison.png
│   │   │   │       │   ├── evaluation_average_reward_bar.png
│   │   │   │       │   ├── evaluation_reward_comparison.png
│   │   │   │       │   ├── evaluation_steps_bar.png
│   │   │   │       │   ├── evaluation_success_bar.png
│   │   │   │       │   ├── llm_average_llm_latency_ms.png
│   │   │   │       │   ├── llm_cache_hits.png
│   │   │   │       │   ├── llm_total_llm_latency_ms.png
│   │   │   │       │   ├── ppo_approx_kl.png
│   │   │   │       │   ├── ppo_clip_fraction.png
│   │   │   │       │   ├── ppo_entropy.png
│   │   │   │       │   ├── ppo_explained_variance.png
│   │   │   │       │   ├── ppo_normalized_entropy.png
│   │   │   │       │   ├── ppo_policy_loss.png
│   │   │   │       │   ├── ppo_value_loss.png
│   │   │   │       │   ├── procedure_adherence_comparison.png
│   │   │   │       │   ├── train_action_frequency_comparison.png
│   │   │   │       │   ├── training_average_reward_bar.png
│   │   │   │       │   ├── training_last_100_success_bar.png
│   │   │   │       │   ├── training_reward_comparison.png
│   │   │   │       │   ├── training_steps_comparison.png
│   │   │   │       │   ├── training_success_bar.png
│   │   │   │       │   ├── training_success_comparison.png
│   │   │   │       │   └── training_wall_clock_bar.png
│   │   │   │       ├── seed_24
│   │   │   │       │   ├── combined_run_matrix.csv
│   │   │   │       │   ├── convergence_episode_bar.png
│   │   │   │       │   ├── evaluation_action_frequency_comparison.png
│   │   │   │       │   ├── evaluation_average_reward_bar.png
│   │   │   │       │   ├── evaluation_reward_comparison.png
│   │   │   │       │   ├── evaluation_steps_bar.png
│   │   │   │       │   ├── evaluation_success_bar.png
│   │   │   │       │   ├── llm_average_llm_latency_ms.png
│   │   │   │       │   ├── llm_cache_hits.png
│   │   │   │       │   ├── llm_total_llm_latency_ms.png
│   │   │   │       │   ├── ppo_approx_kl.png
│   │   │   │       │   ├── ppo_clip_fraction.png
│   │   │   │       │   ├── ppo_entropy.png
│   │   │   │       │   ├── ppo_explained_variance.png
│   │   │   │       │   ├── ppo_normalized_entropy.png
│   │   │   │       │   ├── ppo_policy_loss.png
│   │   │   │       │   ├── ppo_value_loss.png
│   │   │   │       │   ├── procedure_adherence_comparison.png
│   │   │   │       │   ├── train_action_frequency_comparison.png
│   │   │   │       │   ├── training_average_reward_bar.png
│   │   │   │       │   ├── training_last_100_success_bar.png
│   │   │   │       │   ├── training_reward_comparison.png
│   │   │   │       │   ├── training_steps_comparison.png
│   │   │   │       │   ├── training_success_bar.png
│   │   │   │       │   ├── training_success_comparison.png
│   │   │   │       │   └── training_wall_clock_bar.png
│   │   │   │       ├── seed_33
│   │   │   │       │   ├── combined_run_matrix.csv
│   │   │   │       │   ├── convergence_episode_bar.png
│   │   │   │       │   ├── evaluation_action_frequency_comparison.png
│   │   │   │       │   ├── evaluation_average_reward_bar.png
│   │   │   │       │   ├── evaluation_reward_comparison.png
│   │   │   │       │   ├── evaluation_steps_bar.png
│   │   │   │       │   ├── evaluation_success_bar.png
│   │   │   │       │   ├── llm_average_llm_latency_ms.png
│   │   │   │       │   ├── llm_cache_hits.png
│   │   │   │       │   ├── llm_total_llm_latency_ms.png
│   │   │   │       │   ├── ppo_approx_kl.png
│   │   │   │       │   ├── ppo_clip_fraction.png
│   │   │   │       │   ├── ppo_entropy.png
│   │   │   │       │   ├── ppo_explained_variance.png
│   │   │   │       │   ├── ppo_normalized_entropy.png
│   │   │   │       │   ├── ppo_policy_loss.png
│   │   │   │       │   ├── ppo_value_loss.png
│   │   │   │       │   ├── procedure_adherence_comparison.png
│   │   │   │       │   ├── train_action_frequency_comparison.png
│   │   │   │       │   ├── training_average_reward_bar.png
│   │   │   │       │   ├── training_last_100_success_bar.png
│   │   │   │       │   ├── training_reward_comparison.png
│   │   │   │       │   ├── training_steps_comparison.png
│   │   │   │       │   ├── training_success_bar.png
│   │   │   │       │   ├── training_success_comparison.png
│   │   │   │       │   └── training_wall_clock_bar.png
│   │   │   │       ├── seed_42
│   │   │   │       │   ├── combined_run_matrix.csv
│   │   │   │       │   ├── convergence_episode_bar.png
│   │   │   │       │   ├── evaluation_action_frequency_comparison.png
│   │   │   │       │   ├── evaluation_average_reward_bar.png
│   │   │   │       │   ├── evaluation_reward_comparison.png
│   │   │   │       │   ├── evaluation_steps_bar.png
│   │   │   │       │   ├── evaluation_success_bar.png
│   │   │   │       │   ├── llm_average_llm_latency_ms.png
│   │   │   │       │   ├── llm_cache_hits.png
│   │   │   │       │   ├── llm_total_llm_latency_ms.png
│   │   │   │       │   ├── ppo_approx_kl.png
│   │   │   │       │   ├── ppo_clip_fraction.png
│   │   │   │       │   ├── ppo_entropy.png
│   │   │   │       │   ├── ppo_explained_variance.png
│   │   │   │       │   ├── ppo_normalized_entropy.png
│   │   │   │       │   ├── ppo_policy_loss.png
│   │   │   │       │   ├── ppo_value_loss.png
│   │   │   │       │   ├── procedure_adherence_comparison.png
│   │   │   │       │   ├── train_action_frequency_comparison.png
│   │   │   │       │   ├── training_average_reward_bar.png
│   │   │   │       │   ├── training_last_100_success_bar.png
│   │   │   │       │   ├── training_reward_comparison.png
│   │   │   │       │   ├── training_steps_comparison.png
│   │   │   │       │   ├── training_success_bar.png
│   │   │   │       │   ├── training_success_comparison.png
│   │   │   │       │   └── training_wall_clock_bar.png
│   │   │   │       └── seed_50
│   │   │   │           ├── combined_run_matrix.csv
│   │   │   │           ├── convergence_episode_bar.png
│   │   │   │           ├── evaluation_action_frequency_comparison.png
│   │   │   │           ├── evaluation_average_reward_bar.png
│   │   │   │           ├── evaluation_reward_comparison.png
│   │   │   │           ├── evaluation_steps_bar.png
│   │   │   │           ├── evaluation_success_bar.png
│   │   │   │           ├── llm_average_llm_latency_ms.png
│   │   │   │           ├── llm_cache_hits.png
│   │   │   │           ├── llm_total_llm_latency_ms.png
│   │   │   │           ├── ppo_approx_kl.png
│   │   │   │           ├── ppo_clip_fraction.png
│   │   │   │           ├── ppo_entropy.png
│   │   │   │           ├── ppo_explained_variance.png
│   │   │   │           ├── ppo_normalized_entropy.png
│   │   │   │           ├── ppo_policy_loss.png
│   │   │   │           ├── ppo_value_loss.png
│   │   │   │           ├── procedure_adherence_comparison.png
│   │   │   │           ├── train_action_frequency_comparison.png
│   │   │   │           ├── training_average_reward_bar.png
│   │   │   │           ├── training_last_100_success_bar.png
│   │   │   │           ├── training_reward_comparison.png
│   │   │   │           ├── training_steps_comparison.png
│   │   │   │           ├── training_success_bar.png
│   │   │   │           ├── training_success_comparison.png
│   │   │   │           └── training_wall_clock_bar.png
│   │   │   ├── llm_ppo_input_reward_seed_10_20260816_010414
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_reward_seed_24_20260815_214641
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_reward_seed_33_20260815_201033
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_reward_seed_42_20260815_182704
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_reward_seed_50_20260815_232531
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_seed_10_20260816_001356
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_seed_24_20260815_205745
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_seed_33_20260815_191919
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_seed_42_20260815_173247
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_input_seed_50_20260815_223543
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_reward_seed_10_20260816_003743
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_reward_seed_24_20260815_212059
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_reward_seed_33_20260815_194648
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_reward_seed_42_20260815_180030
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── llm_ppo_reward_seed_50_20260815_230019
│   │   │   │   ├── llm__plus__ppo_action_frequency.png
│   │   │   │   ├── llm__plus__ppo_approx_kl.png
│   │   │   │   ├── llm__plus__ppo_clip_fraction.png
│   │   │   │   ├── llm__plus__ppo_clipped_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_entropy.png
│   │   │   │   ├── llm__plus__ppo_evaluation_reward.png
│   │   │   │   ├── llm__plus__ppo_explained_variance.png
│   │   │   │   ├── llm__plus__ppo_guidance_bonus.png
│   │   │   │   ├── llm__plus__ppo_normalized_entropy.png
│   │   │   │   ├── llm__plus__ppo_policy_grad_norm.png
│   │   │   │   ├── llm__plus__ppo_policy_loss.png
│   │   │   │   ├── llm__plus__ppo_procedure_adherence.png
│   │   │   │   ├── llm__plus__ppo_reward_curve.png
│   │   │   │   ├── llm__plus__ppo_steps_curve.png
│   │   │   │   ├── llm__plus__ppo_success_rate.png
│   │   │   │   ├── llm__plus__ppo_termination.png
│   │   │   │   ├── llm__plus__ppo_value_grad_norm.png
│   │   │   │   └── llm__plus__ppo_value_loss.png
│   │   │   ├── ppo_baseline_seed_10_20260815_235021
│   │   │   │   ├── ppo_action_frequency.png
│   │   │   │   ├── ppo_approx_kl.png
│   │   │   │   ├── ppo_clip_fraction.png
│   │   │   │   ├── ppo_clipped_policy_loss.png
│   │   │   │   ├── ppo_entropy.png
│   │   │   │   ├── ppo_evaluation_reward.png
│   │   │   │   ├── ppo_explained_variance.png
│   │   │   │   ├── ppo_normalized_entropy.png
│   │   │   │   ├── ppo_policy_grad_norm.png
│   │   │   │   ├── ppo_policy_loss.png
│   │   │   │   ├── ppo_reward_curve.png
│   │   │   │   ├── ppo_steps_curve.png
│   │   │   │   ├── ppo_success_rate.png
│   │   │   │   ├── ppo_termination.png
│   │   │   │   ├── ppo_value_grad_norm.png
│   │   │   │   └── ppo_value_loss.png
│   │   │   ├── ppo_baseline_seed_24_20260815_203543
│   │   │   │   ├── ppo_action_frequency.png
│   │   │   │   ├── ppo_approx_kl.png
│   │   │   │   ├── ppo_clip_fraction.png
│   │   │   │   ├── ppo_clipped_policy_loss.png
│   │   │   │   ├── ppo_entropy.png
│   │   │   │   ├── ppo_evaluation_reward.png
│   │   │   │   ├── ppo_explained_variance.png
│   │   │   │   ├── ppo_normalized_entropy.png
│   │   │   │   ├── ppo_policy_grad_norm.png
│   │   │   │   ├── ppo_policy_loss.png
│   │   │   │   ├── ppo_reward_curve.png
│   │   │   │   ├── ppo_steps_curve.png
│   │   │   │   ├── ppo_success_rate.png
│   │   │   │   ├── ppo_termination.png
│   │   │   │   ├── ppo_value_grad_norm.png
│   │   │   │   └── ppo_value_loss.png
│   │   │   ├── ppo_baseline_seed_33_20260815_185300
│   │   │   │   ├── ppo_action_frequency.png
│   │   │   │   ├── ppo_approx_kl.png
│   │   │   │   ├── ppo_clip_fraction.png
│   │   │   │   ├── ppo_clipped_policy_loss.png
│   │   │   │   ├── ppo_entropy.png
│   │   │   │   ├── ppo_evaluation_reward.png
│   │   │   │   ├── ppo_explained_variance.png
│   │   │   │   ├── ppo_normalized_entropy.png
│   │   │   │   ├── ppo_policy_grad_norm.png
│   │   │   │   ├── ppo_policy_loss.png
│   │   │   │   ├── ppo_reward_curve.png
│   │   │   │   ├── ppo_steps_curve.png
│   │   │   │   ├── ppo_success_rate.png
│   │   │   │   ├── ppo_termination.png
│   │   │   │   ├── ppo_value_grad_norm.png
│   │   │   │   └── ppo_value_loss.png
│   │   │   ├── ppo_baseline_seed_42_20260815_170644
│   │   │   │   ├── ppo_action_frequency.png
│   │   │   │   ├── ppo_approx_kl.png
│   │   │   │   ├── ppo_clip_fraction.png
│   │   │   │   ├── ppo_clipped_policy_loss.png
│   │   │   │   ├── ppo_entropy.png
│   │   │   │   ├── ppo_evaluation_reward.png
│   │   │   │   ├── ppo_explained_variance.png
│   │   │   │   ├── ppo_normalized_entropy.png
│   │   │   │   ├── ppo_policy_grad_norm.png
│   │   │   │   ├── ppo_policy_loss.png
│   │   │   │   ├── ppo_reward_curve.png
│   │   │   │   ├── ppo_steps_curve.png
│   │   │   │   ├── ppo_success_rate.png
│   │   │   │   ├── ppo_termination.png
│   │   │   │   ├── ppo_value_grad_norm.png
│   │   │   │   └── ppo_value_loss.png
│   │   │   └── ppo_baseline_seed_50_20260815_221231
│   │   │       ├── ppo_action_frequency.png
│   │   │       ├── ppo_approx_kl.png
│   │   │       ├── ppo_clip_fraction.png
│   │   │       ├── ppo_clipped_policy_loss.png
│   │   │       ├── ppo_entropy.png
│   │   │       ├── ppo_evaluation_reward.png
│   │   │       ├── ppo_explained_variance.png
│   │   │       ├── ppo_normalized_entropy.png
│   │   │       ├── ppo_policy_grad_norm.png
│   │   │       ├── ppo_policy_loss.png
│   │   │       ├── ppo_reward_curve.png
│   │   │       ├── ppo_steps_curve.png
│   │   │       ├── ppo_success_rate.png
│   │   │       ├── ppo_termination.png
│   │   │       ├── ppo_value_grad_norm.png
│   │   │       └── ppo_value_loss.png
│   │   ├── llm_cache
│   │   │   └── db64633ce85d6f529f3672e6720da46a6a8773f3f689315e21a81e7327be5083.json
│   │   ├── logs
│   │   │   ├── combined
│   │   │   │   └── final_dissertation_20260815_170644
│   │   │   │       ├── all_seeds
│   │   │   │       │   ├── manifest.json
│   │   │   │       │   ├── suite_manifest.json
│   │   │   │       │   └── training.log
│   │   │   │       ├── seed_10
│   │   │   │       │   └── manifest.json
│   │   │   │       ├── seed_24
│   │   │   │       │   └── manifest.json
│   │   │   │       ├── seed_33
│   │   │   │       │   └── manifest.json
│   │   │   │       ├── seed_42
│   │   │   │       │   └── manifest.json
│   │   │   │       └── seed_50
│   │   │   │           └── manifest.json
│   │   │   ├── llm_ppo_input_reward_seed_10_20260816_010414
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_reward_seed_24_20260815_214641
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_reward_seed_33_20260815_201033
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_reward_seed_42_20260815_182704
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_reward_seed_50_20260815_232531
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_seed_10_20260816_001356
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_seed_24_20260815_205745
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_seed_33_20260815_191919
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_seed_42_20260815_173247
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_input_seed_50_20260815_223543
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_reward_seed_10_20260816_003743
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_reward_seed_24_20260815_212059
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_reward_seed_33_20260815_194648
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_reward_seed_42_20260815_180030
│   │   │   │   └── training.log
│   │   │   ├── llm_ppo_reward_seed_50_20260815_230019
│   │   │   │   └── training.log
│   │   │   ├── ppo_baseline_seed_10_20260815_235021
│   │   │   │   └── training.log
│   │   │   ├── ppo_baseline_seed_24_20260815_203543
│   │   │   │   └── training.log
│   │   │   ├── ppo_baseline_seed_33_20260815_185300
│   │   │   │   └── training.log
│   │   │   ├── ppo_baseline_seed_42_20260815_170644
│   │   │   │   └── training.log
│   │   │   ├── ppo_baseline_seed_50_20260815_221231
│   │   │   │   └── training.log
│   │   │   └── test
│   │   │       ├── rl_environment_test_20260815_015340.xlsx
│   │   │       └── rl_environment_test_20260815_020101.xlsx
│   │   ├── metrics
│   │   │   ├── combined
│   │   │   │   └── final_dissertation_20260815_170644
│   │   │   │       ├── all_seeds
│   │   │   │       │   └── all_seeds_comparison.xlsx
│   │   │   │       ├── seed_10
│   │   │   │       │   └── seed_10_comparison.xlsx
│   │   │   │       ├── seed_24
│   │   │   │       │   └── seed_24_comparison.xlsx
│   │   │   │       ├── seed_33
│   │   │   │       │   └── seed_33_comparison.xlsx
│   │   │   │       ├── seed_42
│   │   │   │       │   └── seed_42_comparison.xlsx
│   │   │   │       └── seed_50
│   │   │   │           └── seed_50_comparison.xlsx
│   │   │   ├── llm_ppo_input_reward_seed_10_20260816_010414
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_reward_seed_24_20260815_214641
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_reward_seed_33_20260815_201033
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_reward_seed_42_20260815_182704
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_reward_seed_50_20260815_232531
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_seed_10_20260816_001356
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_seed_24_20260815_205745
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_seed_33_20260815_191919
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_seed_42_20260815_173247
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_input_seed_50_20260815_223543
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_reward_seed_10_20260816_003743
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_reward_seed_24_20260815_212059
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_reward_seed_33_20260815_194648
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_reward_seed_42_20260815_180030
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── llm_ppo_reward_seed_50_20260815_230019
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── llm_rl_results.xlsx
│   │   │   ├── ppo_baseline_seed_10_20260815_235021
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── ppo_results.xlsx
│   │   │   ├── ppo_baseline_seed_24_20260815_203543
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── ppo_results.xlsx
│   │   │   ├── ppo_baseline_seed_33_20260815_185300
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── ppo_results.xlsx
│   │   │   ├── ppo_baseline_seed_42_20260815_170644
│   │   │   │   ├── raw
│   │   │   │   │   ├── backend_evaluation_episodes.json
│   │   │   │   │   ├── backend_training_episodes.json
│   │   │   │   │   ├── evaluation_runtime.json
│   │   │   │   │   ├── runtime_config.json
│   │   │   │   │   └── training_runtime.json
│   │   │   │   └── ppo_results.xlsx
│   │   │   └── ppo_baseline_seed_50_20260815_221231
│   │   │       ├── raw
│   │   │       │   ├── backend_evaluation_episodes.json
│   │   │       │   ├── backend_training_episodes.json
│   │   │       │   ├── evaluation_runtime.json
│   │   │       │   ├── runtime_config.json
│   │   │       │   └── training_runtime.json
│   │   │       └── ppo_results.xlsx
│   │   └── models
│   │       ├── llm_ppo_input_reward_seed_10_20260816_010414
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
│   │       ├── llm_ppo_input_reward_seed_24_20260815_214641
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
│   │       ├── llm_ppo_input_reward_seed_33_20260815_201033
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
│   │       ├── llm_ppo_input_reward_seed_42_20260815_182704
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
│   │       ├── llm_ppo_input_reward_seed_50_20260815_232531
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
│   │       ├── llm_ppo_input_seed_10_20260816_001356
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
│   │       ├── llm_ppo_input_seed_24_20260815_205745
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
│   │       ├── llm_ppo_input_seed_33_20260815_191919
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
│   │       ├── llm_ppo_input_seed_42_20260815_173247
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
│   │       ├── llm_ppo_input_seed_50_20260815_223543
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
│   │       ├── llm_ppo_reward_seed_10_20260816_003743
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
│   │       ├── llm_ppo_reward_seed_24_20260815_212059
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
│   │       ├── llm_ppo_reward_seed_33_20260815_194648
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
│   │       ├── llm_ppo_reward_seed_42_20260815_180030
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
│   │       ├── llm_ppo_reward_seed_50_20260815_230019
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
│   │       ├── ppo_baseline_seed_10_20260815_235021
│   │       │   ├── ppo_episode_100.pt
│   │       │   ├── ppo_episode_1000.pt
│   │       │   ├── ppo_episode_200.pt
│   │       │   ├── ppo_episode_300.pt
│   │       │   ├── ppo_episode_400.pt
│   │       │   ├── ppo_episode_500.pt
│   │       │   ├── ppo_episode_600.pt
│   │       │   ├── ppo_episode_700.pt
│   │       │   ├── ppo_episode_800.pt
│   │       │   ├── ppo_episode_900.pt
│   │       │   └── ppo_final.pt
│   │       ├── ppo_baseline_seed_24_20260815_203543
│   │       │   ├── ppo_episode_100.pt
│   │       │   ├── ppo_episode_1000.pt
│   │       │   ├── ppo_episode_200.pt
│   │       │   ├── ppo_episode_300.pt
│   │       │   ├── ppo_episode_400.pt
│   │       │   ├── ppo_episode_500.pt
│   │       │   ├── ppo_episode_600.pt
│   │       │   ├── ppo_episode_700.pt
│   │       │   ├── ppo_episode_800.pt
│   │       │   ├── ppo_episode_900.pt
│   │       │   └── ppo_final.pt
│   │       ├── ppo_baseline_seed_33_20260815_185300
│   │       │   ├── ppo_episode_100.pt
│   │       │   ├── ppo_episode_1000.pt
│   │       │   ├── ppo_episode_200.pt
│   │       │   ├── ppo_episode_300.pt
│   │       │   ├── ppo_episode_400.pt
│   │       │   ├── ppo_episode_500.pt
│   │       │   ├── ppo_episode_600.pt
│   │       │   ├── ppo_episode_700.pt
│   │       │   ├── ppo_episode_800.pt
│   │       │   ├── ppo_episode_900.pt
│   │       │   └── ppo_final.pt
│   │       ├── ppo_baseline_seed_42_20260815_170644
│   │       │   ├── ppo_episode_100.pt
│   │       │   ├── ppo_episode_1000.pt
│   │       │   ├── ppo_episode_200.pt
│   │       │   ├── ppo_episode_300.pt
│   │       │   ├── ppo_episode_400.pt
│   │       │   ├── ppo_episode_500.pt
│   │       │   ├── ppo_episode_600.pt
│   │       │   ├── ppo_episode_700.pt
│   │       │   ├── ppo_episode_800.pt
│   │       │   ├── ppo_episode_900.pt
│   │       │   └── ppo_final.pt
│   │       └── ppo_baseline_seed_50_20260815_221231
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
│   │   ├── experiment_suite.py
│   │   └── train.py
│   ├── utils
│   │   ├── comparison_metrics.py
│   │   ├── comparison_visualization.py
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
│   ├── README.md
│   ├── requirements.txt
│   ├── test.py
│   └── visualize.py
├── .gitignore
├── agent_folder_structure.md
├── generate_folder_structure.py
├── LLM_RL Experimental Methodology.pdf
├── LLM_RL System Architecture.pdf
└── README.md
```