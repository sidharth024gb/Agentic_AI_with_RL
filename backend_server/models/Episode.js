import { Schema, model } from "mongoose";

// ==========================================================
// Step Schema
// ==========================================================

const stepSchema = new Schema(
  {
    stepNumber: {
      type: Number,
      required: true,
    },

    action: {
      type: String,
      required: true,
    },

    /*
     * Logical endpoint used by the high-level action.
     *
     * Some actions may internally call the same endpoint
     * multiple times, so this is descriptive rather than
     * representing every individual HTTP request.
     */
    endpoint: {
      type: String,
      default: null,
    },

    /*
     * Reward before LLM guidance shaping.
     */
    baseReward: {
      type: Number,
      default: 0,
    },

    /*
     * Extra reward added because the LLM procedure
     * was followed.
     *
     * RL-only episodes should normally keep this at 0.
     */
    guidanceBonus: {
      type: Number,
      default: 0,
    },

    /*
     * Final reward actually seen by PPO.
     *
     * reward = baseReward + guidanceBonus
     */
    reward: {
      type: Number,
      default: 0,
    },

    /*
     * Did the high-level action execute successfully?
     */
    success: {
      type: Boolean,
      default: true,
    },

    /*
     * An action may succeed technically but have
     * nothing useful to do.
     *
     * Example:
     * APPROVE_INVOICES when no pending invoices exist.
     */
    usefulAction: {
      type: Boolean,
      default: true,
    },

    /*
     * Infrastructure/backend failure.
     *
     * This is NOT an agent mistake.
     */
    environmentError: {
      type: Boolean,
      default: false,
    },

    /*
     * Used only for LLM-guided experiments.
     *
     * null = not applicable
     */
    procedureFollowed: {
      type: Boolean,
      default: null,
    },

    message: {
      type: String,
      default: null,
    },

    stateBefore: {
      type: Schema.Types.Mixed,
    },

    stateAfter: {
      type: Schema.Types.Mixed,
    },

    /*
     * Execution time of this high-level action.
     */
    durationMs: {
      type: Number,
      default: 0,
    },

    timestamp: {
      type: Date,
      default: Date.now,
    },
  },
  {
    _id: false,
  },
);

// ==========================================================
// Episode Schema
// ==========================================================

const episodeSchema = new Schema({
  episodeNumber: {
    type: Number,
    required: true,
    unique: true,
  },

  // ========================================================
  // Experiment Metadata
  // ========================================================

  agentType: {
    type: String,
    enum: ["RL", "LLM_RL"],
    required: true,
  },

  algorithm: {
    type: String,
    enum: ["Q_LEARNING", "DQN", "PPO", "SAC"],
    required: true,
  },

  /*
   * Helps prevent training episodes being mixed
   * with final evaluation episodes.
   */
  phase: {
    type: String,
    enum: ["TRAIN", "EVALUATION", "TEST"],
    default: "TRAIN",
  },

  experimentName: {
    type: String,
    default: null,
  },

  seed: {
    type: Number,
    default: null,
  },

  llmModel: {
    type: String,
    default: null,
  },

  /*
   * Version of the prompt used to generate the LLM plan.
   *
   * Example:
   * finance_planner_v2
   *
   * RL-only episodes:
   * null
   */
  promptVersion: {
    type: String,
    default: null,
  },

  /*
   * Whether this episode reused a previously cached LLM plan.
   *
   * RL-only episodes:
   * false
   */
  llmPlanCached: {
    type: Boolean,
    default: false,
  },

  /*
   * Time spent generating the LLM plan for this episode.
   *
   * For a cache hit this is recorded as 0 because no new
   * LLM generation occurred.
   *
   * RL-only episodes:
   * 0
   */
  llmPlanningTimeMs: {
    type: Number,
    default: 0,
  },

  /*
   * Allows dissertation ablation experiments:
   *
   * NONE
   * INPUT
   * REWARD_SHAPING
   * INPUT_AND_REWARD
   */
  guidanceMode: {
    type: String,
    enum: ["NONE", "INPUT", "REWARD_SHAPING", "INPUT_AND_REWARD"],
    default: "NONE",
  },

  // ========================================================
  // Goal / State
  // ========================================================

  goal: {
    type: String,
    required: true,
  },

  initialState: {
    type: Schema.Types.Mixed,
  },

  finalState: {
    type: Schema.Types.Mixed,
  },

  llmPlan: [
    {
      type: String,
    },
  ],

  // ========================================================
  // Reward Metrics
  // ========================================================

  totalBaseReward: {
    type: Number,
    default: 0,
  },

  totalGuidanceBonus: {
    type: Number,
    default: 0,
  },

  /*
   * Final reward PPO actually received.
   */
  totalReward: {
    type: Number,
    default: 0,
  },

  // ========================================================
  // Action Metrics
  // ========================================================

  totalSteps: {
    type: Number,
    default: 0,
  },

  successfulActions: {
    type: Number,
    default: 0,
  },

  failedActions: {
    type: Number,
    default: 0,
  },

  /*
   * Successfully executed actions that had
   * nothing useful to do.
   */
  noOpActions: {
    type: Number,
    default: 0,
  },

  /*
   * Environment failures are deliberately kept
   * separate from agent failures.
   */
  environmentErrors: {
    type: Number,
    default: 0,
  },

  // ========================================================
  // Completion Metrics
  // ========================================================

  completed: {
    type: Boolean,
    default: false,
  },

  terminatedReason: {
    type: String,
    enum: [
      "GOAL_REACHED",
      "MAX_STEPS",
      "FAILED",
      "TIMEOUT",
      "ENVIRONMENT_ERROR",
      "RESET",
    ],
  },

  executionTimeMs: {
    type: Number,
    default: null,
  },

  actionSequence: [stepSchema],

  createdAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("Episode", episodeSchema);
