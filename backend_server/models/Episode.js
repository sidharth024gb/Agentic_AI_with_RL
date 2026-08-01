import { Schema, model } from "mongoose";

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

    endpoint: {
      type: String,
    },

    reward: {
      type: Number,
      default: 0,
    },

    success: {
      type: Boolean,
      default: true,
    },

    message: {
      type: String,
    },

    stateBefore: {
      type: Schema.Types.Mixed,
    },

    stateAfter: {
      type: Schema.Types.Mixed,
    },

    timestamp: {
      type: Date,
      default: Date.now,
    },
  },
  { _id: false },
);

const episodeSchema = new Schema({
  episodeNumber: {
    type: Number,
    required: true,
    unique: true,
  },

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

  totalReward: {
    type: Number,
    default: 0,
  },

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

  completed: {
    type: Boolean,
    default: false,
  },

  terminatedReason: {
    type: String,
    enum: ["GOAL_REACHED", "MAX_STEPS", "FAILED", "TIMEOUT"],
  },

  executionTimeMs: {
    type: Number,
  },

  llmPlan: [
    {
      type: String,
    },
  ],

  actionSequence: [stepSchema],

  createdAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("Episode", episodeSchema);
