import Episode from "../models/Episode.js";
import { REWARDS } from "../utils/rewards.js";

// Start Episode

export const startEpisode = async (req, res) => {
  try {
    const lastEpisode = await Episode.findOne().sort({ episodeNumber: -1 });

    const episode = await Episode.create({
      episodeNumber: lastEpisode ? lastEpisode.episodeNumber + 1 : 1,

      agentType: req.body.agentType,

      algorithm: req.body.algorithm,

      goal: req.body.goal,

      initialState: req.body.initialState || {},

      llmPlan: req.body.llmPlan || [],
    });

    return res.status(201).json({
      success: true,

      message: "Episode started.",

      episodeId: episode._id,

      episodeNumber: episode.episodeNumber,

      reward: REWARDS.NONE,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      message: err.message,

      reward: REWARDS.SYSTEM_ERROR,
    });
  }
};

// Record One Step

export const addStep = async (req, res) => {
  try {
    const {
      action,

      endpoint,

      reward,

      success,

      message,

      stateBefore,

      stateAfter,
    } = req.body;

    const episode = await Episode.findById(req.params.id);

    if (!episode) {
      return res.status(404).json({
        success: false,

        message: "Episode not found",

        reward: REWARDS.NONE,
      });
    }

    episode.actionSequence.push({
      stepNumber: episode.totalSteps + 1,

      action,

      endpoint,

      reward,

      success,

      message,

      stateBefore,

      stateAfter,
    });

    episode.totalReward += reward;

    episode.totalSteps += 1;

    if (success) episode.successfulActions++;
    else episode.failedActions++;

    await episode.save();

    return res.status(200).json({
      success: true,

      message: "Step recorded.",

      reward,

      totalReward: episode.totalReward,

      totalSteps: episode.totalSteps,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      message: err.message,

      reward: REWARDS.SYSTEM_ERROR,
    });
  }
};

// End Episode

export const endEpisode = async (req, res) => {
  try {
    const {
      finalState,
      completed,
      terminatedReason,
    } = req.body;

    const episode = await Episode.findById(req.params.id);

    if (!episode) {
      return res.status(404).json({
        success: false,

        message: "Episode not found",

        reward: REWARDS.SYSTEM_ERROR,
      });
    }

    episode.completed = completed;

    episode.terminatedReason = terminatedReason;

    episode.finalState = finalState;

    episode.executionTimeMs = Date.now() - episode.createdAt.getTime();

    await episode.save();

    return res.status(200).json({
      success: true,

      message: "Episode completed.",

      completed: episode.completed,

      totalReward: episode.totalReward,

      totalSteps: episode.totalSteps,

      successfulActions: episode.successfulActions,

      failedActions: episode.failedActions,

      executionTimeMs: episode.executionTimeMs,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      message: err.message,

      reward: REWARDS.SYSTEM_ERROR,
    });
  }
};

// Get Episode

export const getEpisode = async (req, res) => {
  try {
    const episode = await Episode.findById(req.params.id);

    if (!episode) {
      return res.status(404).json({
        success: false,

        message: "Episode not found",
      });
    }

    return res.status(200).json({
      success: true,

      episode,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      message: err.message,
    });
  }
};

// Get All Episodes

export const getEpisodes = async (req, res) => {
  try {
    const episodes = await Episode.find().sort({ episodeNumber: -1 });

    return res.status(200).json({
      success: true,

      count: episodes.length,

      episodes,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      message: err.message,
    });
  }
};
