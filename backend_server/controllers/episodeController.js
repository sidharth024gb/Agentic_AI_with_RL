import Episode from "../models/Episode.js";

// ==========================================================
// START EPISODE
// ==========================================================

export const startEpisode = async (req, res) => {
  try {
    const lastEpisode = await Episode.findOne().sort({
      episodeNumber: -1,
    });

    // ------------------------------------------------------
    // Safely normalize LLM planning latency
    // ------------------------------------------------------

    const llmPlanningTimeMs =
      req.body.llmPlanningTimeMs !== undefined &&
      req.body.llmPlanningTimeMs !== null &&
      Number.isFinite(Number(req.body.llmPlanningTimeMs))
        ? Number(req.body.llmPlanningTimeMs)
        : 0;

    const episode = await Episode.create({
      episodeNumber: lastEpisode
        ? lastEpisode.episodeNumber + 1
        : 1,

      // ====================================================
      // Agent / Experiment
      // ====================================================

      agentType: req.body.agentType,

      algorithm: req.body.algorithm,

      phase: req.body.phase || "TRAIN",

      experimentName:
        req.body.experimentName || null,

      seed:
        req.body.seed ?? null,

      // ====================================================
      // LLM Metadata
      // ====================================================

      llmModel:
        req.body.llmModel || null,

      promptVersion:
        req.body.promptVersion || null,

      llmPlanCached:
        req.body.llmPlanCached === true,

      llmPlanningTimeMs,

      guidanceMode:
        req.body.guidanceMode || "NONE",

      llmPlan:
        Array.isArray(req.body.llmPlan)
          ? req.body.llmPlan
          : [],

      // ====================================================
      // Goal / Initial State
      // ====================================================

      goal: req.body.goal,

      initialState:
        req.body.initialState || {},
    });

    return res.status(201).json({
      success: true,

      message: "Episode started.",

      episodeId: episode._id,

      episodeNumber: episode.episodeNumber,
    });

  } catch (err) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      message: err.message,
    });
  }
};

// ==========================================================
// RECORD STEP
// ==========================================================

export const addStep = async (req, res) => {
  try {
    const {
      action,

      endpoint,

      baseReward,

      guidanceBonus,

      reward,

      success = true,

      usefulAction = true,

      environmentError = false,

      procedureFollowed = null,

      message,

      stateBefore,

      stateAfter,

      durationMs = 0,
    } = req.body;

    const episode = await Episode.findById(req.params.id);

    if (!episode) {
      return res.status(404).json({
        success: false,

        message: "Episode not found.",
      });
    }

    // ------------------------------------------------------
    // Normalise rewards
    // ------------------------------------------------------

    const finalReward = Number.isFinite(Number(reward)) ? Number(reward) : 0;

    const guidanceReward = Number.isFinite(Number(guidanceBonus))
      ? Number(guidanceBonus)
      : 0;

    const base =
      baseReward !== undefined &&
      baseReward !== null &&
      Number.isFinite(Number(baseReward))
        ? Number(baseReward)
        : finalReward - guidanceReward;

    // ------------------------------------------------------
    // Store step
    // ------------------------------------------------------

    episode.actionSequence.push({
      stepNumber: episode.totalSteps + 1,

      action,

      endpoint: endpoint || null,

      baseReward: base,

      guidanceBonus: guidanceReward,

      reward: finalReward,

      success: Boolean(success),

      usefulAction: Boolean(usefulAction),

      environmentError: Boolean(environmentError),

      procedureFollowed,

      message: message || null,

      stateBefore,

      stateAfter,

      durationMs: Number(durationMs) || 0,
    });

    // ------------------------------------------------------
    // Episode reward totals
    // ------------------------------------------------------

    episode.totalBaseReward += base;

    episode.totalGuidanceBonus += guidanceReward;

    episode.totalReward += finalReward;

    episode.totalSteps += 1;

    // ------------------------------------------------------
    // Action statistics
    //
    // Environment errors must NOT count as agent failures.
    // ------------------------------------------------------

    if (environmentError) {
      episode.environmentErrors += 1;
    } else if (success) {
      episode.successfulActions += 1;

      if (!usefulAction) {
        episode.noOpActions += 1;
      }
    } else {
      episode.failedActions += 1;
    }

    await episode.save();

    return res.status(200).json({
      success: true,

      message: "Step recorded.",

      step: episode.totalSteps,

      reward: finalReward,

      totalBaseReward: episode.totalBaseReward,

      totalGuidanceBonus: episode.totalGuidanceBonus,

      totalReward: episode.totalReward,

      successfulActions: episode.successfulActions,

      failedActions: episode.failedActions,

      noOpActions: episode.noOpActions,

      environmentErrors: episode.environmentErrors,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      message: err.message,
    });
  }
};

// ==========================================================
// END EPISODE
// ==========================================================

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

        message: "Episode not found.",
      });
    }

    episode.completed = Boolean(completed);

    episode.terminatedReason = terminatedReason;

    episode.finalState = finalState || {};

    episode.executionTimeMs = Date.now() - episode.createdAt.getTime();

    await episode.save();

    return res.status(200).json({
      success: true,

      message: "Episode completed.",

      episodeNumber: episode.episodeNumber,

      completed: episode.completed,

      terminatedReason: episode.terminatedReason,

      totalReward: episode.totalReward,

      totalBaseReward: episode.totalBaseReward,

      totalGuidanceBonus: episode.totalGuidanceBonus,

      totalSteps: episode.totalSteps,

      successfulActions: episode.successfulActions,

      failedActions: episode.failedActions,

      noOpActions: episode.noOpActions,

      environmentErrors: episode.environmentErrors,

      executionTimeMs: episode.executionTimeMs,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      message: err.message,
    });
  }
};

// ==========================================================
// GET EPISODE
// ==========================================================

export const getEpisode = async (req, res) => {
  try {
    const episode = await Episode.findById(req.params.id);

    if (!episode) {
      return res.status(404).json({
        success: false,

        message: "Episode not found.",
      });
    }

    return res.status(200).json({
      success: true,

      episode,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      message: err.message,
    });
  }
};

// ==========================================================
// GET ALL EPISODES
// ==========================================================

export const getEpisodes = async (req, res) => {
  try {
    const filter = {};

    if (req.query.experimentName) {
      filter.experimentName = req.query.experimentName;
    }

    if (req.query.phase) {
      filter.phase = req.query.phase;
    }

    if (req.query.agentType) {
      filter.agentType = req.query.agentType;
    }

    if (req.query.algorithm) {
      filter.algorithm = req.query.algorithm;
    }

    const episodes = await Episode.find(filter)
      .sort({
        episodeNumber: 1,
      })
      .lean();

    return res.status(200).json({
      success: true,

      count: episodes.length,

      episodes,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      message: err.message,
    });
  }
};
