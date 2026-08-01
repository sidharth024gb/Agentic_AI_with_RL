import Invoice from "../models/Invoice.js";
import Account from "../models/Account.js";
import Supplier from "../models/Supplier.js";
import Episode from "../models/Episode.js";

import { REWARDS } from "../utils/rewards.js";

export async function resetEnvironment(req, res) {
  try {
    const { episodeId } = req.body;

    // Restore all invoices
    await Invoice.updateMany(
      {},
      {
        status: "PENDING",
        paidDate: null,
        approvedDate: null,
      },
    );

    // Restore account balances
    await Account.updateMany(
      {},
      {
        $set: {
          currentBalance: 100000,
        },
      },
    );

    // Activate suppliers
    await Supplier.updateMany(
      {},
      {
        $set: {
          isActive: true,
        },
      },
    );

    if (episodeId) {
      await Episode.findByIdAndUpdate(episodeId, {
        completed: false,
        totalReward: 0,
        totalSteps: 0,
        successfulActions: 0,
        failedActions: 0,
        actionSequence: [],
        finalState: null,
        terminatedReason: null,
        executionTimeMs: null,
      });
    }

    return res.json({
      success: true,
      reward: REWARDS.NONE,
      done: false,
      message: "Environment reset.",
    });
  } catch (err) {
    return res.status(500).json({
      success: false,
      environmentError: true,
      retryable: true,
      message: err.message,
    });
  }
}

export async function getState(req, res) {
  try {
    const invoices = await Invoice.find({
      status: {
        $in: ["PENDING", "APPROVED"],
      },
    });

    const accounts = await Account.find();

    const suppliers = await Supplier.find();

    return res.json({
      success: true,

      observation: {
        invoices,

        accounts,

        suppliers,
      },

      done: false,

      reward: REWARDS.NONE,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      retryable: true,

      message: err.message,
    });
  }
}

export async function randomizeEnvironment(req, res) {
  try {
    const invoices = await Invoice.find();

    for (const invoice of invoices) {
      const statuses = ["PENDING", "APPROVED"];

      invoice.status = statuses[Math.floor(Math.random() * statuses.length)];

      invoice.amount = Math.floor(Math.random() * 9000) + 1000;

      await invoice.save();
    }

    return res.json({
      success: true,

      reward: REWARDS.NONE,

      message: "Environment randomized.",
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      retryable: true,

      message: err.message,
    });
  }
}

export async function getReward(req, res) {
  try {
    const { episodeId } = req.query;

    const episode = await Episode.findById(episodeId);

    if (!episode) {
      return res.json({
        success: false,

        reward: null,

        done: true,

        message: "Episode not found.",
      });
    }

    const latestStep =
      episode.actionSequence[episode.actionSequence.length - 1];

    return res.json({
      success: true,

      reward: latestStep ? latestStep.reward : REWARDS.NONE,

      totalReward: episode.totalReward,

      done: episode.completed,

      terminatedReason: episode.terminatedReason,
    });
  } catch (err) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      retryable: true,

      message: err.message,
    });
  }
}
