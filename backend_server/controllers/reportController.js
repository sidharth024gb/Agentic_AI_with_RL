import Transaction from "../models/Transaction.js";
import AuditLog from "../models/AuditLog.js";
import Episode from "../models/Episode.js";

import { REWARDS } from "../utils/rewards.js";

// =====================================
// Get Transactions
// GET /api/report/transactions
// =====================================

export async function getTransactions(req, res) {
  try {
    const { status, invoice, limit = 50, page = 1 } = req.query;

    const filter = {};

    if (status) {
      filter.status = status;
    }

    if (invoice) {
      filter.invoice = invoice;
    }

    const transactions = await Transaction.find(filter)
      .populate("invoice", "invoiceNumber amount status")
      .populate("account", "accountNumber accountName")
      .sort({
        createdAt: -1,
      })
      .limit(Number(limit))
      .skip((Number(page) - 1) * Number(limit));

    const total = await Transaction.countDocuments(filter);

    return res.json({
      success: true,

      message: "Transactions retrieved.",

      reward: REWARDS.SUCCESS,

      count: transactions.length,

      total,

      page: Number(page),

      transactions,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      reward: null,

      message: error.message,
    });
  }
}

// =====================================
// Generate Report
// POST /api/report/generate-report
// =====================================

export async function generateReport(req, res) {
  try {
    const { startDate, endDate, type } = req.body;

    const filter = {};

    if (startDate && endDate) {
      filter.createdAt = {
        $gte: new Date(startDate),
        $lte: new Date(endDate),
      };
    }

    let report = {};

    switch (type) {
      case "TRANSACTION_SUMMARY":
        const transactions = await Transaction.find(filter);

        report = {
          type,

          totalTransactions: transactions.length,

          successfulPayments: transactions.filter((t) => t.status === "SUCCESS")
            .length,

          failedPayments: transactions.filter((t) => t.status === "FAILED")
            .length,

          totalAmount: transactions.reduce((sum, t) => sum + t.amount, 0),
        };

        break;

      case "AUDIT_SUMMARY":
        const logs = await AuditLog.find(filter);

        report = {
          type,

          totalActions: logs.length,

          successfulActions: logs.filter((l) => l.success).length,

          failedActions: logs.filter((l) => !l.success).length,
        };

        break;

      case "AGENT_PERFORMANCE":
        const episodes = await Episode.find(filter);

        report = {
          type,

          episodes: episodes.length,

          averageReward: episodes.length
            ? episodes.reduce((sum, e) => sum + e.totalReward, 0) /
              episodes.length
            : 0,

          successRate: episodes.length
            ? (episodes.filter((e) => e.completed).length / episodes.length) *
              100
            : 0,
        };

        break;

      default:
        return res.status(400).json({
          success: false,

          reward: null,

          message: "Invalid report type.",
        });
    }

    return res.json({
      success: true,

      message: "Report generated.",

      reward: REWARDS.REPORT_GENERATED,

      report,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      reward: null,

      message: error.message,
    });
  }
}

// =====================================
// Get Audit Logs
// GET /api/report/audit-log
// =====================================

export async function getAuditLog(req, res) {
  try {
    const {
      action,

      entityType,

      success,

      limit = 100,
    } = req.query;

    const filter = {};

    if (action) {
      filter.action = action;
    }

    if (entityType) {
      filter.entityType = entityType;
    }

    if (success !== undefined) {
      filter.success = success === "true";
    }

    const logs = await AuditLog.find(filter)
      .populate("user", "username email role")
      .sort({
        createdAt: -1,
      })
      .limit(Number(limit));

    return res.json({
      success: true,

      message: "Audit logs retrieved.",

      reward: REWARDS.SUCCESS,

      count: logs.length,

      logs,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      reward: null,

      message: error.message,
    });
  }
}
