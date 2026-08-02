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

    // Validate pagination params
    if (
      isNaN(Number(limit)) ||
      isNaN(Number(page)) ||
      Number(limit) <= 0 ||
      Number(page) <= 0
    ) {
      return res.status(400).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        message:
          "Invalid pagination parameters. Limit and page must be positive numbers.",
      });
    }

    const filter = {};

    // Validate status filter
    if (status) {
      const allowedStatus = ["PENDING", "SUCCESS", "FAILED", "REVERSED"];

      if (!allowedStatus.includes(status)) {
        return res.status(400).json({
          success: false,

          reward: REWARDS.INVALID_ACTION,

          done: false,

          message: "Invalid transaction status.",
        });
      }

      filter.status = status;
    }

    // Validate invoice id
    if (invoice) {
      if (!invoice.match(/^[0-9a-fA-F]{24}$/)) {
        return res.status(400).json({
          success: false,

          reward: REWARDS.INVALID_ACTION,

          done: false,

          message: "Invalid invoice ID.",
        });
      }

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

    return res.status(200).json({
      success: true,

      message: "Transactions retrieved.",

      reward: REWARDS.SUCCESS,

      done: false,

      count: transactions.length,

      total,

      page: Number(page),

      transactions,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      retryable: true,

      reward: null,

      done: false,

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

    // Validate required parameter
    if (!type) {
      return res.status(400).json({
        success: false,
        reward: null,
        environmentError: false,
        message: "Report type is required.",
      });
    }

    const allowedTypes = [
      "TRANSACTION_SUMMARY",
      "AUDIT_SUMMARY",
      "AGENT_PERFORMANCE",
    ];

    if (!allowedTypes.includes(type)) {
      return res.status(400).json({
        success: false,
        reward: null,
        environmentError: false,
        message: "Invalid report type.",
      });
    }

    // Validate dates if provided
    let filter = {};

    if (startDate || endDate) {
      if (!startDate || !endDate) {
        return res.status(400).json({
          success: false,
          reward: null,
          environmentError: false,
          message:
            "Both startDate and endDate are required when filtering by date.",
        });
      }

      const start = new Date(startDate);
      const end = new Date(endDate);

      if (isNaN(start.getTime()) || isNaN(end.getTime())) {
        return res.status(400).json({
          success: false,
          reward: null,
          environmentError: false,
          message: "Invalid date format.",
        });
      }

      if (start > end) {
        return res.status(400).json({
          success: false,
          reward: null,
          environmentError: false,
          message: "startDate cannot be greater than endDate.",
        });
      }

      filter.createdAt = {
        $gte: start,
        $lte: end,
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
          environmentError: false,
          message: "Invalid report type.",
        });
    }

    return res.status(200).json({
      success: true,

      message: "Report generated.",

      reward: REWARDS.REPORT_GENERATED,

      environmentError: false,

      report,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      retryable: true,

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
    const { action, entityType, success, limit = 100 } = req.query;

    // Validate limit
    const parsedLimit = Number(limit);

    if (isNaN(parsedLimit) || parsedLimit <= 0 || parsedLimit > 500) {
      return res.status(400).json({
        success: false,
        reward: null,
        environmentError: false,
        message: "Limit must be a number between 1 and 500.",
      });
    }

    const filter = {};

    // Validate action if provided
    if (action) {
      if (typeof action !== "string") {
        return res.status(400).json({
          success: false,
          reward: null,
          environmentError: false,
          message: "Invalid action parameter.",
        });
      }

      filter.action = action;
    }

    // Validate entity type if provided
    if (entityType) {
      if (typeof entityType !== "string") {
        return res.status(400).json({
          success: false,
          reward: null,
          environmentError: false,
          message: "Invalid entityType parameter.",
        });
      }

      filter.entityType = entityType;
    }

    // Validate success filter
    if (success !== undefined) {
      if (success !== "true" && success !== "false") {
        return res.status(400).json({
          success: false,
          reward: null,
          environmentError: false,
          message: "success parameter must be true or false.",
        });
      }

      filter.success = success === "true";
    }

    const logs = await AuditLog.find(filter)
      .populate("user", "username email role")
      .sort({
        createdAt: -1,
      })
      .limit(parsedLimit);

    return res.status(200).json({
      success: true,

      message: "Audit logs retrieved.",

      reward: REWARDS.SUCCESS,

      environmentError: false,

      count: logs.length,

      logs,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      retryable: true,

      reward: null,

      message: error.message,
    });
  }
}
