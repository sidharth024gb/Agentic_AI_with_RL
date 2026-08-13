import Account from "../models/Account.js";
import Budget from "../models/Budget.js";
import Transaction from "../models/Transaction.js";

import { REWARDS } from "../utils/rewards.js";

// ==========================================================
// GET ACCOUNTS
// ==========================================================

export async function getAccounts(req, res) {
  try {
    const accounts = await Account.find();

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.NONE,
      done: false,
      accounts,
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

// ==========================================================
// CHECK BUDGET
// ==========================================================

export async function checkBudget(req, res) {
  try {
    const { department, amount = null } = req.body;

    // ======================================================
    // Validate Department
    // ======================================================

    if (!department) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "department is required.",
      });
    }

    // ======================================================
    // Validate Optional Amount
    //
    // amount can be omitted when the environment only needs
    // current remaining budget information.
    // ======================================================

    let requestedAmount = null;

    if (amount !== null && amount !== undefined) {
      requestedAmount = Number(amount);

      if (!Number.isFinite(requestedAmount) || requestedAmount < 0) {
        return res.status(400).json({
          success: false,
          environmentError: false,
          reward: null,
          errorType: "INVALID_REQUEST",
          message: "amount must be a non-negative number.",
        });
      }
    }

    // ======================================================
    // Find Budget
    // ======================================================

    const budget = await Budget.findOne({
      department,
    });

    // ======================================================
    // Budget Missing
    //
    // The CHECK_BUDGET action itself worked.
    //
    // It discovered that this department has no budget.
    // ======================================================

    if (!budget) {
      return res.status(200).json({
        success: true,
        environmentError: false,
        reward: REWARDS.NONE,
        done: false,

        found: false,
        eligible: false,
        withinBudget: false,

        reason: "BUDGET_NOT_FOUND",

        requestedAmount,

        budget: null,

        message: `No budget found for department ${department}.`,
      });
    }

    // ======================================================
    // Current Budget Information
    // ======================================================

    const remainingBudget = Number(budget.remainingBudget);

    // If no amount was supplied, this endpoint is simply
    // returning current budget state.

    const withinBudget =
      requestedAmount === null ? null : requestedAmount <= remainingBudget;

    // ======================================================
    // IMPORTANT
    //
    // Insufficient budget is not a failed CHECK_BUDGET action.
    //
    // The action successfully discovered that the requested
    // amount does not fit.
    // ======================================================

    return res.status(200).json({
      success: true,
      environmentError: false,

      reward: REWARDS.NONE,

      done: false,

      found: true,

      eligible: requestedAmount === null ? true : withinBudget,

      withinBudget,

      reason: withinBudget === false ? "BUDGET_EXCEEDED" : null,

      requestedAmount,

      budget: {
        id: budget._id,
        department: budget.department,
        monthlyBudget: budget.monthlyBudget,
        remainingBudget,
      },

      message:
        withinBudget === false
          ? "Requested amount exceeds the remaining budget."
          : "Budget information retrieved successfully.",
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

// ==========================================================
// TRANSFER MONEY
// ==========================================================

export async function transferMoney(req, res) {
  try {
    const { fromAccount, toAccount, amount } = req.body;

    if (!fromAccount || !toAccount || amount == null) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "fromAccount, toAccount and amount are required.",
      });
    }

    const numericAmount = Number(amount);

    if (!Number.isFinite(numericAmount) || numericAmount <= 0) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "amount must be greater than zero.",
      });
    }

    const source = await Account.findById(fromAccount);

    const destination = await Account.findById(toAccount);

    if (!source || !destination) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_ACTION,
        done: false,
        errorType: "ACCOUNT_NOT_FOUND",
        message: "Account not found.",
      });
    }

    if (source.frozen) {
      return res.status(409).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_ACTION,
        done: false,
        errorType: "ACCOUNT_FROZEN",
        message: "Source account is frozen.",
      });
    }

    if (source.balance < numericAmount) {
      return res.status(409).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INSUFFICIENT_BALANCE,
        done: false,
        errorType: "INSUFFICIENT_BALANCE",
        message: "Insufficient balance.",
      });
    }

    if (numericAmount > source.dailyTransferLimit) {
      return res.status(409).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_ACTION,
        done: false,
        errorType: "TRANSFER_LIMIT_EXCEEDED",
        message: "Transfer limit exceeded.",
      });
    }

    source.balance -= numericAmount;

    destination.balance += numericAmount;

    await source.save();
    await destination.save();

    const transaction = await Transaction.create({
      transactionId: `TXN-${Date.now()}`,
      account: source._id,
      amount: numericAmount,
      status: "SUCCESS",
      type: "PAYMENT_OUT",
    });

    return res.status(201).json({
      success: true,
      environmentError: false,
      reward: REWARDS.SUCCESS,
      done: false,
      transaction,
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

// ==========================================================
// CASH POSITION
// ==========================================================

export async function cashPosition(req, res) {
  try {
    const accounts = await Account.find({
      frozen: false,
    });

    const totalCash = accounts.reduce(
      (sum, account) => sum + Number(account.balance || 0),
      0,
    );

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.NONE,
      done: false,

      cashPosition: {
        totalCash,
        totalAccounts: accounts.length,
        accounts,
      },
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
