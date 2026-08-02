import Account from "../models/Account.js";
import Budget from "../models/Budget.js";
import Transaction from "../models/Transaction.js";

import { REWARDS } from "../utils/rewards.js";

/*
GET /api/account

Return all available accounts
Agent observes treasury state
*/
export async function getAccounts(req, res) {
  try {
    const accounts = await Account.find();

    return res.json({
      success: true,

      reward: REWARDS.NONE,

      done: false,

      accounts,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      message: error.message,
    });
  }
}

/*
POST /api/account/budget/check


Input:

{
    department:"IT",
    amount:5000
}


Checks if payment is allowed
*/
export async function checkBudget(req, res) {
  try {
    const { department, amount } = req.body;

    // Missing required fields
    if (!department || amount == null) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        message: "department and amount are required.",
      });
    }

    const budget = await Budget.findOne({
      department,
    });

    if (!budget) {
      return res.status(404).json({
        success: false,

        reward: REWARDS.BUDGET_EXCEEDED,

        done: false,

        reason: "Budget not found",
      });
    }

    if (budget.remainingBudget < amount) {
      return res.json({
        success: false,

        reward: REWARDS.BUDGET_EXCEEDED,

        done: false,

        reason: "Insufficient budget",

        remainingBudget: budget.remainingBudget,
      });
    }

    return res.json({
      success: true,

      reward: REWARDS.BUDGET_CHECK_SUCCESS,

      done: false,

      budget: {
        department,
        remainingBudget: budget.remainingBudget,
      },
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      message: error.message,
    });
  }
}

/*
POST /api/account/transfer


Transfer money between accounts


Body:

{
 fromAccount:"id",
 toAccount:"id",
 amount:1000
}

*/
export async function transferMoney(req, res) {
  try {
    const { fromAccount, toAccount, amount } = req.body;

    // Missing required fields
    if (!fromAccount || !toAccount || amount == null) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        message: "fromAccount, toAccount and amount are required.",
      });
    }

    const source = await Account.findById(fromAccount);

    const destination = await Account.findById(toAccount);

    if (!source || !destination) {
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        reason: "Account not found",
      });
    }

    if (source.frozen) {
      return res.status(400).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        reason: "Source account frozen",
      });
    }

    if (source.balance < amount) {
      return res.status(400).json({
        success: false,

        reward: REWARDS.INSUFFICIENT_BALANCE,

        done: false,

        reason: "Insufficient balance",
      });
    }

    if (amount > source.dailyTransferLimit) {
      return res.status(400).json({
        success: false,

        reward: REWARDS.BUDGET_EXCEEDED,

        done: false,

        reason: "Transfer limit exceeded",
      });
    }

    source.balance -= amount;

    destination.balance += amount;

    await source.save();

    await destination.save();

    const transaction = await Transaction.create({
      transactionId: `TXN-${Date.now()}`,

      account: source._id,

      amount,

      status: "SUCCESS",

      type: "PAYMENT_OUT",
    });

    return res.status(201).json({
      success: true,

      reward: REWARDS.SUCCESS,

      done: false,

      transaction,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      message: error.message,
    });
  }
}

/*
GET /api/account/cash-position


Returns total available cash
*/
export async function cashPosition(req, res) {
  try {
    const accounts = await Account.find({
      frozen: false,
    });

    const totalCash = accounts.reduce(
      (sum, account) => sum + account.balance,
      0,
    );

    return res.json({
      success: true,

      reward: REWARDS.SUCCESS,

      done: false,

      cashPosition: {
        totalCash,

        accounts: accounts.length,
      },
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      message: error.message,
    });
  }
}
