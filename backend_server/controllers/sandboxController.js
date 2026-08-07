import Invoice from "../models/Invoice.js";
import Supplier from "../models/Supplier.js";
import Account from "../models/Account.js";
import Budget from "../models/Budget.js";
import Transaction from "../models/Transaction.js";
import Episode from "../models/Episode.js";

import { REWARDS } from "../utils/rewards.js";

export async function resetEnvironment(req, res) {
  try {
    const { episodeId } = req.body;

    /*
    ==================================
    1. Clear Previous Environment
    ==================================
    */

    await Invoice.deleteMany({});
    await Supplier.deleteMany({});
    await Transaction.deleteMany({});
    await Budget.deleteMany({});
    await Account.deleteMany({});

    /*
    ==================================
    Helper Functions
    ==================================
    */

    const randomNumber = (min, max) =>
      Math.floor(Math.random() * (max - min + 1)) + min;

    const randomChoice = (array) =>
      array[Math.floor(Math.random() * array.length)];

    /*
    ==================================
    2. Create Random Suppliers
    ==================================
    */

    const supplierCount = randomNumber(5, 15);

    const suppliers = [];

    for (let i = 1; i <= supplierCount; i++) {
      suppliers.push({
        supplierCode: `SUP-${1000 + i}`,

        supplierName: `Supplier Company ${i}`,

        isActive: Math.random() > 0.2,

        riskScore: randomNumber(1, 100),

        rating: randomNumber(1, 5),

        country: randomChoice(["UK", "USA", "Germany", "India"]),

        preferredPaymentMethod: randomChoice(["BANK", "CARD", "WIRE"]),
      });
    }

    const createdSuppliers = await Supplier.insertMany(suppliers);

    /*
    ==================================
    3. Create Random Accounts
    ==================================
    */

    const accountCount = randomNumber(2, 5);

    const accounts = [];

    for (let i = 1; i <= accountCount; i++) {
      accounts.push({
        accountNumber: `ACC-${1000 + i}`,

        accountName: `Account ${i}`,

        accountType: randomChoice([
          "TREASURY",
          "EXPENSE",
          "OPERATIONS",
          "PAYMENT",
        ]),

        currentBalance: randomNumber(20000, 200000),

        currency: "GBP",

        frozen: Math.random() < 0.1,

        dailyTransferLimit: randomNumber(5000, 50000),
      });
    }

    await Account.insertMany(accounts);

    /*
    ==================================
    4. Create Random Budgets
    ==================================
    */

    const departments = [
      "SOFTWARE",
      "TRAVEL",
      "HARDWARE",
      "MARKETING",
      "OPERATIONS",
    ];

    const budgetCount = randomNumber(3, 6);

    const budgets = [];

    for (let i = 0; i < budgetCount; i++) {
      const amount = randomNumber(10000, 100000);

      budgets.push({
        department: departments[i],

        monthlyBudget: amount,

        remainingBudget: randomNumber(0, amount),
      });
    }

    await Budget.insertMany(budgets);

    /*
    ==================================
    5. Create Random Invoices
    ==================================
    */

    const invoiceCount = randomNumber(20, 50);

    const categories = ["SOFTWARE", "SERVICES", "HARDWARE", "TRAVEL"];

    const statuses = ["PENDING_APPROVAL", "APPROVED", "REJECTED"];

    const invoices = [];

    for (let i = 1; i <= invoiceCount; i++) {
      const supplier = randomChoice(createdSuppliers);

      invoices.push({
        invoiceNumber: `INV-${10000 + i}`,

        supplier: supplier._id,

        amount: randomNumber(500, 20000),

        status: randomChoice(statuses),

        category: randomChoice(categories),

        priority: randomChoice(["LOW", "MEDIUM", "HIGH"]),

        paymentMethod: supplier.preferredPaymentMethod,

        requiresManagerApproval: Math.random() < 0.3,

        dueDate: new Date(Date.now() + randomNumber(1, 60) * 86400000),

        description: `Invoice generated for ${supplier.supplierName}`,
      });
    }

    await Invoice.insertMany(invoices);

    /*
    ==================================
    6. Create Transaction History
    ==================================
    */

    const transactionCount = randomNumber(5, 20);

    const transactions = [];

    for (let i = 1; i <= transactionCount; i++) {
      transactions.push({
        transactionId: `TXN-${10000 + i}`,

        amount: randomNumber(500, 10000),

        status: randomChoice(["PENDING", "SUCCESS", "FAILED", "REVERSED"]),

        paymentMethod: randomChoice(["BANK", "CARD", "WIRE"]),

        description: "Previous supplier transaction",
      });
    }

    await Transaction.insertMany(transactions);

    /*
    ==================================
    7. Reset Episode
    ==================================
    */

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

    /*
    ==================================
    8. Return Initial State
    ==================================
    */

    return res.json({
      success: true,

      reward: REWARDS.NONE,

      done: false,

      stateCreated: {
        suppliers: createdSuppliers.length,

        invoices: invoiceCount,

        accounts: accountCount,

        budgets: budgetCount,

        transactions: transactionCount,
      },

      message: "RL environment reset with random training scenario.",
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
        $in: ["PENDING_APPROVAL", "APPROVED"],
      },
    });

    const accounts = await Account.find();

    const suppliers = await Supplier.find();

    const budgets = await Budget.find();

    const transactions = await Transaction.find();

    return res.json({
      success: true,

      observation: {
        invoices,
        accounts,
        suppliers,
        budgets,
        transactions,
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

export async function getReward(req, res) {
  try {
    const { episodeId } = req.params;

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
