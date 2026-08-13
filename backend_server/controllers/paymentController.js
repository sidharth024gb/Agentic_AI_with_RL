import Invoice from "../models/Invoice.js";
import Transaction from "../models/Transaction.js";
import Account from "../models/Account.js";
import Budget from "../models/Budget.js";

import { REWARDS } from "../utils/rewards.js";

// ==========================================================
// Configuration
// ==========================================================

const HIGH_RISK_THRESHOLD = 70;

// ==========================================================
// PAY INVOICE
// ==========================================================

export async function payInvoice(req, res) {
  try {
    const { invoiceId, accountId } = req.body;

    // ======================================================
    // Validate Request
    // ======================================================

    if (!invoiceId || !accountId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        done: false,
        errorType: "INVALID_REQUEST",
        message: "invoiceId and accountId are required.",
      });
    }

    // ======================================================
    // 1. FIND INVOICE
    // ======================================================

    const invoice = await Invoice.findById(invoiceId).populate("supplier");

    if (!invoice) {
      return res.status(404).json({
        success: false,
        environmentError: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        done: false,

        errorType: "INVOICE_NOT_FOUND",

        message: "Invoice not found.",
      });
    }

    // ======================================================
    // 2. INVOICE MUST BE APPROVED
    // ======================================================

    if (invoice.status !== "APPROVED") {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        errorType: "INVOICE_NOT_APPROVED",

        message: "Invoice is not approved.",
      });
    }

    // ======================================================
    // 3. DUPLICATE INVOICE
    // ======================================================

    if (invoice.duplicateFlag) {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.DUPLICATE_INVOICE,

        done: false,

        errorType: "DUPLICATE_INVOICE",

        message: "Duplicate invoice cannot be paid.",
      });
    }

    // ======================================================
    // 4. SUPPLIER EXISTS
    // ======================================================

    if (!invoice.supplier) {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.SUPPLIER_NOT_FOUND,

        done: false,

        errorType: "SUPPLIER_NOT_FOUND",

        message: "Supplier does not exist.",
      });
    }

    // ======================================================
    // 5. SUPPLIER ACTIVE
    // ======================================================

    if (!invoice.supplier.active) {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.SUPPLIER_INACTIVE,

        done: false,

        errorType: "SUPPLIER_INACTIVE",

        message: "Supplier is inactive.",
      });
    }

    // ======================================================
    // 6. SUPPLIER RISK
    //
    // Must match supplierController validation.
    // ======================================================

    if (Number(invoice.supplier.riskScore) > HIGH_RISK_THRESHOLD) {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.SUPPLIER_HIGH_RISK,

        done: false,

        errorType: "SUPPLIER_HIGH_RISK",

        message: "High-risk supplier cannot be paid.",
      });
    }

    // ======================================================
    // 7. FIND BUDGET
    // ======================================================

    const budget = await Budget.findOne({
      department: invoice.category,
    });

    if (!budget) {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        errorType: "BUDGET_NOT_FOUND",

        message: `No budget found for department ${invoice.category}.`,
      });
    }

    // ======================================================
    // 8. CHECK REMAINING BUDGET
    // ======================================================

    if (Number(budget.remainingBudget) < Number(invoice.amount)) {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.BUDGET_EXCEEDED,

        done: false,

        errorType: "BUDGET_EXCEEDED",

        message: "Insufficient remaining budget.",

        state: {
          budget,
        },
      });
    }

    // ======================================================
    // 9. FIND PAYMENT ACCOUNT
    // ======================================================

    const account = await Account.findById(accountId);

    if (!account) {
      return res.status(404).json({
        success: false,
        environmentError: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        errorType: "ACCOUNT_NOT_FOUND",

        message: "Payment account not found.",
      });
    }

    // ======================================================
    // 10. ACCOUNT FROZEN
    // ======================================================

    if (account.frozen) {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        errorType: "ACCOUNT_FROZEN",

        message: "Account is frozen.",
      });
    }

    // ======================================================
    // 11. ACCOUNT BALANCE
    // ======================================================

    if (Number(account.balance) < Number(invoice.amount)) {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.INSUFFICIENT_BALANCE,

        done: false,

        errorType: "INSUFFICIENT_BALANCE",

        message: "Insufficient account balance.",
      });
    }

    // ======================================================
    // 12. EXECUTE PAYMENT
    // ======================================================

    account.balance -= invoice.amount;

    budget.remainingBudget -= invoice.amount;

    invoice.status = "PAID";

    invoice.paymentAttempts += 1;

    // ======================================================
    // 13. SAVE BUSINESS STATE
    // ======================================================

    await account.save();

    await budget.save();

    await invoice.save();

    // ======================================================
    // 14. CREATE TRANSACTION
    // ======================================================

    const transaction = await Transaction.create({
      transactionId: `TX-${Date.now()}`,

      invoice: invoice._id,

      account: account._id,

      amount: invoice.amount,

      paymentMethod: invoice.paymentMethod || "BANK",

      status: "SUCCESS",
    });

    // ======================================================
    // 15. SUCCESS
    // ======================================================

    return res.status(200).json({
      success: true,
      environmentError: false,

      reward: REWARDS.PAYMENT_SUCCESS,

      done: false,

      message: "Invoice paid successfully.",

      state: {
        invoice,
        transaction,
        account,
        budget,
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

// ==========================================================
// REFUND PAYMENT
// ==========================================================

export async function refund(req, res) {
  try {
    const { transactionId } = req.body;

    if (!transactionId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "transactionId is required.",
      });
    }

    const transaction =
      await Transaction.findById(transactionId).populate("invoice");

    if (!transaction) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_ACTION,
        done: false,
        errorType: "TRANSACTION_NOT_FOUND",
        message: "Transaction not found.",
      });
    }

    if (transaction.status !== "SUCCESS") {
      return res.status(409).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_WORKFLOW,
        done: false,
        errorType: "INVALID_WORKFLOW",
        message: "Only successful payments can be refunded.",
      });
    }

    const account = await Account.findById(transaction.account);

    if (!account) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "ACCOUNT_NOT_FOUND",
        message: "Account not found.",
      });
    }

    account.balance += transaction.amount;

    await account.save();

    transaction.status = "REVERSED";

    await transaction.save();

    await Invoice.findByIdAndUpdate(transaction.invoice, {
      status: "APPROVED",
    });

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.SUCCESS,
      done: false,

      message: "Payment refunded.",

      state: {
        transactionStatus: transaction.status,

        accountBalance: account.balance,
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

// ==========================================================
// CANCEL PAYMENT
// ==========================================================

export async function cancelPayment(req, res) {
  try {
    const { transactionId } = req.body;

    if (!transactionId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "transactionId is required.",
      });
    }

    const transaction = await Transaction.findById(transactionId);

    if (!transaction) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_ACTION,
        done: false,
        errorType: "TRANSACTION_NOT_FOUND",
        message: "Transaction not found.",
      });
    }

    if (transaction.status !== "PENDING") {
      return res.status(409).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_WORKFLOW,
        done: false,
        errorType: "INVALID_WORKFLOW",
        message: "Only pending payments can be cancelled.",
      });
    }

    transaction.status = "FAILED";

    transaction.failureReason = "Cancelled by agent";

    await transaction.save();

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.SUCCESS,
      done: false,

      message: "Payment cancelled.",

      state: {
        transactionStatus: transaction.status,
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

// ==========================================================
// RETRY PAYMENT
// ==========================================================

export async function retryPayment(req, res) {
  try {
    const { invoiceId } = req.body;

    if (!invoiceId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "invoiceId is required.",
      });
    }

    const invoice = await Invoice.findById(invoiceId);

    if (!invoice) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVOICE_NOT_FOUND,
        done: false,
        errorType: "INVOICE_NOT_FOUND",
        message: "Invoice not found.",
      });
    }

    if (invoice.status === "PAID") {
      return res.status(409).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_ACTION,
        done: false,
        errorType: "INVOICE_ALREADY_PAID",
        message: "Invoice already paid.",
      });
    }

    invoice.paymentAttempts += 1;

    await invoice.save();

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.SUCCESS,
      done: false,

      message: "Payment retry initiated.",

      invoice,
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
