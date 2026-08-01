import Invoice from "../models/Invoice.js";
import Transaction from "../models/Transaction.js";
import Account from "../models/Account.js";

import { REWARDS } from "../utils/rewards.js";

// PAY INVOICE
export async function payInvoice(req, res) {
  try {
    const { invoiceId, accountId } = req.body;

    const invoice = await Invoice.findById(invoiceId).populate("supplier");

    if (!invoice) {
      return res.json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        done: false,

        errorType: "INVOICE_NOT_FOUND",

        message: "Invoice not found.",
      });
    }

    // Business rule:
    // Invoice must be approved before payment

    if (invoice.status !== "APPROVED") {
      return res.json({
        success: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        errorType: "INVOICE_NOT_APPROVED",

        message: "Invoice is not approved.",
      });
    }

    // Duplicate invoice protection

    if (invoice.duplicateFlag) {
      return res.json({
        success: false,

        reward: REWARDS.DUPLICATE_INVOICE,

        done: false,

        errorType: "DUPLICATE_INVOICE",

        message: "Duplicate invoice cannot be paid.",
      });
    }

    // Supplier validation

    if (!invoice.supplier || !invoice.supplier.active) {
      return res.json({
        success: false,

        reward: REWARDS.SUPPLIER_INACTIVE,

        done: false,

        errorType: "SUPPLIER_INACTIVE",

        message: "Supplier inactive.",
      });
    }

    const account = await Account.findById(accountId);

    if (!account) {
      return res.json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        errorType: "ACCOUNT_NOT_FOUND",

        message: "Payment account not found.",
      });
    }

    if (account.frozen) {
      return res.json({
        success: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        errorType: "ACCOUNT_FROZEN",

        message: "Account is frozen.",
      });
    }

    if (account.balance < invoice.amount) {
      return res.json({
        success: false,

        reward: REWARDS.INSUFFICIENT_BALANCE,

        done: false,

        errorType: "INSUFFICIENT_BALANCE",

        message: "Insufficient account balance.",
      });
    }

    // Execute payment

    account.balance -= invoice.amount;

    await account.save();

    invoice.status = "PAID";

    invoice.paymentAttempts += 1;

    await invoice.save();

    const transaction = await Transaction.create({
      referenceId: `TX-${Date.now()}`,

      invoice: invoice._id,

      account: account._id,

      amount: invoice.amount,

      paymentMethod: invoice.paymentMethod || "BANK",

      status: "SUCCESS",
    });

    return res.json({
      success: true,

      reward: REWARDS.PAYMENT_SUCCESS,

      done: false,

      message: "Invoice paid successfully.",

      state: {
        invoiceStatus: invoice.status,

        transactionId: transaction._id,

        accountBalance: account.balance,
      },
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      retryable: true,

      message: error.message,
    });
  }
}

// REFUND PAYMENT

export async function refund(req, res) {
  try {
    const { transactionId } = req.body;

    const transaction =
      await Transaction.findById(transactionId).populate("invoice");

    if (!transaction) {
      return res.json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        message: "Transaction not found.",
      });
    }

    if (transaction.status !== "SUCCESS") {
      return res.json({
        success: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        message: "Only successful payments can be refunded.",
      });
    }

    const account = await Account.findById(transaction.account);

    account.balance += transaction.amount;

    await account.save();

    transaction.status = "REVERSED";

    await transaction.save();

    await Invoice.findByIdAndUpdate(
      transaction.invoice._id,

      {
        status: "APPROVED",
      },
    );

    return res.json({
      success: true,

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

      message: error.message,
    });
  }
}

// CANCEL PAYMENT

export async function cancelPayment(req, res) {
  try {
    const { transactionId } = req.body;

    const transaction = await Transaction.findById(transactionId);

    if (!transaction) {
      return res.json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        message: "Transaction not found.",
      });
    }

    if (transaction.status !== "PENDING") {
      return res.json({
        success: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        message: "Only pending payments can be cancelled.",
      });
    }

    transaction.status = "FAILED";

    transaction.failureReason = "Cancelled by agent";

    await transaction.save();

    return res.json({
      success: true,

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

      message: error.message,
    });
  }
}

// RETRY PAYMENT

export async function retryPayment(req, res) {
  try {
    const { invoiceId } = req.body;

    const invoice = await Invoice.findById(invoiceId);

    if (!invoice) {
      return res.json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        done: false,

        message: "Invoice not found.",
      });
    }

    if (invoice.status === "PAID") {
      return res.json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        message: "Invoice already paid.",
      });
    }

    invoice.paymentAttempts += 1;

    await invoice.save();

    return res.json({
      success: true,

      reward: REWARDS.SUCCESS,

      done: false,

      message: "Payment retry initiated.",

      state: {
        paymentAttempts: invoice.paymentAttempts,
      },
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      retryable: true,

      message: error.message,
    });
  }
}
