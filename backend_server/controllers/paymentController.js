import Invoice from "../models/Invoice.js";
import Transaction from "../models/Transaction.js";
import Account from "../models/Account.js";
import Budget from "../models/Budget.js"

import { REWARDS } from "../utils/rewards.js";

// PAY INVOICE
export async function payInvoice(req, res) {
  try {
    const { invoiceId, accountId } = req.body;

    // =====================================================
    // 1. FIND INVOICE
    // =====================================================

    const invoice = await Invoice.findById(invoiceId).populate("supplier");

    if (!invoice) {
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        done: false,

        errorType: "INVOICE_NOT_FOUND",

        message: "Invoice not found.",
      });
    }

    // =====================================================
    // 2. INVOICE MUST BE APPROVED
    // =====================================================

    if (invoice.status !== "APPROVED") {
      return res.status(409).json({
        success: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        errorType: "INVOICE_NOT_APPROVED",

        message: "Invoice is not approved.",
      });
    }

    // =====================================================
    // 3. DUPLICATE INVOICE CHECK
    // =====================================================

    if (invoice.duplicateFlag) {
      return res.status(409).json({
        success: false,

        reward: REWARDS.DUPLICATE_INVOICE,

        done: false,

        errorType: "DUPLICATE_INVOICE",

        message: "Duplicate invoice cannot be paid.",
      });
    }

    // =====================================================
    // 4. SUPPLIER VALIDATION
    // =====================================================

    if (!invoice.supplier || !invoice.supplier.active) {
      return res.status(409).json({
        success: false,

        reward: REWARDS.SUPPLIER_INACTIVE,

        done: false,

        errorType: "SUPPLIER_INACTIVE",

        message: "Supplier inactive.",
      });
    }

    // =====================================================
    // 5. BUDGET CHECK
    //
    // Invoice category maps directly to
    // Budget department.
    //
    // Example:
    //
    // invoice.category = "SOFTWARE"
    //
    //        ↓
    //
    // budget.department = "SOFTWARE"
    // =====================================================

    const budget = await Budget.findOne({
      department: invoice.category,
    });

    // No matching budget

    if (!budget) {
      return res.status(409).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        errorType: "BUDGET_NOT_FOUND",

        message: `No budget found for department ${invoice.category}.`,
      });
    }

    // =====================================================
    // 6. CHECK REMAINING BUDGET
    // =====================================================

    if (budget.remainingBudget < invoice.amount) {
      return res.status(409).json({
        success: false,

        reward: REWARDS.BUDGET_EXCEEDED,

        done: false,

        errorType: "BUDGET_EXCEEDED",

        message: "Insufficient remaining budget.",

        state: {
          budget,
        },
      });
    }

    // =====================================================
    // 7. FIND PAYMENT ACCOUNT
    // =====================================================

    const account = await Account.findById(accountId);

    if (!account) {
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        errorType: "ACCOUNT_NOT_FOUND",

        message: "Payment account not found.",
      });
    }

    // =====================================================
    // 8. ACCOUNT FROZEN CHECK
    // =====================================================

    if (account.frozen) {
      return res.status(409).json({
        success: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        errorType: "ACCOUNT_FROZEN",

        message: "Account is frozen.",
      });
    }

    // =====================================================
    // 9. ACCOUNT BALANCE CHECK
    // =====================================================

    if (account.balance < invoice.amount) {
      return res.status(409).json({
        success: false,

        reward: REWARDS.INSUFFICIENT_BALANCE,

        done: false,

        errorType: "INSUFFICIENT_BALANCE",

        message: "Insufficient account balance.",
      });
    }

    // =====================================================
    // 10. EXECUTE PAYMENT
    // =====================================================

    account.balance -= invoice.amount;

    // Deduct the invoice amount from
    // the category's remaining budget.

    budget.remainingBudget -= invoice.amount;

    invoice.status = "PAID";

    invoice.paymentAttempts += 1;

    // =====================================================
    // 11. SAVE UPDATED DATA
    // =====================================================

    await account.save();

    await budget.save();

    await invoice.save();

    // =====================================================
    // 12. CREATE TRANSACTION
    // =====================================================

    const transaction = await Transaction.create({
      transactionId: `TX-${Date.now()}`,

      invoice: invoice._id,

      account: account._id,

      amount: invoice.amount,

      paymentMethod: invoice.paymentMethod || "BANK",

      status: "SUCCESS",
    });

    // =====================================================
    // 13. SUCCESS RESPONSE
    // =====================================================

    return res.status(200).json({
      success: true,

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
    // Environment/system error.
    // Do NOT give the RL agent a negative reward.

    return res.status(500).json({
      success: false,

      environmentError: true,

      retryable: true,

      reward: null,

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
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        message: "Transaction not found.",
      });
    }

    if (transaction.status !== "SUCCESS") {
      return res.status(409).json({
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
      transaction.invoice,

      {
        status: "APPROVED",
      },
    );

    return res.status(200).json({
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
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        message: "Transaction not found.",
      });
    }

    if (transaction.status !== "PENDING") {
      return res.status(409).json({
        success: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        message: "Only pending payments can be cancelled.",
      });
    }

    transaction.status = "FAILED";

    transaction.failureReason = "Cancelled by agent";

    await transaction.save();

    return res.status(200).json({
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
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        done: false,

        message: "Invoice not found.",
      });
    }

    if (invoice.status === "PAID") {
      return res.status(409).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        done: false,

        message: "Invoice already paid.",
      });
    }

    invoice.paymentAttempts += 1;

    await invoice.save();

    return res.status(200).json({
      success: true,

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

      message: error.message,
    });
  }
}
