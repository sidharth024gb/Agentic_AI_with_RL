import { Invoice, Transaction, Account } from "../models/index.js";

// @desc    Get all invoices
// @route   GET /api/finance/invoices
export const getInvoices = async (req, res) => {
  try {
    const invoices = await Invoice.find({});
    res.status(200).json(invoices);
  } catch (error) {
    res
      .status(500)
      .json({ error: "Failed to fetch invoices: " + error.message });
  }
};

// @desc    Approve or Reject an invoice
// @route   PATCH /api/finance/invoices/:id/status
export const updateInvoiceStatus = async (req, res) => {
  try {
    const { status } = req.body; // e.g., 'APPROVED', 'REJECTED'
    const invoice = await Invoice.findById(req.params.id);

    if (!invoice) {
      return res.status(404).json({ error: "Invoice not found" });
    }

    invoice.status = status;
    await invoice.save();

    res
      .status(200)
      .json({ message: `Invoice status updated to ${status}`, invoice });
  } catch (error) {
    res
      .status(500)
      .json({ error: "Failed to update invoice: " + error.message });
  }
};

// @desc    Pay an approved invoice and deduct from treasury account
// @route   POST /api/finance/pay
export const executePayment = async (req, res) => {
  try {
    const { invoiceId, accountNumber } = req.body;

    const invoice = await Invoice.findById(invoiceId);
    if (!invoice) {
      return res.status(404).json({ error: "Invoice not found" });
    }

    if (invoice.status !== "APPROVED") {
      return res
        .status(400)
        .json({ error: "Invoice must be APPROVED before payment" });
    }

    const account = await Account.findOne({
      accountNumber: accountNumber || "ACC-CORP-001",
    });
    if (!account) {
      return res
        .status(404)
        .json({ error: "Target treasury account not found" });
    }

    if (account.balance < invoice.amount) {
      return res
        .status(400)
        .json({ error: "Insufficient funds in treasury account" });
    }

    // Deduct balance
    account.balance -= invoice.amount;
    account.updatedAt = Date.now();
    await account.save();

    // Mark invoice as paid
    invoice.status = "PAID";
    await invoice.save();

    // Log ledger transaction
    const transaction = await Transaction.create({
      referenceId: `TX-${Date.now()}`,
      invoiceId: invoice._id,
      accountId: account._id,
      type: "PAYMENT_OUT",
      amount: invoice.amount,
      reconciled: false,
    });

    res.status(200).json({
      message: "Payment executed successfully",
      newBalance: account.balance,
      transaction,
    });
  } catch (error) {
    res
      .status(500)
      .json({ error: "Payment execution failed: " + error.message });
  }
};

// @desc    Reconcile pending transactions
// @route   POST /api/finance/reconcile
export const reconcileTransaction = async (req, res) => {
  try {
    const { transactionId } = req.body;

    const transaction = await Transaction.findById(transactionId);
    if (!transaction) {
      return res.status(404).json({ error: "Transaction not found" });
    }

    transaction.reconciled = true;
    await transaction.save();

    res
      .status(200)
      .json({ message: "Transaction reconciled successfully", transaction });
  } catch (error) {
    res.status(500).json({ error: "Reconciliation failed: " + error.message });
  }
};
