import { User, Invoice, Transaction, Account } from "../models/modelsIndex.js";

// @desc    Reset dynamic data to baseline state for RL agent training
// @route   POST /api/sandbox/reset
export const resetSandbox = async (req, res) => {
  try {
    // 1. Clear dynamic collections
    await Invoice.deleteMany({});
    await Transaction.deleteMany({});
    await Account.deleteMany({});

    // 2. Seed initial Account balance
    const account = await Account.create({
      accountNumber: "ACC-CORP-001",
      accountName: "Main Corporate Operating Account",
      balance: 1000000,
      currency: "GBP",
    });

    // 3. Seed initial Invoices for RL Agent to act upon
    const seededInvoices = await Invoice.insertMany([
      {
        invoiceNumber: "INV-1001",
        vendorName: "Acme Cloud Services",
        amount: 4500,
        description: "Monthly Cloud Infrastructure Bill",
        dueDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days from now
        status: "PENDING_APPROVAL",
      },
      {
        invoiceNumber: "INV-1002",
        vendorName: "Global Logistics Ltd",
        amount: 12000,
        description: "Monthly Storage Infrastructure Bill",
        dueDate: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000),
        status: "PENDING_APPROVAL",
      },
      {
        invoiceNumber: "INV-1003",
        vendorName: "Office Supplies Co",
        amount: 850,
        description: "Monthly Office Management Bill",
        dueDate: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000),
        status: "APPROVED",
      },
    ]);

    res.status(200).json({
      message: "Sandbox environment successfully reset.",
      account,
      invoicesCount: seededInvoices.length,
    });
  } catch (error) {
    res.status(500).json({ error: "Reset failed: " + error.message });
  }
};

// @desc    Get current system metrics/observation state for RL agent
// @route   GET /api/sandbox/state
export const getSandboxState = async (req, res) => {
  try {
    const account = await Account.findOne({ accountNumber: "ACC-CORP-001" });
    const pendingInvoices = await Invoice.countDocuments({
      status: "PENDING_APPROVAL",
    });
    const approvedInvoices = await Invoice.countDocuments({
      status: "APPROVED",
    });
    const paidInvoices = await Invoice.countDocuments({ status: "PAID" });
    const totalTransactions = await Transaction.countDocuments({});
    const unreconciledTx = await Transaction.countDocuments({
      reconciled: false,
    });

    res.status(200).json({
      account: account,
      metrics: {
        pendingInvoices,
        approvedInvoices,
        paidInvoices,
        totalTransactions,
        unreconciledTx,
      },
    });
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch state: " + error.message });
  }
};
