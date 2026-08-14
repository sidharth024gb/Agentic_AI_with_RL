import Invoice from "../models/Invoice.js";
import Supplier from "../models/Supplier.js";

import { REWARDS } from "../utils/rewards.js";

// ==========================================================
// CREATE INVOICE
// ==========================================================

export async function createInvoice(req, res) {
  try {
    const {
      supplierId,
      amount,
      dueDate,
      description,
      paymentMethod,
      priority,
      category,
    } = req.body;

    if (!supplierId || amount == null || !dueDate) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_ACTION,
        errorType: "INVALID_REQUEST",
        message: "Supplier, amount and due date are required.",
      });
    }

    const supplier = await Supplier.findById(supplierId);

    if (!supplier) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: REWARDS.SUPPLIER_NOT_FOUND,
        errorType: "SUPPLIER_NOT_FOUND",
        message: "Supplier not found.",
      });
    }

    const invoice = await Invoice.create({
      supplier: supplier._id,
      amount,
      dueDate,
      description,
      paymentMethod,
      priority,
      category,
      createdBy: req.user._id,
    });

    return res.status(201).json({
      success: true,
      environmentError: false,
      reward: REWARDS.SUCCESS,
      message: "Invoice created successfully.",
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

// ==========================================================
// GET ALL INVOICES
//
// IMPORTANT FOR REPRODUCIBILITY:
// Do not sort by createdAt. MongoDB timestamps differ between
// resets even when the same seeded scenario is regenerated.
// invoiceNumber is generated deterministically by reset().
// _id is included as a stable tie-breaker.
// ==========================================================

export async function getInvoices(req, res) {
  try {
    const invoices = await Invoice.find({})
      .populate("supplier", "supplierName riskScore active rating")
      .sort({
        invoiceNumber: 1,
        _id: 1,
      });

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.INVOICE_FOUND,
      count: invoices.length,
      invoices,
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
// GET SINGLE INVOICE
// ==========================================================

export async function getInvoice(req, res) {
  try {
    const invoice = await Invoice.findById(req.params.id)
      .populate("supplier")
      .populate("approvedBy", "username email");

    if (!invoice) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVOICE_NOT_FOUND,
        errorType: "INVOICE_NOT_FOUND",
        message: "Invoice not found.",
      });
    }

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.INVOICE_FOUND,
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

// ==========================================================
// UPDATE INVOICE STATUS
// ==========================================================

export async function updateStatus(req, res) {
  try {
    const { status } = req.body;

    const invoice = await Invoice.findById(req.params.id);

    if (!invoice) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVOICE_NOT_FOUND,
        errorType: "INVOICE_NOT_FOUND",
        message: "Invoice not found.",
      });
    }

    const allowedStatus = ["PENDING_APPROVAL", "APPROVED", "REJECTED", "PAID"];

    if (!allowedStatus.includes(status)) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: REWARDS.INVALID_ACTION,
        errorType: "INVALID_STATUS",
        message: "Invalid invoice status.",
      });
    }

    invoice.status = status;

    if (status === "APPROVED") {
      invoice.approvedBy = req.user._id;
    }

    await invoice.save();

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.SUCCESS,
      message: `Invoice status changed to ${status}.`,
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

// ==========================================================
// CHECK DUPLICATE INVOICE
// ==========================================================

export async function checkDuplicate(req, res) {
  try {
    const { invoiceId, supplierId, amount, dueDate } = req.body;

    if (!supplierId || amount == null || !dueDate) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "supplierId, amount and dueDate are required.",
      });
    }

    // ======================================================
    // Validate Date
    // ======================================================

    const parsedDueDate = new Date(dueDate);

    if (Number.isNaN(parsedDueDate.getTime())) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "Invalid dueDate.",
      });
    }

    // ======================================================
    // Load Current Invoice When ID Is Available
    //
    // This lets us:
    //
    // 1. respect an explicit duplicateFlag
    // 2. exclude the invoice from matching itself
    // ======================================================

    let currentInvoice = null;

    if (invoiceId) {
      currentInvoice = await Invoice.findById(invoiceId);

      if (!currentInvoice) {
        return res.status(404).json({
          success: false,
          environmentError: false,
          reward: null,
          errorType: "INVOICE_NOT_FOUND",
          message: "Invoice not found.",
        });
      }
    }

    // ======================================================
    // Find Matching Invoices
    // ======================================================

    const query = {
      supplier: supplierId,
      amount,
      status: {
        $ne: "REJECTED",
      },
      dueDate: parsedDueDate,
    };

    // CRITICAL:
    // Do not let an invoice detect itself as its duplicate.

    if (invoiceId) {
      query._id = {
        $ne: invoiceId,
      };
    }

    const matchingInvoices = await Invoice.find(query).select(
      "_id duplicateFlag invoiceNumber",
    );

    // ======================================================
    // Determine Duplicate
    // ======================================================

    let duplicate = false;

    // Explicit duplicate flag on the current invoice.
    if (currentInvoice?.duplicateFlag) {
      duplicate = true;
    }

    // Another matching invoice exists.
    if (invoiceId && matchingInvoices.length > 0) {
      duplicate = true;
    }

    // Backwards-compatible behaviour if invoiceId was not sent.
    //
    // Since the current invoice is part of the matching query,
    // more than one matching invoice means a genuine duplicate.
    if (!invoiceId) {
      if (matchingInvoices.length > 1) {
        duplicate = true;
      }

      if (matchingInvoices.some((invoice) => invoice.duplicateFlag === true)) {
        duplicate = true;
      }
    }

    // ======================================================
    // IMPORTANT RL SEMANTICS
    //
    // The duplicate-check ACTION succeeded whether the invoice
    // is duplicate or not.
    //
    // duplicate=true is a discovered business condition.
    // It is NOT an action failure.
    // ======================================================

    return res.status(200).json({
      success: true,
      environmentError: false,

      reward: REWARDS.NONE,

      duplicate,

      valid: !duplicate,

      reason: duplicate ? "DUPLICATE_INVOICE" : null,

      invoiceId: invoiceId || null,

      matchingInvoiceIds: matchingInvoices.map((invoice) => invoice._id),

      message: duplicate
        ? "Duplicate invoice detected."
        : "No duplicate invoice found.",
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
// ARCHIVE INVOICE
// ==========================================================

export async function archiveInvoice(req, res) {
  try {
    const { invoiceId } = req.body;

    if (!invoiceId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "Invoice ID is required.",
      });
    }

    const invoice = await Invoice.findById(invoiceId);

    if (!invoice) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVOICE_NOT_FOUND",
        message: "Invoice not found.",
      });
    }

    invoice.status = "REJECTED";

    await invoice.save();

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.SUCCESS,
      message: "Invoice archived successfully.",
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
