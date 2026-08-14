import Invoice from "../models/Invoice.js";
import AuditLog from "../models/AuditLog.js";

import { REWARDS } from "../utils/rewards.js";

// Approve Invoice

export async function approveInvoice(req, res) {
  try {
    const { invoiceId } = req.body;

    // ======================================================
    // Validate Request
    // ======================================================

    if (!invoiceId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        done: false,
        errorType: "INVALID_REQUEST",
        message: "invoiceId is required.",
      });
    }

    // ======================================================
    // Find Invoice
    // ======================================================

    const invoice = await Invoice.findById(invoiceId);

    if (!invoice) {
      return res.status(404).json({
        success: false,
        environmentError: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        done: false,

        errorType: "INVOICE_NOT_FOUND",

        message: "Invoice does not exist.",
      });
    }

    // ======================================================
    // Workflow Validation
    // ======================================================

    if (invoice.status !== "PENDING_APPROVAL") {
      return res.status(409).json({
        success: false,
        environmentError: false,

        reward: REWARDS.INVALID_WORKFLOW,

        done: false,

        errorType: "INVALID_WORKFLOW",

        message: `Invoice cannot be approved from ${invoice.status}.`,
      });
    }

    // ======================================================
    // Approve
    // ======================================================

    invoice.status = "APPROVED";

    invoice.approvedBy = req.user._id;

    await invoice.save();

    // ======================================================
    // Audit
    // ======================================================

    await AuditLog.create({
      user: req.user._id,

      action: "APPROVE_INVOICE",

      entityType: "Invoice",

      entityId: invoice._id,

      success: true,

      reward: REWARDS.APPROVE_SUCCESS,

      message: "Invoice approved successfully.",
    });

    // ======================================================
    // Response
    // ======================================================

    return res.status(200).json({
      success: true,
      environmentError: false,

      reward: REWARDS.APPROVE_SUCCESS,

      done: false,

      state: {
        invoiceId: invoice._id,
        status: invoice.status,
      },

      message: "Invoice approved.",
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

// Reject Invoice

export async function rejectInvoice(req, res) {
  try {
    const { invoiceId, reason } = req.body;

    if (!invoiceId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        message: "invoiceId is required.",
      });
    }

    const invoice = await Invoice.findById(invoiceId);

    if (!invoice) {
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        errorType: "INVOICE_NOT_FOUND",

        message: "Invoice does not exist.",
      });
    }

    if (invoice.status !== "PENDING_APPROVAL") {
      return res.status(400).json({
        success: false,

        reward: REWARDS.INVALID_WORKFLOW,

        errorType: "INVALID_WORKFLOW",

        message: "Invoice cannot be rejected.",
      });
    }

    invoice.status = "REJECTED";

    invoice.rejectionReason = reason || "Rejected by approver";

    invoice.approvedBy = req.user._id;

    await invoice.save();

    await AuditLog.create({
      user: req.user._id,

      action: "REJECT_INVOICE",

      entityType: "Invoice",

      entityId: invoice._id,

      success: true,

      reward: REWARDS.SUCCESS,

      message: "Invoice rejected.",
    });

    return res.json({
      success: true,

      reward: REWARDS.SUCCESS,

      done: false,

      invoice,

      message: "Invoice rejected.",
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
