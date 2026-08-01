import Invoice from "../models/Invoice.js";
import Supplier from "../models/Supplier.js";

import { REWARDS } from "../utils/rewards.js";

// Create Invoice

export async function createInvoice(req, res) {
  try {
    const invoice = await Invoice.create({
      supplier: req.body.supplier,

      amount: req.body.amount,

      dueDate: req.body.dueDate,

      description: req.body.description,

      paymentMethod: req.body.paymentMethod,

      priority: req.body.priority,

      category: req.body.category,

      createdBy: req.user._id,
    });

    return res.status(201).json({
      success: true,

      reward: REWARDS.SUCCESS,

      message: "Invoice created successfully.",

      invoice,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      reward: null,

      message: error.message,
    });
  }
}

// Get all invoices

export async function getInvoices(req, res) {
  try {
    const invoices = await Invoice.find()
      .populate("supplier", "supplierName riskScore active")
      .sort({
        createdAt: -1,
      });

    return res.json({
      success: true,

      reward: REWARDS.INVOICE_FOUND,

      count: invoices.length,

      invoices,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      reward: null,

      message: error.message,
    });
  }
}

// Get single invoice

export async function getInvoice(req, res) {
  try {
    const invoice = await Invoice.findById(req.params.id)
      .populate("supplier")
      .populate("approvedBy", "username email");

    if (!invoice) {
      return res.json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        message: "Invoice not found.",
      });
    }

    return res.json({
      success: true,

      reward: REWARDS.INVOICE_FOUND,

      invoice,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      reward: null,

      message: error.message,
    });
  }
}

// Update invoice status

export async function updateStatus(req, res) {
  try {
    const { status } = req.body;

    const invoice = await Invoice.findById(req.params.id);

    if (!invoice) {
      return res.json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        message: "Invoice not found.",
      });
    }

    const allowedStatus = ["PENDING_APPROVAL", "APPROVED", "REJECTED", "PAID"];

    if (!allowedStatus.includes(status)) {
      return res.json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        message: "Invalid invoice status.",
      });
    }

    invoice.status = status;

    if (status === "APPROVED") {
      invoice.approvedBy = req.user._id;
    }

    await invoice.save();

    return res.json({
      success: true,

      reward: REWARDS.SUCCESS,

      message: `Invoice status changed to ${status}`,

      invoice,
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      reward: null,

      message: error.message,
    });
  }
}

// Check duplicate invoice

export async function checkDuplicate(req, res) {
  try {
    const { supplier, amount } = req.body;

    const duplicate = await Invoice.findOne({
      supplier,

      amount,

      status: {
        $ne: "REJECTED",
      },
    });

    if (duplicate) {
      return res.json({
        success: false,

        reward: REWARDS.DUPLICATE_INVOICE,

        duplicate: true,

        message: "Duplicate invoice detected.",

        invoiceId: duplicate._id,
      });
    }

    return res.json({
      success: true,

      reward: REWARDS.SUCCESS,

      duplicate: false,

      message: "No duplicate invoice found.",
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      reward: null,

      message: error.message,
    });
  }
}

// Archive invoice

export async function archiveInvoice(req, res) {
  try {
    const { invoiceId } = req.body;

    const invoice = await Invoice.findById(invoiceId);

    if (!invoice) {
      return res.json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        message: "Invoice not found.",
      });
    }

    invoice.status = "REJECTED";

    await invoice.save();

    return res.json({
      success: true,

      reward: REWARDS.SUCCESS,

      message: "Invoice archived successfully.",
    });
  } catch (error) {
    return res.status(500).json({
      success: false,

      environmentError: true,

      reward: null,

      message: error.message,
    });
  }
}
