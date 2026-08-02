import Invoice from "../models/Invoice.js";
import Supplier from "../models/Supplier.js";

import { REWARDS } from "../utils/rewards.js";

// Create Invoice

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

    if (!supplierId || !amount || !dueDate) {
      return res.status(400).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        message: "Supplier, amount and due date are required.",
      });
    }

    const supplier = await Supplier.findById({ _id: supplierId });

    if (!supplier) {
      return res.status(404).json({
        success: false,
        message: "Supplier not found",
        reward: REWARDS.SUPPLIER_INACTIVE,
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
    const invoices = await Invoice.find({})
      .populate("supplier", "supplierName riskScore active")
      .sort({
        createdAt: -1,
      });

    return res.status(200).json({
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
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        message: "Invoice not found.",
      });
    }

    return res.status(200).json({
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
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        message: "Invoice not found.",
      });
    }

    const allowedStatus = ["PENDING_APPROVAL", "APPROVED", "REJECTED", "PAID"];

    if (!allowedStatus.includes(status)) {
      return res.status(400).json({
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

    return res.status(200).json({
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
    const { supplierId, amount, dueDate } = req.body;

    if (!supplierId || !amount || !dueDate) {
      return res.status(400).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        message: "SupplierId and amount and dueDate are required.",
      });
    }

    const duplicate = await Invoice.findOne({
      supplier: supplierId,
  
      amount,

      status: {
        $ne: "REJECTED",
      },

      dueDate: {
        $eq: dueDate,
      },
    });

    if (duplicate) {
      return res.status(200).json({
        success: false,

        reward: REWARDS.DUPLICATE_INVOICE,

        duplicate: true,

        message: "Duplicate invoice detected.",

        invoiceId: duplicate._id,
      });
    }

    return res.status(200).json({
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

    if (!invoiceId) {
      return res.status(400).json({
        success: false,

        reward: REWARDS.INVALID_ACTION,

        message: "Invoice ID is required.",
      });
    }

    const invoice = await Invoice.findById(invoiceId);

    if (!invoice) {
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        message: "Invoice not found.",
      });
    }

    invoice.status = "REJECTED";

    await invoice.save();

    return res.status(200).json({
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
