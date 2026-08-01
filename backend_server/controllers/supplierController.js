import Supplier from "../models/Supplier.js";
import { REWARDS } from "../utils/rewards.js";

// Get all suppliers

export async function getSuppliers(req, res) {
  try {
    const suppliers = await Supplier.find();

    return res.json({
      success: true,

      reward: REWARDS.SUCCESS,

      count: suppliers.length,

      suppliers,
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

// Validate supplier before payment

export async function validateSupplier(req, res) {
  try {
    const { supplierId } = req.body;

    const supplier = await Supplier.findById(supplierId);

    if (!supplier) {
      return res.json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        errorType: "SUPPLIER_NOT_FOUND",

        message: "Supplier does not exist.",
      });
    }

    // Supplier inactive

    if (!supplier.active) {
      return res.json({
        success: false,

        reward: REWARDS.SUPPLIER_INACTIVE,

        errorType: "SUPPLIER_INACTIVE",

        supplier,

        message: "Supplier is blacklisted or inactive.",
      });
    }

    // High risk supplier

    if (supplier.riskScore > 70) {
      return res.json({
        success: false,

        reward: -15,

        errorType: "HIGH_RISK_SUPPLIER",

        supplier,

        message: "Supplier requires additional review.",
      });
    }

    return res.json({
      success: true,

      reward: REWARDS.SUPPLIER_VALIDATED,

      supplier,

      validation: {
        active: true,

        riskScore: supplier.riskScore,

        rating: supplier.rating,
      },

      message: "Supplier validated successfully.",
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

// Blacklist supplier

export async function blacklistSupplier(req, res) {
  try {
    const { supplierId, reason } = req.body;

    const supplier = await Supplier.findById(supplierId);

    if (!supplier) {
      return res.json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        errorType: "SUPPLIER_NOT_FOUND",

        message: "Supplier not found.",
      });
    }

    supplier.active = false;

    await supplier.save();

    return res.json({
      success: true,

      reward: 10,

      supplierId: supplier._id,

      supplierName: supplier.supplierName,

      message: "Supplier blacklisted successfully.",

      reason: reason || "No reason provided",
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
