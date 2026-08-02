import Supplier from "../models/Supplier.js";
import { REWARDS } from "../utils/rewards.js";

// Get all suppliers

export async function getSuppliers(req, res) {
  try {
    // Optional: Reject unsupported query parameters
    const allowedParams = [];

    const receivedParams = Object.keys(req.query);

    const invalidParams = receivedParams.filter(
      (param) => !allowedParams.includes(param),
    );

    if (invalidParams.length > 0) {
      return res.status(400).json({
        success: false,

        reward: null,

        environmentError: false,

        message: `Invalid query parameters: ${invalidParams.join(", ")}`,
      });
    }

    const suppliers = await Supplier.find();

    return res.status(200).json({
      success: true,

      environmentError: false,

      reward: REWARDS.SUCCESS,

      count: suppliers.length,

      suppliers,
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

// Validate supplier before payment

export async function validateSupplier(req, res) {
  try {
    const { supplierId } = req.body;

    // Bad request
    if (!supplierId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "supplierId is required.",
      });
    }

    const supplier = await Supplier.findById(supplierId);

    // Supplier doesn't exist
    if (!supplier) {
      return res.status(404).json({
        success: false,
        reward: REWARDS.SUPPLIER_NOT_FOUND, // create this reward instead of INVOICE_NOT_FOUND
        errorType: "SUPPLIER_NOT_FOUND",
        message: "Supplier does not exist.",
      });
    }

    // Business rule: inactive supplier
    if (!supplier.active) {
      return res.status(200).json({
        success: false,
        reward: REWARDS.SUPPLIER_INACTIVE,
        errorType: "SUPPLIER_INACTIVE",
        supplier,
        message: "Supplier is blacklisted or inactive.",
      });
    }

    // Business rule: high-risk supplier
    if (supplier.riskScore > 70) {
      return res.status(200).json({
        success: false,
        reward: REWARDS.SUPPLIER_HIGH_RISK,
        errorType: "HIGH_RISK_SUPPLIER",
        supplier,
        message: "Supplier requires additional review [High Risk Supplier]",
      });
    }

    // Success
    return res.status(200).json({
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

    // Validate required parameter
    if (!supplierId) {
      return res.status(400).json({
        success: false,

        reward: null,

        environmentError: false,

        message: "supplierId is required.",
      });
    }

    // Validate parameter types
    if (typeof supplierId !== "string") {
      return res.status(400).json({
        success: false,

        reward: null,

        environmentError: false,

        message: "Invalid supplierId format.",
      });
    }

    if (reason !== undefined && typeof reason !== "string") {
      return res.status(400).json({
        success: false,

        reward: null,

        environmentError: false,

        message: "Reason must be a string.",
      });
    }

    const supplier = await Supplier.findById(supplierId);

    if (!supplier) {
      return res.status(404).json({
        success: false,

        reward: REWARDS.INVOICE_NOT_FOUND,

        environmentError: false,

        errorType: "SUPPLIER_NOT_FOUND",

        message: "Supplier not found.",
      });
    }

    supplier.active = false;

    await supplier.save();

    return res.status(200).json({
      success: true,

      environmentError: false,

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

      reward: null,

      message: error.message,
    });
  }
}
