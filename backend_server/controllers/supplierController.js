import Supplier from "../models/Supplier.js";

import { REWARDS } from "../utils/rewards.js";

// ==========================================================
// Configuration
// ==========================================================

const HIGH_RISK_THRESHOLD = 70;

// ==========================================================
// GET ALL SUPPLIERS
// ==========================================================

export async function getSuppliers(req, res) {
  try {
    const allowedParams = [];

    const receivedParams = Object.keys(req.query);

    const invalidParams = receivedParams.filter(
      (param) => !allowedParams.includes(param),
    );

    if (invalidParams.length > 0) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: `Invalid query parameters: ${invalidParams.join(", ")}`,
      });
    }

    const suppliers = await Supplier.find();

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.NONE,
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

// ==========================================================
// VALIDATE SUPPLIER
// ==========================================================

export async function validateSupplier(req, res) {
  try {
    const { supplierId } = req.body;

    // ======================================================
    // Invalid API Request
    // ======================================================

    if (!supplierId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "supplierId is required.",
      });
    }

    // ======================================================
    // Find Supplier
    // ======================================================

    const supplier = await Supplier.findById(supplierId);

    // ======================================================
    // Supplier Missing
    //
    // The validation operation itself completed successfully.
    //
    // The supplier is simply not eligible.
    // ======================================================

    if (!supplier) {
      return res.status(200).json({
        success: true,
        environmentError: false,
        reward: REWARDS.NONE,

        valid: false,
        eligible: false,

        reason: "SUPPLIER_NOT_FOUND",

        supplier: null,

        message: "Supplier does not exist.",
      });
    }

    // ======================================================
    // Supplier Inactive
    // ======================================================

    if (!supplier.active) {
      return res.status(200).json({
        success: true,
        environmentError: false,
        reward: REWARDS.NONE,

        valid: false,
        eligible: false,

        reason: "SUPPLIER_INACTIVE",

        supplier,

        validation: {
          active: false,
          riskScore: supplier.riskScore,
          rating: supplier.rating,
        },

        message: "Supplier is inactive and cannot be used for payment.",
      });
    }

    // ======================================================
    // High Risk Supplier
    // ======================================================

    if (supplier.riskScore > HIGH_RISK_THRESHOLD) {
      return res.status(200).json({
        success: true,
        environmentError: false,
        reward: REWARDS.NONE,

        valid: false,
        eligible: false,

        // Standardized name.
        reason: "SUPPLIER_HIGH_RISK",

        supplier,

        validation: {
          active: true,
          riskScore: supplier.riskScore,
          rating: supplier.rating,
        },

        message: "Supplier is high risk and is not eligible for payment.",
      });
    }

    // ======================================================
    // Valid Supplier
    // ======================================================

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.SUPPLIER_VALIDATED,

      valid: true,
      eligible: true,

      reason: null,

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
      reward: null,
      message: error.message,
    });
  }
}

// ==========================================================
// BLACKLIST SUPPLIER
// ==========================================================

export async function blacklistSupplier(req, res) {
  try {
    const { supplierId, reason } = req.body;

    if (!supplierId) {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "supplierId is required.",
      });
    }

    if (typeof supplierId !== "string") {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "Invalid supplierId format.",
      });
    }

    if (reason !== undefined && typeof reason !== "string") {
      return res.status(400).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "INVALID_REQUEST",
        message: "Reason must be a string.",
      });
    }

    const supplier = await Supplier.findById(supplierId);

    if (!supplier) {
      return res.status(404).json({
        success: false,
        environmentError: false,
        reward: null,
        errorType: "SUPPLIER_NOT_FOUND",
        message: "Supplier not found.",
      });
    }

    supplier.active = false;

    await supplier.save();

    return res.status(200).json({
      success: true,
      environmentError: false,
      reward: REWARDS.SUCCESS,

      supplierId: supplier._id,
      supplierName: supplier.supplierName,

      message: "Supplier blacklisted successfully.",

      reason: reason || "No reason provided.",
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
