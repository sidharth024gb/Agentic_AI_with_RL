import express from "express";
import {
  getInvoices,
  updateInvoiceStatus,
  executePayment,
  reconcileTransaction,
} from "../controllers/financeController.js";
import { protect } from "../middleware/authMiddleware.js";

const router = express.Router();

// Apply protect middleware to all financial actions
router.get("/invoices", protect, getInvoices);
router.patch("/invoices/:id/status", protect, updateInvoiceStatus);
router.post("/pay", protect, executePayment);
router.post("/reconcile", protect, reconcileTransaction);

export default router;
