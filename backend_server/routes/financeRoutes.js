import express from "express";
import {
  createInvoice,
  getInvoices,
  updateInvoiceStatus,
  executePayment,
  reconcileTransaction,
  getTransactions,
  getAccounts,
} from "../controllers/financeController.js";
import { protect } from "../middleware/authMiddleware.js";

const router = express.Router();

// Apply protect middleware to all financial actions
router.post("/invoices", protect, createInvoice);
router.get("/invoices", protect, getInvoices);
router.patch("/invoices/:id/status", protect, updateInvoiceStatus);
router.post("/pay", protect, executePayment);
router.post("/reconcile", protect, reconcileTransaction);
router.get("/transactions", protect, getTransactions);
router.get("/accounts", protect, getAccounts);

export default router;
