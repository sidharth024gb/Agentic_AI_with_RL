import { Router } from "express";

import {
  approveInvoice,
  rejectInvoice,
} from "../controllers/approvalController.js";

import { authorizePermission } from "../middleware/authMiddleware.js";

import { actionLogger } from "../middleware/actionLogger.js";

const router = Router();

// ==========================================================
// APPROVE INVOICE
// ==========================================================

router.patch(
  "/approve",
  authorizePermission("APPROVE_INVOICE"),
  actionLogger("APPROVE_INVOICE"),
  approveInvoice,
);

// ==========================================================
// REJECT INVOICE
// ==========================================================

router.patch(
  "/reject",
  authorizePermission("REJECT_INVOICE"),
  actionLogger("REJECT_INVOICE"),
  rejectInvoice,
);

export default router;
