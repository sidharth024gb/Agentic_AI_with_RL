import { Router } from "express";
import {
  approveInvoice,
  rejectInvoice,
} from "../controllers/approvalController.js";
import { authorizePermission } from "../middleware/authMiddleware.js";
import { actionLogger } from "../middleware/actionLogger.js";

const router = Router();

router.post(
  "/approve",
  authorizePermission("APPROVE_INVOICE"),
  actionLogger("APPROVE_INVOICE"),
  approveInvoice,
);

router.post(
  "/reject",
  authorizePermission("REJECT_INVOICE"),
  actionLogger("REJECT_INVOICE"),
  rejectInvoice,
);

export default router;
