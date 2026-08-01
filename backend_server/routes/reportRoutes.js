import { Router } from "express";
import {
  getTransactions,
  generateReport,
  getAuditLog,
} from "../controllers/reportController.js";
import { authorizePermission } from "../middleware/authMiddleware.js";
import { actionLogger } from "../middleware/actionLogger.js";

const router = Router();

router.get(
  "/transactions",
  authorizePermission("GENERATE_REPORT"),
  actionLogger("VIEW_TRANSACTIONS"),
  getTransactions,
);

router.post(
  "/generate-report",
  authorizePermission("GENERATE_REPORT"),
  actionLogger("GENERATE_REPORT"),
  generateReport,
);

router.get(
  "/audit-log",
  authorizePermission("VIEW_AUDIT"),
  actionLogger("VIEW_AUDIT"),
  getAuditLog,
);

export default router;
