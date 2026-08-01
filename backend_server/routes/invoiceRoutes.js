import { Router } from "express";
import {
  createInvoice,
  getInvoices,
  getInvoice,
  updateStatus,
  checkDuplicate,
  archiveInvoice,
} from "../controllers/invoiceController.js";
import { authorizePermission } from "../middleware/authMiddleware.js";
import { actionLogger } from "../middleware/actionLogger.js";

const router = Router();

router.post(
  "/",
  authorizePermission("CREATE_INVOICE"),
  actionLogger("CREATE_INVOICE"),
  createInvoice,
);

router.get(
  "/",
  authorizePermission("READ_INVOICE"),
  actionLogger("READ_INVOICE"),
  getInvoices,
);

router.get(
  "/:id",
  authorizePermission("READ_INVOICE"),
  actionLogger("VIEW_INVOICE"),
  getInvoice,
);

router.patch(
  "/:id/status",
  authorizePermission("UPDATE_INVOICE"),
  actionLogger("UPDATE_INVOICE"),
  updateStatus,
);

router.post(
  "/duplicate-check",
  authorizePermission("READ_INVOICE"),
  actionLogger("DUPLICATE_CHECK"),
  checkDuplicate,
);

router.post(
  "/archive",
  authorizePermission("UPDATE_INVOICE"),
  actionLogger("ARCHIVE_INVOICE"),
  archiveInvoice,
);

export default router;
