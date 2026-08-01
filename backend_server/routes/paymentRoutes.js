import { Router } from "express";
import {
  payInvoice,
  refund,
  cancelPayment,
  retryPayment,
} from "../controllers/paymentController.js";
import { authorizePermission } from "../middleware/authMiddleware.js";
import { actionLogger } from "../middleware/actionLogger.js";

const router = Router();

router.post(
  "/pay",
  authorizePermission("PAY_INVOICE"),
  actionLogger("PAY_INVOICE"),
  payInvoice,
);

router.post(
  "/refund",
  authorizePermission("REFUND_PAYMENT"),
  actionLogger("REFUND_PAYMENT"),
  refund,
);

router.post(
  "/cancel-payment",
  authorizePermission("PAY_INVOICE"),
  actionLogger("CANCEL_PAYMENT"),
  cancelPayment,
);

router.post(
  "/retry-payment",
  authorizePermission("PAY_INVOICE"),
  actionLogger("RETRY_PAYMENT"),
  retryPayment,
);

export default router;
