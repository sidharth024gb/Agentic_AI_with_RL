import { Router } from "express";
import {
  getAccounts,
  checkBudget,
  transferMoney,
  cashPosition,
} from "../controllers/accountController.js";
import { authorizePermission } from "../middleware/authMiddleware.js"
import { actionLogger } from "../middleware/actionLogger.js";

const router = Router();

router.get(
  "/",
  authorizePermission("READ_ACCOUNT"),
  actionLogger("VIEW_ACCOUNT"),
  getAccounts,
);

router.post(
  "/budget/check",
  authorizePermission("CHECK_BUDGET"),
  actionLogger("CHECK_BUDGET"),
  checkBudget,
);

router.post(
  "/transfer",
  authorizePermission("TRANSFER_MONEY"),
  actionLogger("TRANSFER_MONEY"),
  transferMoney,
);

router.get(
  "/cash-position",
  authorizePermission("READ_ACCOUNT"),
  actionLogger("CASH_POSITION"),
  cashPosition,
);

export default router;
