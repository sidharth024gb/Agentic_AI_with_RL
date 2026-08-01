import { Router } from "express";
import {
  getSuppliers,
  validateSupplier,
  blacklistSupplier,
} from "../controllers/supplierController.js";
import { authorizePermission } from "../middleware/authMiddleware.js";
import { actionLogger } from "../middleware/actionLogger.js";

const router = Router();

router.get(
  "/",
  authorizePermission("VALIDATE_SUPPLIER"),
  actionLogger("VIEW_SUPPLIERS"),
  getSuppliers,
);

router.post(
  "/validate",
  authorizePermission("VALIDATE_SUPPLIER"),
  actionLogger("VALIDATE_SUPPLIER"),
  validateSupplier,
);

router.post(
  "/blacklist",
  authorizePermission("BLACKLIST_SUPPLIER"),
  actionLogger("BLACKLIST_SUPPLIER"),
  blacklistSupplier,
);

export default router;
