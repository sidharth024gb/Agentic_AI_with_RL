import { Router } from "express";
import {
  resetEnvironment,
  getState,
  getReward,
} from "../controllers/sandboxController.js";
import { authorizePermission } from "../middleware/authMiddleware.js";
import { actionLogger } from "../middleware/actionLogger.js";

const router = Router();

router.post(
  "/reset",
  authorizePermission("RESET_ENVIRONMENT"),
  actionLogger("RESET_ENVIRONMENT"),
  resetEnvironment,
);

router.get(
  "/state",
  authorizePermission("VIEW_STATE"),
  actionLogger("VIEW_STATE"),
  getState,
);

router.get(
  "/reward/:episodeId",
  authorizePermission("VIEW_STATE"),
  actionLogger("VIEW_REWARD"),
  getReward,
);

export default router;
