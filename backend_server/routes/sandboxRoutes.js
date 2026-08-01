import { Router } from "express";
import {
  resetEnvironment,
  getState,
  randomizeEnvironment,
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

router.post(
  "/randomize",
  authorizePermission("RESET_ENVIRONMENT"),
  actionLogger("RANDOMIZE_ENVIRONMENT"),
  randomizeEnvironment,
);

router.get(
  "/reward",
  authorizePermission("VIEW_STATE"),
  actionLogger("VIEW_REWARD"),
  getReward,
);

export default router;
