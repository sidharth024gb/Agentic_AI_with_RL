import express from "express";
import { authorizePermission } from "../middleware/authMiddleware.js";
import {
  startEpisode,
  addStep,
  endEpisode,
  getEpisode,
  getEpisodes,
} from "../controllers/episodeController.js";
import { actionLogger } from "../middleware/actionLogger.js";

const router = express.Router();

router.post(
  "/start",
  authorizePermission("RESET_ENVIRONMENT"),
  actionLogger("START_EPISODE"),
  startEpisode,
);

router.post(
  "/:id/step",
  authorizePermission("VIEW_STATE"),
  actionLogger("STEP_EPISODE"),
  addStep,
);

router.post(
  "/:id/end",
  authorizePermission("VIEW_STATE"),
  actionLogger("END_EPISODE"),
  endEpisode,
);

router.get("/", authorizePermission("VIEW_STATE"), getEpisodes);

router.get("/:id", authorizePermission("VIEW_STATE"), getEpisode);

export default router;
