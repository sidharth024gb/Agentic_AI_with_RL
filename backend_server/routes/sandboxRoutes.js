import express from "express";
import { resetSandbox, getSandboxState } from "../controllers/sandboxController.js";

const router = express.Router();

router.post("/reset", resetSandbox);
router.get("/state", getSandboxState);

export default router;