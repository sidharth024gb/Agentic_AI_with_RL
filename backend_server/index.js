import express from "express";
import dotenv from "dotenv";
import cors from "cors";

import connectDB from "./config/db.js";

// Route Imports
import authRoutes from "./routes/authRoutes.js";
import invoiceRoutes from "./routes/invoiceRoutes.js";
import approvalRoutes from "./routes/approvalRoutes.js";
import paymentRoutes from "./routes/paymentRoutes.js";
import accountRoutes from "./routes/accountRoutes.js";
import supplierRoutes from "./routes/supplierRoutes.js";
import reportRoutes from "./routes/reportRoutes.js";
import sandboxRoutes from "./routes/sandboxRoutes.js";
import episodeRoutes from "./routes/episodeRoutes.js";
import { protect } from "./middleware/authMiddleware.js";

dotenv.config();

const app = express();

// Connect Database
connectDB();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health Check
app.get("/", (req, res) => {
  res.status(200).json({
    success: true,
    message: "Finance Backend Sandbox API Running",
    environment: process.env.NODE_ENV || "development",
  });
});

// ====================
// API Routes
// ====================

// Authentication
app.use("/api/auth", authRoutes);

// Invoice Management
app.use("/api/invoice", protect, invoiceRoutes);

// Approval Workflow
app.use("/api/approval", protect, approvalRoutes);

// Payments
app.use("/api/payment", protect, paymentRoutes);

// Treasury / Accounts
app.use("/api/account", protect, accountRoutes);

// Suppliers
app.use("/api/supplier", protect, supplierRoutes);

// Reporting
app.use("/api/report", protect, reportRoutes);

// RL Sandbox
app.use("/api/sandbox", protect, sandboxRoutes);

// RL Episode Log
app.use("/api/episode", protect, episodeRoutes);


// ====================
// 404 Handler
// ====================
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: "Route not found",
  });
});

// ====================
// Global Error Handler
// ====================
app.use((err, req, res, next) => {
  console.error(err);

  res.status(err.status || 500).json({
    success: false,
    message: err.message || "Internal Server Error",
  });
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(
    `🚀 Server running in ${process.env.NODE_ENV || "development"} mode on port ${PORT}`,
  );
});
