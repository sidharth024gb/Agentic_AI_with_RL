import express from "express";
import dotenv from "dotenv";
import connectDB from "./config/db.js";

import authRoutes from "./routes/authRoutes.js";
import financeRoutes from "./routes/financeRoutes.js";
import sandboxRoutes from "./routes/sandboxRoutes.js";
import { protect } from "./middleware/authMiddleware.js";

dotenv.config();

const app = express();

// Body Parser Middleware
app.use(express.json());

// Connect Database
connectDB();

// Base route for health check
app.get("/", (req, res) => {
  res.send("Finance Backend Sandbox API Server Running");
});

// Mount Routes
app.use("/api/auth", authRoutes);
app.use("/api/finance", financeRoutes);
app.use("/api/sandbox", sandboxRoutes);

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(
    `Server running in ${process.env.NODE_ENV || "development"} mode on port ${PORT}`,
  );
});
