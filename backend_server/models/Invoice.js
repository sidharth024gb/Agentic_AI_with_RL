import { Schema, model } from "mongoose";
import { randomBytes } from "crypto";

const invoiceSchema = new Schema({
  invoiceNumber: {
    type: String,
    unique: true,
    default: () => `INV-${randomBytes(4).toString("hex").toUpperCase()}`,
  },

  supplier: {
    type: Schema.Types.ObjectId,
    ref: "Supplier",
    required: true,
  },

  amount: {
    type: Number,
    required: true,
  },

  dueDate: {
    type: Date,
    required: true,
  },

  description: String,

  paymentMethod: {
    type: String,
    enum: ["BANK", "WIRE", "CARD"],
  },

  status: {
    type: String,
    enum: ["PENDING_APPROVAL", "APPROVED", "REJECTED", "PAID"],
    default: "PENDING_APPROVAL",
  },

  priority: {
    type: String,
    enum: ["LOW", "MEDIUM", "HIGH"],
    default: "MEDIUM",
  },

  duplicateFlag: {
    type: Boolean,
    default: false,
  },

  requiresManagerApproval: {
    type: Boolean,
    default: false,
  },

  approvedBy: {
    type: Schema.Types.ObjectId,
    ref: "User",
  },

  paymentAttempts: {
    type: Number,
    default: 0,
  },

  category: {
    type: String,
    enum: ["SOFTWARE", "SERVICES", "TRAVEL", "HARDWARE"],
  },

  createdBy: {
    type: Schema.Types.ObjectId,
    ref: "User",
  },

  createdAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("Invoice", invoiceSchema);