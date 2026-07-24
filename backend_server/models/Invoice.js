import { Schema, model } from "mongoose";
import crypto from "crypto"

const invoiceSchema = new Schema({
  invoiceNumber: {
    type: String,
    required: true,
    unique: true,
    // ✅ Automatically generates a random 8-character hex string on creation
    default: () => `INV-${crypto.randomBytes(4).toString("hex").toUpperCase()}`,
  },
  vendorName: {
    type: String,
    required: true,
  },
  amount: {
    type: Number,
    required: true,
    min: 0,
  },
  dueDate: {
    type: Date,
    required: true,
  },
  description: {
    type: String,
    required: true,
  },
  paymentMethod: {
    type: String,
  },
  status: {
    type: String,
    enum: ["PENDING_APPROVAL", "APPROVED", "REJECTED", "PAID"],
    default: "PENDING_APPROVAL",
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
