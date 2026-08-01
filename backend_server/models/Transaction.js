import { Schema, model } from "mongoose";

const transactionSchema = new Schema({
  referenceId: {
    type: String,
    unique: true,
    required: true,
  },

  invoice: {
    type: Schema.Types.ObjectId,
    ref: "Invoice",
  },

  account: {
    type: Schema.Types.ObjectId,
    ref: "Account",
  },

  amount: Number,

  type: {
    type: String,
    enum: ["PAYMENT_OUT", "DEPOSIT_IN", "REFUND"],
  },

  status: {
    type: String,
    enum: ["PENDING", "SUCCESS", "FAILED", "REVERSED"],
    default: "PENDING",
  },

  reconciled: {
    type: Boolean,
    default: false,
  },

  failureReason: String,

  paymentMethod: {
    type: String,
    enum: ["BANK", "WIRE", "CARD"],
  },

  executedAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("Transaction", transactionSchema);