import { Schema, model } from "mongoose";

const transactionSchema = new Schema({
  referenceId: {
    type: String,
    required: true,
    unique: true,
  },
  invoiceId: {
    type: Schema.Types.ObjectId,
    ref: "Invoice",
    default: null,
  },
  accountId: {
    type: Schema.Types.ObjectId,
    ref: "Account",
    required: true,
  },
  type: {
    type: String,
    enum: ["PAYMENT_OUT", "DEPOSIT_IN"],
    required: true,
  },
  amount: {
    type: Number,
    required: true,
  },
  reconciled: {
    type: Boolean,
    default: false,
  },
  executedAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("Transaction", transactionSchema);
