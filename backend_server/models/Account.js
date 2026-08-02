import { Schema, model } from "mongoose";

const accountSchema = new Schema({
  accountNumber: {
    type: String,
    unique: true,
    required: true,
  },

  accountName: String,

  accountType: {
    type: String,
    enum: ["TREASURY", "OPERATIONS", "EXPENSE", "PAYMENT"],
    default: "PAYMENT",
  },

  balance: {
    type: Number,
    default: 1000000,
  },

  currency: {
    type: String,
    default: "GBP",
  },

  frozen: {
    type: Boolean,
    default: false,
  },

  dailyTransferLimit: {
    type: Number,
    default: 100000,
  },

  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("Account", accountSchema);