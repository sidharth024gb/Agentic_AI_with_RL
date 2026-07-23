import { Schema, model } from "mongoose";

const accountSchema = new Schema({
  accountNumber: {
    type: String,
    required: true,
    unique: true,
  },
  accountName: {
    type: String,
    required: true,
  },
  balance: {
    type: Number,
    required: true,
    default: 1000000, // Initial corporate treasury seed balance
  },
  currency: {
    type: String,
    default: "GBP",
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("Account", accountSchema);
