import { Schema, model } from "mongoose";

const supplierSchema = new Schema({
  supplierCode: {
    type: String,
    unique: true,
  },

  supplierName: {
    type: String,
    required: true,
  },

  active: {
    type: Boolean,
    default: true,
  },

  riskScore: {
    type: Number,
    default: 10,
  },

  preferredPaymentMethod: {
    type: String,
    enum: ["BANK", "WIRE", "CARD"],
    default: "BANK",
  },

  country: String,

  rating: {
    type: Number,
    min: 1,
    max: 5,
    default: 5,
  },
});

export default model("Supplier", supplierSchema);