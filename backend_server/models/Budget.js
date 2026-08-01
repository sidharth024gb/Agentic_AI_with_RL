import { Schema, model } from "mongoose";

const budgetSchema = new Schema({
  department: {
    type: String,
    unique: true,
  },

  monthlyBudget: Number,

  remainingBudget: Number,

  manager: {
    type: Schema.Types.ObjectId,
    ref: "User",
  },

  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("Budget", budgetSchema);
