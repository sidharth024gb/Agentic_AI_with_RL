import { Schema, model } from "mongoose";

const userSchema = new Schema({
  username: {
    type: String,
    required: true,
  },

  email: {
    type: String,
    unique: true,
    required: true,
  },

  password: {
    type: String,
    required: true,
  },

  role: {
    type: String,
    enum: ["ADMIN", "FINANCE_MANAGER", "AGENT_BOT"],
    default: "AGENT_BOT",
  },

  permissions: [String],

  createdAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("User", userSchema);