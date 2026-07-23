import { Schema, model } from "mongoose";

const userSchema = new Schema({
  name: {
    type: String,
    required: true,
    trim: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true,
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
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

export default model("User", userSchema);