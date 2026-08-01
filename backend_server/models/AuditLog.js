import { Schema, model } from "mongoose";

const auditLogSchema = new Schema({
  user: {
    type: Schema.Types.ObjectId,
    ref: "User",
  },

  action: String,

  entityType: String,

  entityId: String,

  success: Boolean,

  message: String,

  reward: {
    type: Number,
    default: 0,
  },

  timestamp: {
    type: Date,
    default: Date.now,
  },
});

export default model("AuditLog", auditLogSchema);
