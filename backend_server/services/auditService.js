import AuditLog from "../models/AuditLog.js";

export const createAuditLog = async ({
  user,
  action,
  entityType,
  entityId,
  success,
  message,
  reward,
}) => {
  await AuditLog.create({
    user,

    action,

    entityType,

    entityId,

    success,

    message,

    reward,
  });
};
