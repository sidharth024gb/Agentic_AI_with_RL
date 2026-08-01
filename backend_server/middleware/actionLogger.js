import { createAuditLog } from "../services/auditService.js";

export const actionLogger = (action) => {
  return async (req, res, next) => {
    res.on("finish", async () => {
      try {
        await createAuditLog({
          user: req.user?._id,

          action,

          entityType: "API",

          entityId: req.params.id || null,

          success: res.statusCode < 400,

          message: `${action} executed`,
        });
      } catch (error) {
        console.log("Audit log failed:", error.message);
      }
    });

    next();
  };
};
