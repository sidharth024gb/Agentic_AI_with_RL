import jwt from "jsonwebtoken";
import User from "../models/User.js";
import { REWARDS } from "../utils/rewards.js"

export const protect = async (req, res, next) => {
  try {
    let token;

    if (
      req.headers.authorization &&
      req.headers.authorization.startsWith("Bearer ")
    ) {
      token = req.headers.authorization.split(" ")[1];
    }

    if (!token) {
      return res.status(401).json({
        success: false,
        message: "Not authorized. No token provided.",
      });
    }

    const decoded = jwt.verify(
      token,
      process.env.JWT_SECRET || "sandbox_secret",
    );

    const user = await User.findById(decoded.id).select("-password");

    if (!user) {
      return res.status(401).json({
        success: false,
        message: "User no longer exists.",
      });
    }

    req.user = user;

    next();
  } catch (error) {
    return res.status(401).json({
      success: false,
      message: "Not authorized. Invalid token.",
    });
  }
};

export const authorizePermission = (...permissions) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: "Not authenticated",
      });
    }

    const userPermissions = req.user.permissions || [];

    const hasPermission = permissions.some((permission) =>
      userPermissions.includes(permission),
    );

    if (!hasPermission) {
      return res.status(403).json({
        success: false,
        reward: REWARDS.UNAUTHORIZED_ACTION,
        message: "Insufficient permissions",
      });
    }

    next();
  };
};