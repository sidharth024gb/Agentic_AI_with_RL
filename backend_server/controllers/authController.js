import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";

import User from "../models/User.js";

const generateToken = (id) => {
  return jwt.sign({ id }, process.env.JWT_SECRET || "sandbox_secret", {
    expiresIn: "7d",
  });
};

const getPermissions = (role) => {
  switch (role) {
    case "ADMIN":
      return [
        "READ_INVOICE",
        "CREATE_INVOICE",
        "UPDATE_INVOICE",
        "APPROVE_INVOICE",
        "REJECT_INVOICE",
        "PAY_INVOICE",
        "REFUND_PAYMENT",
        "CHECK_BUDGET",
        "READ_ACCOUNT",
        "TRANSFER_MONEY",
        "VALIDATE_SUPPLIER",
        "BLACKLIST_SUPPLIER",
        "GENERATE_REPORT",
        "VIEW_AUDIT",
        "RESET_ENVIRONMENT",
        "VIEW_STATE",
      ];

    case "FINANCE_MANAGER":
      return [
        "READ_INVOICE",
        "CREATE_INVOICE",
        "UPDATE_INVOICE",
        "APPROVE_INVOICE",
        "REJECT_INVOICE",
        "PAY_INVOICE",
        "CHECK_BUDGET",
        "READ_ACCOUNT",
        "TRANSFER_MONEY",
        "VALIDATE_SUPPLIER",
        "GENERATE_REPORT",
        "VIEW_STATE",
      ];

    case "AGENT_BOT":
      return [
        "READ_INVOICE",
        "APPROVE_INVOICE",
        "PAY_INVOICE",
        "CHECK_BUDGET",
        "READ_ACCOUNT",
        "VALIDATE_SUPPLIER",
        "GENERATE_REPORT",
        "RESET_ENVIRONMENT",
        "VIEW_STATE",
      ];

    default:
      return [];
  }
};

export async function register(req, res) {
  try {
    const { username, email, password, role = "AGENT_BOT" } = req.body;

    if (!username || !email || !password) {
      return res.status(400).json({
        success: false,
        message: "Username, email and password are required.",
      });
    }

    const exists = await User.findOne({
      $or: [{ email }, { username }],
    });

    if (exists) {
      return res.status(400).json({
        success: false,
        message: "User already exists.",
      });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const user = await User.create({
      username,
      email,
      password: hashedPassword,
      role,
      permissions: getPermissions(role),
    });

    const token = generateToken(user._id);

    return res.status(201).json({
      success: true,
      message: "User registered successfully.",

      token,

      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        role: user.role,
        permissions: user.permissions,
      },
    });
  } catch (err) {
    return res.status(500).json({
      success: false,
      environmentError: true,
      message: err.message,
    });
  }
}

export async function login(req, res) {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });

    if (!user) {
      return res.status(401).json({
        success: false,
        message: "Invalid email or password.",
      });
    }

    const validPassword = await bcrypt.compare(password, user.password);

    if (!validPassword) {
      return res.status(401).json({
        success: false,
        message: "Invalid email or password.",
      });
    }

    const token = generateToken(user._id);

    return res.json({
      success: true,
      message: "Login successful.",

      token,

      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        role: user.role,
        permissions: user.permissions,
      },
    });
  } catch (err) {
    return res.status(500).json({
      success: false,
      environmentError: true,
      message: err.message,
    });
  }
}

export async function getProfile(req, res) {
  try {
    return res.status(200).json({
      success: true,

      user: {
        id: req.user._id,
        username: req.user.username,
        email: req.user.email,
        role: req.user.role,
        permissions: req.user.permissions,
        createdAt: req.user.createdAt,
      },
    });
  } catch (err) {
    return res.status(500).json({
      success: false,
      environmentError: true,
      message: err.message,
    });
  }
}
