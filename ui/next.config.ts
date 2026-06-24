import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a self-contained server bundle (.next/standalone + server.js) for the
  // Docker runner stage.
  output: "standalone",
  // Pin the project root so Next doesn't pick up a parent-directory lockfile.
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
