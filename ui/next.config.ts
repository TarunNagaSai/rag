import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the project root so Next doesn't pick up a parent-directory lockfile.
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
