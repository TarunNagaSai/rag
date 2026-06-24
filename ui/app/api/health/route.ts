import { NextResponse } from "next/server";

// Lightweight liveness probe used by the Docker/compose healthcheck.
export const dynamic = "force-dynamic";

export function GET() {
  return NextResponse.json({ status: "ok" });
}
