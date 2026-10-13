import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import type { RunState } from "../types";

export function StatusBadge({ state }: { state: RunState }) {
  const icon =
    state === "completed" ? (
      <CheckCircle2 size={18} />
    ) : state === "failed" ? (
      <XCircle size={18} />
    ) : (
      <Loader2 className="spin" size={18} />
    );
  return <span className={`status-badge ${state}`}>{icon}{state}</span>;
}
