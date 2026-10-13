import type { ReactNode } from "react";

export function SectionTitle({
  title,
  extra,
}: {
  title: string;
  extra?: ReactNode;
}) {
  return (
    <div className="section-title">
      <h2>{title}</h2>
      {extra}
    </div>
  );
}
