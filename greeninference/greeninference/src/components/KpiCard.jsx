import Card from "./Card";

export default function KpiCard({
  label,
  value,
  hint,
  accent = "emerald",
  badge = "",
  className = "",
  valueClassName = "",
  labelClassName = "",
  hintClassName = "",
}) {
  const isSystemState = label === "System State";
  const isThermalLag = label === "Thermal Lag";

  const accentMap = {
    emerald: "ds-kpi--success",
    amber: "ds-kpi--warning",
    teal: "ds-kpi--neutral",
    lime: "ds-kpi--neutral",
    neutral: "ds-kpi--neutral",
  };

  const badgeToneMap = {
    estimated: "border-[#E2C89E] bg-[#FFF4E5] text-[#A15C00]",
    observed: "border-[#BDD0FF] bg-[#E8F0FF] text-[#1F4FD8]",
    research: "border-[#C9DCCB] bg-[#EEF5EF] text-[#1E7D3A]",
  };

  const normalizedBadge = String(badge || "").toLowerCase().trim();

  return (
    <Card className={className}>
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <div
            className={[
              "ds-kpi__label",
              isSystemState ? "font-extrabold text-[#8E531D]" : "",
              isThermalLag ? "font-semibold text-[#46525D]" : "",
              labelClassName,
            ]
              .join(" ")
              .trim()}
          >
            {label}
          </div>

          {badge ? (
            <span
              className={[
                "inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em]",
                badgeToneMap[normalizedBadge] ||
                  "border-[#D3D9D0] bg-[#F3F5F0] text-[#5C645D]",
              ]
                .join(" ")
                .trim()}
            >
              {badge}
            </span>
          ) : null}
        </div>

        <div
          className={[
            "ds-kpi__value",
            accentMap[accent] || "ds-kpi--success",
            isSystemState
              ? "font-black text-[#8E531D] drop-shadow-[0_2px_8px_rgba(180,105,31,0.12)]"
              : "",
            isThermalLag ? "font-extrabold" : "",
            valueClassName,
          ]
            .join(" ")
            .trim()}
        >
          {value}
        </div>

        {hint && (
          <div
            className={[
              "ds-kpi__hint",
              isSystemState ? "font-medium text-[#8E531D]" : "",
              hintClassName,
            ]
              .join(" ")
              .trim()}
          >
            <span className="meta-badge">{hint}</span>
          </div>
        )}
      </div>
    </Card>
  );
}