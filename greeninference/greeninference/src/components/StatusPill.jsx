export default function StatusPill({ tone = "neutral", children }) {
  const toneMap = {
    success: "border-[rgba(104,173,63,0.22)] bg-[rgba(236,245,239,0.95)] text-[#567f39]",
    warn: "border-[rgba(169,95,24,0.22)] bg-[rgba(246,239,232,0.96)] text-[#a95f18]",
    neutral: "border-[#D3D9D0] bg-[rgba(236,239,232,0.94)] text-[#5F6B67]",
  };

  return (
    <span
      className={[
        "meta-badge inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold",
        toneMap[tone] || toneMap.neutral,
      ].join(" ")}
    >
      {children}
    </span>
  );
}
