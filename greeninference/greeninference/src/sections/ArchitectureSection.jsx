import React, { useMemo, useState } from "react";
import SectionHeader from "../components/SectionHeader";
import Card from "../components/Card";
import SignalItem from "../components/SignalItem";
import StatusPill from "../components/StatusPill";
import architectureImg from "../assets/architecture.png";

const controlNodes = [
  { title: "Request", subtitle: "User task", emphasis: "edge" },
  { title: "Estimate", subtitle: "Work and heat check", emphasis: "support" },
  { title: "Policy", subtitle: "Decision point", emphasis: "primary" },
  { title: "Cooling", subtitle: "Response timing", emphasis: "support" },
  { title: "Results", subtitle: "Impact view", emphasis: "edge" },
];

const energyEvidenceCards = [
  {
    title: "Cooling share",
    value: "30–50%",
    meta: "Research-backed",
    text: "Cooling can account for 30–50% of total data center energy use.",
    tone: "emerald",
  },
  {
    title: "Why this matters",
    value: "Cooling starts late",
    meta: "Simulated view",
    text: "Cooling that reacts late continues using energy after the spike.",
    tone: "amber",
  },
  {
    title: "Research example",
    value: "Up to 40%",
    meta: "Research-based",
    text: "DeepMind reported up to 40% cooling reduction.",
    tone: "neutral",
  },
];

const regionalExposureCards = [
  {
    region: "Texas",
    level: "Higher",
    text: "This modeled comparison shows higher downstream carbon exposure in this view.",
  },
  {
    region: "California",
    level: "Moderate",
    text: "This modeled comparison shows meaningful carbon exposure, but below the highest case here.",
  },
  {
    region: "Germany",
    level: "Moderate",
    text: "This modeled comparison shows that the same workload can still create notable downstream impact.",
  },
  {
    region: "France",
    level: "Lower",
    text: "This modeled comparison shows lower downstream carbon exposure in this view.",
  },
];

const impactInsightCards = [
  {
    title: "Cooling delay",
    value: "Extra energy can continue after the spike",
    text: "Late cooling can keep facility overhead active after compute demand has already risen.",
  },
  {
    title: "Regional carbon exposure",
    value: "The same workload can have different impact",
    text: "Location changes downstream carbon impact, even when the workload itself stays the same.",
  },
  {
    title: "Earlier coordination",
    value: "Lower total impact is more likely",
    text: "Earlier decisions can help reduce both cooling overhead and carbon exposure.",
  },
];

export default function ArchitectureSection() {
  const [mode, setMode] = useState("coordinated");

  const pipelineSignals = useMemo(() => {
    if (mode === "coordinated") {
      return [
        "Policy → coordinated mode selected",
        "Cooling window → opened earlier",
        "Expected effect → lower extra energy use",
      ];
    }

    return [
      "Policy → reactive mode retained",
      "Cooling window → opens after the load spike",
      "Expected effect → lag stays open longer",
    ];
  }, [mode]);

  const decisionTrace = useMemo(() => {
    if (mode === "coordinated") {
      return [
        { label: "Mode", value: "Coordinated" },
        { label: "Cooling action", value: "Starts earlier" },
        { label: "Expected effect", value: "Lag window reduced" },
      ];
    }

    return [
      { label: "Mode", value: "Reactive" },
      { label: "Cooling action", value: "Follows demand later" },
      { label: "Expected effect", value: "Lag window remains open" },
    ];
  }, [mode]);

  const strongestRegion = useMemo(() => {
    return (
      regionalExposureCards.find((item) => item.level === "Higher") ||
      regionalExposureCards[0]
    );
  }, []);

  return (
    <section className="space-y-10 control-section">
      <div id="control" className="section-inner section-inner--offset space-y-6">
        <div className="decision-layer-shell">
          <div className="decision-layer-shell__header">
            <SectionHeader
              eyebrow="Control"
              title="Decision Layer"
              subtitle="How the app chooses between normal cooling and earlier cooling"
              actions={
                <div className="flex flex-wrap items-center gap-2">
                  <StatusPill tone="neutral">Modeled view</StatusPill>
                  <StatusPill tone="warn">Policy active</StatusPill>
                </div>
              }
            />
          </div>

          <div className="decision-layer-grid">
            <Card className="space-y-5 control-evidence-card">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-2">
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                    Flow
                  </div>
                  <div className="text-[1.05rem] font-semibold text-[#213428]">
                    Policy-centered operating flow
                  </div>
                  <div className="max-w-2xl text-sm text-[#55635B]">
                    The decision point sits in the middle of the system. It checks the request,
                    estimates likely heat, and then changes the cooling response.
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <StatusPill tone="neutral">Simple flow</StatusPill>
                  <StatusPill tone="warn">
                    {mode === "coordinated"
                      ? "Coordinated path active"
                      : "Reactive path active"}
                  </StatusPill>
                </div>
              </div>

              <div className="control-pipeline-grid" aria-label="Control pipeline flow">
                {controlNodes.map((node, index) => {
                  const nodeClass =
                    node.emphasis === "primary"
                      ? "policy-node control-policy-node"
                      : node.emphasis === "edge"
                        ? "opacity-80 border-[rgba(92,116,98,0.08)] bg-[rgba(255,255,255,0.64)]"
                        : "border-[rgba(92,116,98,0.1)] bg-[rgba(255,255,255,0.78)]";

                  return (
                    <React.Fragment key={node.title}>
                      <div
                        className={[
                          "control-pipeline-node rounded-[20px] border px-4 py-4",
                          nodeClass,
                        ].join(" ")}
                      >
                        <div className="text-sm font-semibold text-[#213428]">
                          {node.title}
                        </div>
                        <div className="mt-2 text-xs uppercase tracking-[0.2em] text-[#7D8B82]">
                          {node.subtitle}
                        </div>
                      </div>

                      {index < controlNodes.length - 1 && (
                        <div className="control-pipeline-arrow opacity-75" aria-hidden="true">
                          →
                        </div>
                      )}
                    </React.Fragment>
                  );
                })}
              </div>

              <div className="rounded-[20px] border border-[#E3EBE4] bg-[#F7FAF7] px-4 py-4">
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#6E7F71]">
                  Decision trace
                </div>
                <div className="mt-3 grid gap-3 md:grid-cols-3">
                  {decisionTrace.map((item, index) => (
                    <div
                      key={item.label}
                      className={[
                        "rounded-[16px] border px-4 py-3",
                        index === 0
                          ? "border-[rgba(104,173,63,0.16)] bg-[#F2F8EF]"
                          : "border-[#DCE6DE] bg-white/80",
                      ].join(" ")}
                    >
                      <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#7D8B82]">
                        {item.label}
                      </div>
                      <div className="mt-2 text-sm font-semibold text-[#213428]">
                        {item.value}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>

            <div className="control-right-stack">
              <Card className="space-y-4 control-stack-card control-state-card">
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-2">
                    <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                      State
                    </div>
                    <div className="text-lg font-semibold text-[#213428]">
                      Policy flow is visible
                    </div>
                    <div className="text-sm text-[#55635B]">
                      The policy layer is where workload, cooling, and impact signals come
                      together.
                    </div>
                  </div>
                  <StatusPill tone="neutral">Modeled view</StatusPill>
                </div>

                <div className="live-interpretation space-y-2 rounded-[18px] border border-[#E7EFE8] bg-[#F7FAF7] px-4 py-4">
                  <div className="text-xs uppercase tracking-[0.2em] text-[#7D8B82]">
                    Live interpretation
                  </div>
                  {pipelineSignals.map((signal) => (
                    <SignalItem key={signal} text={signal} />
                  ))}
                </div>
              </Card>

              <Card className="space-y-4 control-decision-card">
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-2">
                    <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                      Control
                    </div>
                    <div className="text-lg font-semibold text-[#182A1F] control-decision-card__title">
                      Decision mode
                    </div>
                  </div>
                  <StatusPill tone="success">Mode switch</StatusPill>
                </div>

                <div className="flex flex-wrap gap-2">
                  {["reactive", "coordinated"].map((option) => {
                    const active = mode === option;

                    return (
                      <button
                        key={option}
                        type="button"
                        onClick={() => setMode(option)}
                        className={[
                          "rounded-full border px-4 py-2 text-sm font-medium transition",
                          active
                            ? "border-[#5A7B68] bg-[#ECF4EE] text-[#213428]"
                            : "border-[#DCE6DE] bg-white/80 text-[#5A665F] hover:border-[#B6C7B8]",
                        ].join(" ")}
                      >
                        {option === "reactive" ? "Reactive" : "Coordinated"}
                      </button>
                    );
                  })}
                </div>

                <div className="control-decision-compare">
                  <div className="control-decision-compare__item control-decision-compare__item--reactive">
                    Reactive → cooling starts later
                  </div>
                  <div className="control-decision-compare__item control-decision-compare__item--coordinated">
                    Coordinated → cooling starts earlier
                  </div>
                </div>

                <div className="rounded-[18px] border border-[#DCE6DE] bg-white/70 px-4 py-3 text-sm text-[#55635B]">
                  {mode === "coordinated"
                    ? "Policy outcome → cooling begins earlier, before the lag window grows too much."
                    : "Policy outcome → cooling follows the event after lag has already formed."}
                </div>
              </Card>

              <Card className="space-y-3 control-stack-card control-constraints-card">
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                  Constraints
                </div>
                <div className="rounded-[18px] border border-[#DCE6DE] bg-white/75 px-4 py-4">
                  <div className="flex flex-col gap-3 md:flex-row md:flex-wrap md:items-center md:gap-4">
                    <div className="rounded-full border border-[rgba(180,105,31,0.14)] bg-[#FBF4EC] px-3 py-1.5 text-sm font-medium text-[#8A5923]">
                      Temperature limit checked
                    </div>
                    <div className="rounded-full border border-[rgba(104,173,63,0.14)] bg-[#F2F8EF] px-3 py-1.5 text-sm font-medium text-[#45682B]">
                      Carbon impact monitored
                    </div>
                    <div className="rounded-full border border-[#DCE6DE] bg-[#F7FAF7] px-3 py-1.5 text-sm font-medium text-[#55635B]">
                      Cost view active
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          </div>

          <div className="decision-layer-architecture">
            <Card className="space-y-4 architecture-visual">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-2">
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                    Architecture
                  </div>
                  <div className="text-[1.05rem] font-semibold text-[#213428]">
                    Decision highlighted in the full system flow
                  </div>
                  <div className="max-w-3xl text-sm text-[#55635B]">
                    This supporting diagram shows where the decision layer sits in the
                    complete request-to-results path. The policy step is the key point that
                    changes how cooling responds.
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <StatusPill tone="neutral">Architecture view</StatusPill>
                  <StatusPill tone="neutral">Decision highlighted</StatusPill>
                </div>
              </div>

              <div className="rounded-[18px] border border-[rgba(104,173,63,0.16)] bg-[#F3F8F0] px-4 py-3 text-sm font-medium text-[#4F6F35] architecture-banner">
                This shows how the system moves from request → decision → cooling → results.
              </div>
              <div className="architecture-image-shell">
                <img
                  src={architectureImg}
                  alt="System architecture showing request, estimate, decision, cooling, and results flow"
                  className="w-full rounded-[18px] object-contain"
                />
                </div>
            </Card>
          </div>
        </div>
      </div>

      <div id="impact" className="section-inner section-inner--offset space-y-8 impact-section">
        <div className="impact-shell">
            <SectionHeader
              eyebrow="Impact"
              title="Impact Analysis"
              subtitle="Cooling delay increases both energy use and carbon impact"
              actions={
                  <div className="flex flex-wrap items-center gap-2">
                  <StatusPill tone="neutral">30–50% cooling share</StatusPill>
                  <StatusPill tone="neutral">Research-based</StatusPill>
                  <StatusPill tone="warn">Regional comparison</StatusPill>
                  <StatusPill tone="neutral">Modeled view</StatusPill>
            </div>
            
          }
        />

        <div className="grid gap-6 lg:grid-cols-2">
          <Card className="space-y-5 impact-evidence-card">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                  Energy
                </div>
                <div className="text-[1.05rem] font-semibold text-[#213428]">
                  Research-backed energy view
                </div>
                <div className="text-sm text-[#55635B]">
                  Cooling is often a large share of total data center energy use.
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <StatusPill tone="neutral">Research-based</StatusPill>
                <StatusPill tone="neutral">Simple evidence</StatusPill>
              </div>
            </div>

            <div className="impact-energy-grid">
              {energyEvidenceCards.map((entry, index) => (
                <div
                  key={entry.title}
                  className={[
                    "impact-mini-card rounded-[18px] border px-4 py-4",
                    index === 0
                      ? "impact-mini-card--primary border-[rgba(104,173,63,0.18)] bg-[#F2F8EF]"
                      : entry.tone === "amber"
                        ? "border-[rgba(180,105,31,0.14)] bg-[#FBF4EC]"
                        : "border-[#DCE6DE] bg-white/80",
                  ].join(" ")}
                >
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#7D8B82]">
                    {entry.title}
                  </div>
                  <div className="mt-2 text-base font-semibold text-[#213428]">
                    {entry.value}
                  </div>
                  <div className="mt-1 text-xs font-medium text-[#6E7F71]">{entry.meta}</div>
                  <div className="mt-3 text-sm text-[#55635B]">{entry.text}</div>
                </div>
              ))}
            </div>

            <div className="rounded-[18px] border border-[#E7EFE8] bg-[#F7FAF7] px-4 py-3 text-sm text-[#55635B] impact-energy-takeaway">
              Cooling is a major driver of facility energy in AI workloads.
            </div>
          </Card>

          <Card className="space-y-5 impact-evidence-card">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                  Carbon
                </div>
                <div className="text-[1.05rem] font-semibold text-[#213428]">
                  Regional carbon exposure
                </div>
                <div className="text-sm text-[#55635B]">
                  This view is comparative, not absolute. It shows that location can change
                  the downstream carbon impact of the same workload.
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <StatusPill tone="warn">Modeled comparison</StatusPill>
                <StatusPill tone="neutral">No exact live values</StatusPill>
              </div>
            </div>

            <div className="rounded-[18px] border border-[#E7EFE8] bg-[#F7FAF7] px-4 py-3 text-sm text-[#55635B]">
              Highest modeled exposure in this comparison:{" "}
              <span className="font-semibold text-[#213428]">{strongestRegion.region}</span>.
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {regionalExposureCards.map((entry) => (
                <div
                  key={entry.region}
                  className="rounded-[18px] border border-[#DCE6DE] bg-white/80 px-4 py-4"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-[#213428]">{entry.region}</div>
                    <span
                      className={[
                        "rounded-full px-2.5 py-1 text-xs font-semibold",
                        entry.level === "Higher"
                          ? "border border-[rgba(180,105,31,0.14)] bg-[#FBF4EC] text-[#8A5923]"
                          : entry.level === "Lower"
                            ? "border border-[rgba(104,173,63,0.16)] bg-[#F2F8EF] text-[#45682B]"
                            : "border border-[#DCE6DE] bg-[#F7FAF7] text-[#55635B]",
                      ].join(" ")}
                    >
                      {entry.level}
                    </span>
                  </div>
                  <div className="mt-3 text-sm text-[#55635B]">{entry.text}</div>
                </div>
              ))}
            </div>

            <div className="rounded-[18px] border border-[#E7EFE8] bg-[#F7FAF7] px-4 py-3 text-sm text-[#55635B]">
              Safe takeaway → the same workload can create different carbon outcomes
              depending on where it runs.
            </div>
          </Card>
        </div>

        <div className="grid gap-4 md:grid-cols-3 impact-insight-row">
          {impactInsightCards.map((card, index) => (
            <Card
              key={card.title}
              className={`space-y-3 impact-insight-card impact-insight-card--${index + 1}`}
            >
              <div className="text-xs font-semibold uppercase tracking-[0.22em] text-[#6E7F71]">
                Insight
              </div>
              <div className="space-y-1">
                <div className="text-[1rem] font-semibold text-[#213428]">{card.title}</div>
                <div className="text-sm font-medium text-[#4D5C53]">{card.value}</div>
              </div>
              <div className="text-sm text-[#55635B]">{card.text}</div>
            </Card>
          ))}
        </div>
      </div>
       </div>
    </section>
  );
}