import { createElement, useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Cpu,
  LayoutGrid,
  Leaf,
} from "lucide-react";

const links = [
  { id: "overview", label: "Overview", icon: LayoutGrid },
  { id: "thermal", label: "Thermal", icon: Activity },
  { id: "control", label: "Control", icon: Cpu },
  { id: "impact", label: "Impact", icon: BarChart3 },
  { id: "demo", label: "Simulation", icon: Leaf },
];

export default function Navbar() {
  const [activeId, setActiveId] = useState("overview");

  useEffect(() => {
    const sections = links
      .map(({ id }) => document.getElementById(id))
      .filter(Boolean);

    if (!sections.length) return undefined;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (visible?.target?.id) {
          setActiveId(visible.target.id);
        }
      },
      {
        rootMargin: "-28% 0px -52% 0px",
        threshold: [0.2, 0.35, 0.5, 0.65],
      }
    );

    sections.forEach((section) => observer.observe(section));

    return () => observer.disconnect();
  }, []);

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (!el) return;
    setActiveId(id);
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="nav-shell">
      <div className="nav-shell__inner flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <button
          type="button"
          onClick={() => scrollTo("overview")}
          className="flex items-center gap-3 text-left"
          aria-label="Go to Overview"
        >
          <div className="nav-brand-mark">
            <Leaf className="text-[var(--color-accent)]" size={18} />
          </div>

          <div className="leading-tight">
            <div className="font-bold tracking-[var(--tracking-heading)] text-[var(--color-text-strong)]">
              GreenInference
            </div>
            <div className="text-[var(--font-label-size)] text-[var(--color-text-muted)]">
              AI Cooling Decision Support
            </div>
          </div>
        </button>

        <nav
          className="hidden md:flex md:flex-wrap md:items-center md:justify-end md:gap-2"
          aria-label="Section navigation"
        >
          {links.map(({ id, label, icon: Icon }) => {
            const isActive = activeId === id;

            return (
              <button
                key={id}
                type="button"
                onClick={() => scrollTo(id)}
                aria-current={isActive ? "page" : undefined}
                className={[
                  "nav-pill nav-tab inline-flex items-center gap-2 whitespace-nowrap rounded-full px-3.5 py-2 text-sm font-medium transition",
                  isActive
                    ? "active border-[rgba(104,173,63,0.2)] bg-[rgba(244,250,241,0.98)] text-[var(--color-text-strong)] shadow-[0_8px_18px_rgba(56,96,68,0.07)]"
                    : "text-[var(--color-text-body)] hover:border-[rgba(92,116,98,0.16)] hover:bg-[rgba(252,253,250,0.98)]",
                ]
                  .join(" ")
                  .trim()}
              >
                {createElement(Icon, {
                  size: 15,
                  className: isActive
                    ? "text-[var(--color-accent)]"
                    : "text-[var(--color-text-muted)]",
                })}
                <span>{label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </div>
  );
}