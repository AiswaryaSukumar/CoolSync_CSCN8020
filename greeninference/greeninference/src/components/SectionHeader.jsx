import * as FramerMotion from "framer-motion";

export default function SectionHeader({ eyebrow, title, subtitle, actions = null }) {
  return (
    <FramerMotion.motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="section-shell__head"
    >
      <div className="flex flex-wrap items-start justify-between gap-4 lg:gap-6">
        <div className="min-w-0 max-w-[72ch]">
          {eyebrow && (
            <div className="section-eyebrow">
              <span className="h-2 w-2 rounded-full bg-[var(--color-accent-strong)] shadow-[0_0_0_4px_var(--color-accent-soft)]" />
              {eyebrow}
            </div>
          )}

          <h2 className="section-title">{title}</h2>

          {subtitle ? <p className="section-subtitle">{subtitle}</p> : null}
        </div>

        {actions ? (
          <div className="shrink-0 self-start lg:pt-1">
            {actions}
          </div>
        ) : null}
      </div>
    </FramerMotion.motion.div>
  );
}