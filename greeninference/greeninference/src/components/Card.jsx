// src/components/Card.jsx
import * as FramerMotion from "framer-motion";

export default function Card({ className = "", children }) {
  return (
    <FramerMotion.motion.div
      whileHover={{ y: -1.5 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
      className={["ds-card", className].join(" ")}
    >
      <div className="relative z-10">{children}</div>
    </FramerMotion.motion.div>
  );
}
