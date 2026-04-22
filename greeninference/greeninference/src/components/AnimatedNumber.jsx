import { useEffect, useRef, useState } from "react";

export default function AnimatedNumber({
  value,
  decimals = 2,
  duration = 600
}) {
  const [display, setDisplay] = useState(value ?? 0);

  const rafRef = useRef(null);
  const startRef = useRef(null);
  const fromRef = useRef(display);

  useEffect(() => {
    const to = Number(value ?? 0);
    const from = Number(fromRef.current ?? 0);

    cancelAnimationFrame(rafRef.current);
    startRef.current = null;

    const step = (ts) => {
      if (!startRef.current) startRef.current = ts;

      const progress = Math.min((ts - startRef.current) / duration, 1);

      // smooth easing
      const eased = 1 - Math.pow(1 - progress, 3);

      const current = from + (to - from) * eased;

      setDisplay(current);

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        fromRef.current = to;
      }
    };

    rafRef.current = requestAnimationFrame(step);

    return () => cancelAnimationFrame(rafRef.current);
  }, [value, duration]);

  return display.toFixed(decimals);
}