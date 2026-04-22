export default function SignalItem({ text }) {
  return (
    <div className="signal-item flex items-start gap-2 text-sm text-[#55635B]">
      <span className="mt-[0.35rem] h-1.5 w-1.5 rounded-full bg-[#8AA08E]" />
      <span>{text}</span>
    </div>
  );
}