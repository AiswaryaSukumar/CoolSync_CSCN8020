// Parses a user-uploaded schedule CSV into the same events[] format as scenario JSON.
// Expected columns: time_min, prompt, users  (users is optional, defaults to 1)
//
// Example CSV:
//   time_min,prompt,users
//   10,"What is the capital of France?",2
//   38,"Write a Python REST API",3

const BURST_CLASS_LABELS = ["Short", "Medium", "Long", "VeryLong"];

// Estimate burst class from prompt text (mirrors scenarioLoader.estimateClass)
function estimateClass(promptText) {
  const words  = (promptText || "").trim().split(/\s+/).length;
  const tokens = Math.max(1, Math.round(words * 1.35));
  if (tokens < 64)  return 0;
  if (tokens < 256) return 1;
  if (tokens < 800) return 2;
  return 3;
}

export function parseScheduleCsv(text) {
  const lines = text
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(l => l.length > 0);

  if (lines.length < 2) return { events: [], error: "파일이 비어 있거나 헤더만 있습니다." };

  // Parse header (case-insensitive, trim)
  const header = parseRow(lines[0]).map(h => h.toLowerCase().trim());
  const timeIdx   = header.indexOf("time_min");
  const promptIdx = header.indexOf("prompt");
  const usersIdx  = header.indexOf("users");

  if (timeIdx === -1 || promptIdx === -1) {
    return { events: [], error: "필수 컬럼이 없습니다. time_min과 prompt 컬럼이 필요합니다." };
  }

  const events = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = parseRow(lines[i]);
    const time_min = Number(cols[timeIdx]);
    const prompt   = String(cols[promptIdx] ?? "").trim();
    const users    = usersIdx !== -1 ? Math.max(1, Number(cols[usersIdx]) || 1) : 1;

    if (!isFinite(time_min) || !prompt) continue;

    const force_class = estimateClass(prompt);
    events.push({
      time_min,
      prompt,
      users,
      force_class,
      note: `${BURST_CLASS_LABELS[force_class]} ×${users}`,
    });
  }

  if (events.length === 0) {
    return { events: [], error: "유효한 이벤트 행이 없습니다." };
  }

  events.sort((a, b) => a.time_min - b.time_min);
  const totalMinutes = Math.ceil(events[events.length - 1].time_min) + 30;

  return { events, totalMinutes, error: null };
}

// RFC 4180-ish CSV row parser (handles quoted fields with commas/newlines inside)
function parseRow(line) {
  const result = [];
  let cur = "";
  let inQuote = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuote && line[i + 1] === '"') { cur += '"'; i++; }
      else inQuote = !inQuote;
    } else if (ch === ',' && !inQuote) {
      result.push(cur); cur = "";
    } else {
      cur += ch;
    }
  }
  result.push(cur);
  return result;
}

export function scheduleEventsToScenario(events, totalMinutes) {
  return {
    scenario_name: "Custom Schedule",
    description:   "업로드된 스케줄 CSV",
    total_minutes: totalMinutes ?? (Math.ceil(events[events.length - 1]?.time_min ?? 30) + 30),
    events,
  };
}

export const BURST_CLASS_LABEL = BURST_CLASS_LABELS;
export const BURST_CLASS_COLOR = ["#5F6B67", "#7A7F54", "#B4691F", "#8B2020"];
export { estimateClass };
