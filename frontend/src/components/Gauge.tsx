import { STATE_COLOR } from "./ui";
import type { DriverState } from "../types";

const R = 90, CX = 110, CY = 110;
const pt = (deg: number, r = R) => [CX + r * Math.cos((Math.PI * deg) / 180), CY - r * Math.sin((Math.PI * deg) / 180)];

/** Alertness score as a 180 degree instrument dial with tick marks. */
export default function Gauge({ score, state }: { score: number; state: DriverState }) {
  const color = STATE_COLOR[state];
  const end = 180 - (Math.max(0, Math.min(100, score)) / 100) * 180;
  const [sx, sy] = pt(180), [ex, ey] = pt(end), [tx, ty] = pt(0);
  const ticks = Array.from({ length: 21 }, (_, i) => 180 - i * 9);

  return (
    <svg viewBox="0 0 220 135" role="img" aria-label={`Alertness score ${score} out of 100`} className="w-full">
      <path d={`M ${sx} ${sy} A ${R} ${R} 0 0 1 ${tx} ${ty}`} fill="none" stroke="#1b464d" strokeWidth="14" strokeLinecap="round" />
      {score > 0 && (
        <path d={`M ${sx} ${sy} A ${R} ${R} 0 0 1 ${ex} ${ey}`} fill="none" stroke={color} strokeWidth="14"
              strokeLinecap="round" style={{ transition: "stroke .3s" }} />
      )}
      {ticks.map((d, i) => {
        const [x1, y1] = pt(d, 72), [x2, y2] = pt(d, i % 5 === 0 ? 63 : 67);
        return <line key={d} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#2b5a62" strokeWidth={i % 5 === 0 ? 2 : 1} />;
      })}
      <text x={CX} y={CY - 6} textAnchor="middle" fontFamily="Barlow Condensed" fontWeight="700" fontSize="62" fill="#e9f1ef">{score}</text>
      <text x={CX} y={CY + 16} textAnchor="middle" fontSize="11" fill="#93b3b0">alertness out of 100</text>
    </svg>
  );
}
