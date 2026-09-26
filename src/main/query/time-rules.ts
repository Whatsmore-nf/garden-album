export interface TimeRange { start: number; end: number; label: string }
const DAY = 86_400_000;
const SEASON_MONTHS: Record<string, [number, number]> =
  { 春天: [3, 5], 夏天: [6, 8], 秋天: [9, 11], 冬天: [12, 2] };
const SEASON_YEAR_OFFSET: Record<string, number> =
  { 前年: -2, 去年: -1, 今年: 0, 上个: -1, 上一: -1 };
const UNIT_DAYS: Record<string, number> =
  { 天: 1, 日: 1, 周: 7, 星期: 7, 个月: 30, 月: 30, 年: 365 };
const CN_DIGITS: Record<string, number> =
  { 一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9 };
/** 中文数字解析（支持 一/两/…/十/十一/二十/二十三 与阿拉伯数字） */
export function parseCnNum(s: string): number | null {
  if (/^\d+$/.test(s)) return +s;
  if (s === "十") return 10;
  if (/^[一两二三四五六七八九]十$/.test(s)) return (CN_DIGITS[s[0]] ?? 0) * 10;
  if (/^十[一两二三四五六七八九]$/.test(s)) return 10 + (CN_DIGITS[s[1]] ?? 0);
  if (/^[一两二三四五六七八九]十[一两二三四五六七八九]$/.test(s))
    return (CN_DIGITS[s[0]] ?? 0) * 10 + (CN_DIGITS[s[2]] ?? 0);
  return CN_DIGITS[s] ?? null;
}
/** 纯规则时间解析。规则顺序即优先级，新增模式时必须同步修改 stripTimePhrases */
export function parseTimeRange(q: string, now = new Date()): TimeRange | null {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  // 1) 季节（可带 前年/去年/今年/上个 前缀；冬季跨年）
  let m = q.match(/(前年|去年|今年|上[个一]?)?\s*的?\s*(春天|夏天|秋天|冬天)/);
  if (m) {
    const off = m[1] ? (SEASON_YEAR_OFFSET[m[1]] ?? 0) : 0;
    const [m1, m2] = SEASON_MONTHS[m[2]];
    const y = now.getFullYear() + off;
    return m1 <= m2
      ? { start: +new Date(y, m1 - 1, 1), end: +new Date(y, m2, 1), label: m[0] }
      : { start: +new Date(y - 1, 11, 1), end: +new Date(y, 2, 1), label: m[0] };
  }
  // 2) “最近30天 / 近3个月 / 过去两周”（支持中文数字）
  m = q.match(/(?:最?近|过去)\s*(\d+|[一两二三四五六七八九十]+)\s*(天|日|周|星期|个月|月|年)/);
  if (m) {
    const n = /^\d+$/.test(m[1]) ? +m[1] : (parseCnNum(m[1]) ?? 1);
    const ms = n * (UNIT_DAYS[m[2]] ?? 1) * DAY;
    return { start: now.getTime() - ms, end: now.getTime(), label: m[0] };
  }
  // 3) 裸“最近/近期” → 默认 30 天
  m = q.match(/最近|近期/);
  if (m) return { start: now.getTime() - 30 * DAY, end: now.getTime(), label: m[0] };
  // 4) 今天/昨天/前天
  m = q.match(/今天|昨天|前天/);
  if (m) {
    const back = ({ 今天: 0, 昨天: 1, 前天: 2 } as Record<string, number>)[m[0]]!;
    return { start: startOfToday - back * DAY, end: startOfToday + DAY, label: m[0] };
  }
  // 5) 上/这/本 + 周（自然周，周一起）或 月（自然月）
  m = q.match(/(上|这|本)一?个?(周|星期|月)/);
  if (m) {
    if (m[2] === "月") {
      const s = new Date(now.getFullYear(), now.getMonth() - (m[1] === "上" ? 1 : 0), 1);
      return { start: +s, end: +new Date(s.getFullYear(), s.getMonth() + 1, 1), label: m[0] };
    }
    const dow = (now.getDay() + 6) % 7;
    const monday = startOfToday - dow * DAY;
    return m[1] === "上"
      ? { start: monday - 7 * DAY, end: monday, label: m[0] }
      : { start: monday, end: monday + 7 * DAY, label: m[0] };
  }
  // 6) 绝对：2023年3月 / 2023年（月份做边界校验，'2023年13月' 视为非法）
  m = q.match(/(20\d{2})\s*年\s*(\d{1,2})\s*月?/);
  if (m) {
    const mo = +m[2];
    if (mo < 1 || mo > 12) return null;
    return { start: +new Date(+m[1], mo - 1, 1), end: +new Date(+m[1], mo, 1), label: m[0] };
  }
  m = q.match(/(20\d{2})\s*年/);
  if (m) return { start: +new Date(+m[1], 0, 1), end: +new Date(+m[1] + 1, 0, 1), label: m[0] };
  return null;
}
// 与 parseTimeRange 的模式表保持同步（必须用字符串构造，正则字面量跨行书写会导致失效）。
const TIME_PHRASE_SRC =
  "(?:(?:前年|去年|今年|上[个一]?)?\\s*的?\\s*(?:春天|夏天|秋天|冬天))" +
  "|(?:(?:最?近|过去)\\s*(?:\\d+|[一两二三四五六七八九十]+)\\s*(?:天|日|周|星期|个月|月|年))" +
  "|(?:最近|近期)" +
  "|(?:今天|昨天|前天)" +
  "|(?:(?:上|这|本)一?个?(?:周|星期|月))" +
  "|(?:20\\d{2}\\s*年\\s*\\d{1,2}\\s*月?)" +
  "|(?:20\\d{2}\\s*年)";
/** 剥离全部时间短语。用途：1) 标签匹配前清理；2) fuzzy 判断前剥掉“最近30天”的“最” */
export const stripTimePhrases = (q: string): string =>
  q.replace(new RegExp(TIME_PHRASE_SRC, "g"), "");
