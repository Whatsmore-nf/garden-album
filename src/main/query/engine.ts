import type { DB } from "../db/schema";
import type { VlmService } from "../inference/vlm";
import type { Vocab } from "../ingest/tagging";
import { parseTimeRange, stripTimePhrases, parseCnNum, type TimeRange } from "./time-rules";
import { rerank } from "./rerank";
export interface QueryPlan {
  raw: string;
  time?: TimeRange;
  must: number[];
  mustNot: number[];
  persons: string[];
  noStrangers: boolean;
  minFaces?: number;
  orFallback: boolean;
  residual: string;
}
/** 关系白名单：persons:update 的入参校验与 planQuery 的关系词表共用一份 */
export const RELATION_WHITELIST =
  ["self", "father", "mother", "partner", "child", "grandparent"] as const;
export type Relation = typeof RELATION_WHITELIST[number];
const NEGATION = /[没无不]/;
const RELATIONS: Array<[RegExp, Relation]> = [
  [/我(?=[和跟与同])|自己|本人/, "self"],
  [/爸爸|父亲|老爸/, "father"],
  [/妈妈|母亲|老妈/, "mother"],
  [/老公|丈夫|妻子|老婆|伴侣|男朋友|女朋友|对象|爱人/, "partner"],
  [/孩子|儿子|女儿|宝宝|小孩/, "child"],
  [/爷爷|奶奶|外公|外婆|祖父|祖母/, "grandparent"],
];
// 程度语义信号（触发缩略图精排）。判断对象是“剥离时间短语后的原文”，不是 residual
const FUZZY = /(自然|特别|比较|满意|好看|适合|最[好美像喜欢])/;
const STOPCHARS = /[\s\d\x00，。、的了和跟与照片张在是什么时候里那这一些拍有点几看见来去没个天月年周号]/g;
export function planQuery(raw: string, vocab: Vocab): QueryPlan {
  const plan: QueryPlan = {
    raw, must: [], mustNot: [], persons: [],
    noStrangers: false, orFallback: false, residual: "",
  };
  plan.time = parseTimeRange(raw) ?? undefined;
  let s = stripTimePhrases(raw);
  const terms = vocab.flatMap(v => [
    { text: v.zh, id: v.id },
    ...v.aliases.map(a => ({ text: a, id: v.id })),
  ]).sort((a, b) => b.text.length - a.text.length);
  for (const t of terms) {
    const i = s.indexOf(t.text);
    if (i < 0) continue;
    const prefix = s.slice(Math.max(0, i - 2), i);
    (NEGATION.test(prefix) ? plan.mustNot : plan.must).push(t.id);
    s = s.slice(0, i) + "\x00".repeat(t.text.length) + s.slice(i + t.text.length);
  }
  for (const [re, rel] of RELATIONS) {
    const m = s.match(re);
    if (m) { plan.persons.push(rel); s = s.replace(m[0], "\x00"); }
  }
  const stranger = vocab.find(v => v.slug === "stranger");
  if (stranger && plan.mustNot.includes(stranger.id)) plan.noStrangers = true;
  const fm = s.match(/([一两二三四五六七八九十\d]+)\s*个?(?:人|张脸)/);
  if (fm) {
    const n = parseCnNum(fm[1]);
    if (n != null && n > 0) plan.minFaces = n;
    s = s.replace(fm[0], "\x00");
  }
  plan.residual = s.replace(STOPCHARS, "");
  // ★ 去重：'海边和沙滩' 两个 alias 都指向 beach → must=[beach,beach]
  //   不去重会被 search 误判为“多标签 AND 过严”而错误触发 OR 降级
  plan.must    = [...new Set(plan.must)];
  plan.mustNot = [...new Set(plan.mustNot)];
  plan.persons = [...new Set(plan.persons)];
  return plan;
}
export async function expandWithModel(plan: QueryPlan, vlm: VlmService, vocab: Vocab): Promise<void> {
  const system = `你是相册查询解析器。把中文查询片段映射为词表 slug。
只输出 JSON：{"must":["slug"],"mustNot":["slug"]}。只能用给定词表，宁缺勿滥。
词表：${vocab.map(v => v.slug + "=" + v.zh).join(", ")}`;
  const r = await vlm.expandQueryText(
    system,
    `完整查询：${plan.raw}\n需要理解的片段：${plan.residual}`,
  ) as { must?: string[]; mustNot?: string[] };
  const bySlug = new Map(vocab.map(v => [v.slug, v.id]));
  for (const sl of r.must ?? []) {
    const id = bySlug.get(sl);
    if (id != null && !plan.must.includes(id)) plan.must.push(id);
  }
  for (const sl of r.mustNot ?? []) {
    const id = bySlug.get(sl);
    if (id != null && !plan.mustNot.includes(id)) plan.mustNot.push(id);
  }
  plan.residual = "";
}
function buildWhere(p: QueryPlan): { sql: string; params: unknown[] } {
  const w: string[] = [];
  const params: unknown[] = [];
  if (p.time) {
    w.push("p.taken_at >= ? AND p.taken_at < ?");
    params.push(p.time.start, p.time.end);
  }
  if (p.minFaces != null) {
    w.push("(SELECT COUNT(*) FROM photo_faces pf WHERE pf.photo_id = p.id) >= ?");
    params.push(p.minFaces);
  }
  if (p.noStrangers) {
    w.push("NOT EXISTS (SELECT 1 FROM photo_faces pf WHERE pf.photo_id = p.id AND pf.person_id IS NULL)");
  }
  for (const rel of p.persons) {
    w.push(`EXISTS (SELECT 1 FROM photo_faces pf JOIN persons ps ON ps.id = pf.person_id
            WHERE pf.photo_id = p.id AND ps.relation = ?)`);
    params.push(rel);
  }
  for (const id of p.mustNot) {
    w.push("NOT EXISTS (SELECT 1 FROM photo_tags pt WHERE pt.photo_id = p.id AND pt.tag_id = ?)");
    params.push(id);
  }
  if (p.must.length) {
    if (p.orFallback) {
      const ph = p.must.map(() => "?").join(",");
      w.push(`p.id IN (SELECT photo_id FROM photo_tags WHERE tag_id IN (${ph}) GROUP BY photo_id)`);
      params.push(...p.must);
    } else {
      for (const id of p.must) {
        w.push("EXISTS (SELECT 1 FROM photo_tags pt WHERE pt.photo_id = p.id AND pt.tag_id = ?)");
        params.push(id);
      }
    }
  }
  return { sql: w.length ? "WHERE " + w.join(" AND ") : "", params };
}
export function runQuery(db: DB, p: QueryPlan, limit = 200): unknown[] {
  const { sql, params } = buildWhere(p);
  return db.prepare(`SELECT p.* FROM photos p ${sql} ORDER BY p.taken_at DESC LIMIT ?`)
    .all(...params, limit);
}
export function lastSeen(db: DB, relation: string): unknown {
  return db.prepare(`
    SELECT p.path, p.taken_at FROM photos p
    JOIN photo_faces pf ON pf.photo_id = p.id
    JOIN persons ps ON ps.id = pf.person_id
    WHERE ps.relation = ? ORDER BY p.taken_at DESC LIMIT 1
  `).get(relation);
}
export async function search(
  raw: string, db: DB, vlm: VlmService, vocab: Vocab,
  thumbOf: (id: number) => Promise<string | null>,
) {
  const plan = planQuery(raw, vocab);
  if (/(上次|最近一次)/.test(raw) && plan.persons.length > 0) {
    const r = lastSeen(db, plan.persons[0]);
    return { photos: r ? [r] : [], plan, widened: false, reranked: false };
  }
  const fuzzy = FUZZY.test(stripTimePhrases(raw));
  if (plan.residual.length >= 2) {
    try { await expandWithModel(plan, vlm, vocab); }
    catch (e) { console.warn("[search] 查询展开失败，降级为纯规则结果", e); }
  }
  let rows = runQuery(db, plan) as any[];
  let widened = false;
  if (rows.length === 0 && plan.persons.includes("self") && plan.persons.length > 1) {
    rows = runQuery(db, { ...plan, persons: plan.persons.filter(p => p !== "self") });
    widened = true;
  }
  if (rows.length === 0 && plan.must.length > 1) {
    rows = runQuery(db, { ...plan, orFallback: true });
    widened = true;
  }
  if (fuzzy && rows.length > 0 && rows.length <= 60) {
    try { rows = await rerank(raw, rows, vlm, thumbOf); }
    catch (e) { console.warn("[search] 精排失败，保持规则序", e); }
  }
  return { photos: rows, plan, widened, reranked: fuzzy };
}
