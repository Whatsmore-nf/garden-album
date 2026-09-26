import type { DB } from "./db/schema";
import type { VlmService } from "./inference/vlm";
import type { Vocab } from "./ingest/tagging";
import { planQuery, expandWithModel, runQuery, type QueryPlan } from "./query/engine";
import { parseTimeRange } from "./query/time-rules";
// 智能相册 = 持久化的查询 DSL。
// “AI 自动持续更新”的本质：相册不物化照片列表，打开时重新执行查询。
// 相对时间（“最近30天”“去年夏天”）必须存**原始短语**而不是绝对时间戳，
// 打开时重跑 parseTimeRange 才能让相册随“现在”滚动；否则相册就是一张快照。
export interface AlbumDSL {
  v: 1;
  raw: string;
  timeExpr?: string | null;                                    // 相对/绝对时间短语原文
  time?: { start: number; end: number; label: string };        // 兼容旧库的绝对区间
  mustSlugs: string[];
  mustNotSlugs: string[];
  persons: string[];
  noStrangers: boolean;
  minFaces?: number;
}
const isObject = (x: unknown): x is Record<string, unknown> =>
  !!x && typeof x === "object" && !Array.isArray(x);
const isFiniteNum = (x: unknown): x is number =>
  typeof x === "number" && Number.isFinite(x);
const asStrArray = (x: unknown): string[] =>
  Array.isArray(x) ? x.filter((s): s is string => typeof s === "string") : [];
export async function createSmartAlbum(
  db: DB, vlm: VlmService, vocab: Vocab, name: string, naturalQuery: string
): Promise<number> {
  const plan = planQuery(naturalQuery, vocab);
  if (plan.residual.length >= 2) {
    try { await expandWithModel(plan, vlm, vocab); }
    catch { /* 展开失败也能建相册，只是召回低 */ }
  }
  const byId = new Map(vocab.map(v => [v.id, v.slug]));
  const dsl: AlbumDSL = {
    v: 1, raw: naturalQuery,
    timeExpr: plan.time?.label ?? null,                         // ★ 存短语不存时间戳
    mustSlugs:    plan.must.map(id => byId.get(id)).filter((s): s is string => !!s),
    mustNotSlugs: plan.mustNot.map(id => byId.get(id)).filter((s): s is string => !!s),
    persons: plan.persons, noStrangers: plan.noStrangers, minFaces: plan.minFaces,
  };
  return Number(db.prepare(
    `INSERT INTO smart_albums (name, dsl, created_at) VALUES (?, ?, ?)`
  ).run(name, JSON.stringify(dsl), Date.now()).lastInsertRowid);
}
export function listSmartAlbums(db: DB): unknown[] {
  return db.prepare(
    `SELECT id, name, created_at FROM smart_albums ORDER BY created_at DESC`
  ).all();
}
export function openSmartAlbum(db: DB, id: number, vocab: Vocab) {
  const album = db.prepare(`SELECT id, name, dsl FROM smart_albums WHERE id = ?`).get(id) as
    { id: number; name: string; dsl: string } | undefined;
  if (!album) return null;
  // ★ dsl 反序列化后不可信：'null'/'[]'/'"x"' 都能过 JSON.parse，必须做类型守卫
  let parsed: unknown;
  try { parsed = JSON.parse(album.dsl); }
  catch { return { album: { id: album.id, name: album.name }, photos: [] }; }
  if (!isObject(parsed)) {
    return { album: { id: album.id, name: album.name }, photos: [] };
  }
  const dsl = parsed as Partial<AlbumDSL>;
  // 打开时校验：词表演化后失效的 slug 静默丢弃
  const bySlug = new Map(vocab.map(v => [v.slug, v.id]));
  // 时间：优先重跑短语（相对时间滚动）；旧库回退到绝对值但要做 finite 校验
  let time = undefined as QueryPlan["time"];
  if (typeof dsl.timeExpr === "string" && dsl.timeExpr.length > 0) {
    time = parseTimeRange(dsl.timeExpr) ?? undefined;
  } else if (isObject(dsl.time) && isFiniteNum((dsl.time as any).start) && isFiniteNum((dsl.time as any).end)) {
    const t = dsl.time as { start: number; end: number; label?: unknown };
    time = { start: t.start, end: t.end, label: String(t.label ?? "") };
  }
  const minFaces = isFiniteNum(dsl.minFaces) && dsl.minFaces > 0 ? Math.floor(dsl.minFaces) : undefined;
  const plan: QueryPlan = {
    raw: typeof dsl.raw === "string" ? dsl.raw : "",
    time,
    must: [], mustNot: [],
    persons: asStrArray(dsl.persons),
    noStrangers: !!dsl.noStrangers,
    minFaces,
    orFallback: false, residual: "",
  };
  for (const s of asStrArray(dsl.mustSlugs))    { const t = bySlug.get(s); if (t != null) plan.must.push(t); }
  for (const s of asStrArray(dsl.mustNotSlugs)) { const t = bySlug.get(s); if (t != null) plan.mustNot.push(t); }
  return { album: { id: album.id, name: album.name }, photos: runQuery(db, plan) };
}
