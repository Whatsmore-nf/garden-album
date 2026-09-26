import type { DB } from "./db/schema";
import { hammingHex } from "./ingest/phash";
export interface CleanupCandidate { id: number; path: string; reasons: string[] }
export interface CleanupSuggestion {
  id: string;
  kind: "duplicate" | "blurry" | "screenshot";
  keepPath?: string;
  candidates: CleanupCandidate[];
  confidence: number;
}
export function purgePhotoRows(db: DB, photoId: number): void {
  db.transaction(() => {
    db.prepare(`DELETE FROM vec_faces WHERE face_rowid IN
      (SELECT face_rowid FROM photo_faces WHERE photo_id = ?)`).run(photoId);
    db.prepare(`DELETE FROM fts_tags WHERE photo_id = ?`).run(photoId);
    db.prepare(`DELETE FROM fts_ocr  WHERE photo_id = ?`).run(photoId);
    db.prepare(`DELETE FROM photos   WHERE id = ?`).run(photoId);
  })();
}
export function purgePhotoByPath(db: DB, filePath: string): void {
  const row = db.prepare(`SELECT id FROM photos WHERE path = ?`).get(filePath) as
    { id: number } | undefined;
  if (row) purgePhotoRows(db, row.id);
}
export function findDuplicates(db: DB): CleanupSuggestion[] {
  const rows = db.prepare(
    `SELECT id, path, phash, ai_quality FROM photos WHERE phash IS NOT NULL AND stage = 1`
  ).all() as Array<{ id: number; path: string; phash: string; ai_quality: number | null }>;
  const buckets = new Map<string, typeof rows>();
  for (const r of rows) for (let s = 0; s < 8; s++) {
    const key = r.phash.slice(s * 2, s * 2 + 2);
    const arr = buckets.get(key) ?? (buckets.set(key, []).get(key)!);
    arr.push(r);
  }
  const seen = new Set<string>();
  const out: CleanupSuggestion[] = [];
  for (const group of buckets.values()) {
    if (group.length < 2) continue;
    for (let i = 0; i < group.length; i++) for (let j = i + 1; j < group.length; j++) {
      const a = group[i], b = group[j];
      if (a.id === b.id) continue;
      const key = a.id < b.id ? `${a.id}:${b.id}` : `${b.id}:${a.id}`;
      if (seen.has(key)) continue;
      const d = hammingHex(a.phash, b.phash);
      if (d > 6) continue;
      seen.add(key);
      const [keep, drop] = (a.ai_quality ?? 0) >= (b.ai_quality ?? 0) ? [a, b] : [b, a];
      out.push({
        id: `dup-${drop.id}`, kind: "duplicate", keepPath: keep.path,
        candidates: [{
          id: drop.id, path: drop.path,
          reasons: [
            `与保留照片的感知哈希距离 ${d}（阈值 6），内容高度相似`,
            `质量分 ${keep.ai_quality ?? "?"} ≥ ${drop.ai_quality ?? "?"}`,
          ],
        }],
        confidence: 1 - d / 16,
      });
    }
  }
  return out;
}
export function findLowQuality(db: DB): CleanupSuggestion[] {
  const out: CleanupSuggestion[] = [];
  const rows = db.prepare(
    `SELECT id, path, ai_quality, ai_screenshot FROM photos WHERE stage = 1`
  ).all() as Array<{ id: number; path: string; ai_quality: number | null; ai_screenshot: number | null }>;
  for (const r of rows) {
    if ((r.ai_quality ?? 1) < 0.35)
      out.push({
        id: `blur-${r.id}`, kind: "blurry", confidence: 1 - (r.ai_quality ?? 0),
        candidates: [{
          id: r.id, path: r.path,
          reasons: [`模型清晰度评分 ${(r.ai_quality ?? 0).toFixed(2)}（阈值 0.35）`],
        }],
      });
    if ((r.ai_screenshot ?? 0) > 0.85)
      out.push({
        id: `ss-${r.id}`, kind: "screenshot", confidence: r.ai_screenshot ?? 0,
        candidates: [{
          id: r.id, path: r.path,
          reasons: [`截图概率 ${(r.ai_screenshot ?? 0).toFixed(2)}（阈值 0.85）`],
        }],
      });
  }
  return out;
}
/** ⚠️ 唯一删除入口。
 *  1) 只接受 photo id，主进程查库取路径，渲染进程无法任意 rm
 *  2) 只进系统回收站
 *  3) 删除后清 photos / FTS / 向量。purge 单独 try/catch：
 *     文件已进回收站但 purge 抛异常时，如实记入 failed 让人工/重试介入，
 *     而不是把已经完成的删除动作一起吞掉 */
export async function applyConfirmedByIds(
  db: DB, ids: number[]
): Promise<{ trashed: number[]; failed: number[] }> {
  if (!Array.isArray(ids) || ids.length === 0)
    throw new Error("cleanup:apply 只接受非空的照片 id 数组");
  const ph = ids.map(() => "?").join(",");
  const rows = db.prepare(`SELECT id, path FROM photos WHERE id IN (${ph})`)
    .all(...ids) as Array<{ id: number; path: string }>;
  const { shell } = await import("electron");
  const trashed: number[] = [];
  const failed: number[] = [];
  for (const r of rows) {
    try {
      await shell.trashItem(r.path);
    } catch (e) {
      console.warn("[cleanup] 回收站操作失败", r.path, e);
      failed.push(r.id);
      continue;
    }
    try {
      purgePhotoRows(db, r.id);
      trashed.push(r.id);
    } catch (e) {
      // 文件已经进回收站，但索引没清干净 —— 单独计数，避免幽灵照片
      console.error("[cleanup] 索引清除失败（文件已进回收站）", r.path, e);
      failed.push(r.id);
    }
  }
  return { trashed, failed };
}
