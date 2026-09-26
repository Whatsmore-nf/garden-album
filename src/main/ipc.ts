import { ipcMain } from "electron";
import type { DB } from "./db/schema";
import type { VlmService } from "./inference/vlm";
import type { IngestPipeline } from "./ingest/pipeline";
import type { Vocab } from "./ingest/tagging";
import { search, lastSeen, RELATION_WHITELIST } from "./query/engine";
import { createSmartAlbum, listSmartAlbums, openSmartAlbum } from "./albums";
import { findDuplicates, findLowQuality, applyConfirmedByIds } from "./cleanup";
export interface IpcDeps {
  db: DB;
  vlm: VlmService;
  pipeline: IngestPipeline;
  vocab: Vocab;
  thumbOf: (id: number) => Promise<string | null>;
}
export function registerIpc(deps: IpcDeps): void {
  const { db, vlm, pipeline, vocab, thumbOf } = deps;
  ipcMain.handle("search", (_e, q: unknown) =>
    search(typeof q === "string" ? q : "", db, vlm, vocab, thumbOf));
  ipcMain.handle("lastSeen", (_e, rel: unknown) =>
    lastSeen(db, typeof rel === "string" && (RELATION_WHITELIST as readonly string[]).includes(rel) ? rel : ""));
  ipcMain.handle("stats", () => {
    const c = (sql: string): number => (db.prepare(sql).get() as { c?: number })?.c ?? 0;
    return {
      photos:  c(`SELECT COUNT(*) c FROM photos`),
      tagged:  c(`SELECT COUNT(*) c FROM photos WHERE stage = 1`),
      pending: c(`SELECT COUNT(*) c FROM photos WHERE stage = 0`),
      namedPersons: c(`SELECT COUNT(*) c FROM persons WHERE name IS NOT NULL`),
    };
  });
  ipcMain.handle("album:create", (_e, name: unknown, q: unknown) =>
    createSmartAlbum(db, vlm, vocab,
      String(name ?? "未命名相册").slice(0, 60), String(q ?? "")));
  ipcMain.handle("album:list", () => listSmartAlbums(db));
  ipcMain.handle("album:open", (_e, id: unknown) => openSmartAlbum(db, Number(id), vocab));
  ipcMain.handle("album:delete", (_e, id: unknown) => {
    db.prepare(`DELETE FROM smart_albums WHERE id = ?`).run(Number(id));
    return { deleted: true };
  });
  ipcMain.handle("persons:list", () =>
    db.prepare(`
      SELECT p.id, p.name, p.relation, COUNT(DISTINCT pf.photo_id) AS photos
      FROM persons p LEFT JOIN photo_faces pf ON pf.person_id = p.id
      GROUP BY p.id ORDER BY photos DESC
    `).all());
  ipcMain.handle("persons:update", (_e, id: unknown, patch: unknown) => {
    const pid = Number(id);
    if (!Number.isInteger(pid)) throw new Error("persons:update 需要整数 id");
    const p = (patch ?? {}) as { name?: unknown; relation?: unknown };
    let name: string | undefined;
    if (p.name !== undefined) name = String(p.name).slice(0, 40);
    let relation: string | null | undefined;
    if (p.relation !== undefined) {
      relation = p.relation === null ? null : String(p.relation);
      if (relation !== null && !(RELATION_WHITELIST as readonly string[]).includes(relation))
        throw new Error(`relation 必须是 ${RELATION_WHITELIST.join("/")} 或 null`);
    }
    if (name === undefined && relation === undefined) return { updated: 0 };
    const cur = db.prepare(`SELECT name, relation FROM persons WHERE id = ?`).get(pid) as
      { name: string | null; relation: string | null } | undefined;
    if (!cur) throw new Error("person 不存在");
    db.prepare(`UPDATE persons SET name = ?, relation = ? WHERE id = ?`)
      .run(name !== undefined ? name : cur.name,
           relation !== undefined ? relation : cur.relation, pid);
    return { updated: 1 };
  });
  ipcMain.handle("ingest:scan", (_e, root: unknown) =>
    pipeline.scan(typeof root === "string" ? root : ""));
  ipcMain.handle("ingest:watch", (_e, root: unknown) => {
    pipeline.startWatching(typeof root === "string" ? root : "");
    return { watching: true };
  });
  ipcMain.handle("cleanup:suggest", () => [...findDuplicates(db), ...findLowQuality(db)]);
  ipcMain.handle("cleanup:apply", (_e, ids: unknown) =>
    applyConfirmedByIds(db,
      Array.isArray(ids) ? ids.map(Number).filter(Number.isInteger) : []));
}
