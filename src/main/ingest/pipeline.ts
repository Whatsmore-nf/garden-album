import { readdirSync, statSync, readFileSync, existsSync, openSync, readSync, closeSync } from "node:fs";
import path from "node:path";
import sharp from "sharp";
import ExifReader from "exifreader";
import { watch, type FSWatcher } from "chokidar";
import type { DB } from "../db/schema";
import type { VlmService } from "../inference/vlm";
import type { FaceService } from "./faces";
import { phash } from "./phash";
import { tagPhoto, storeTagResult, type Vocab } from "./tagging";
import { bus } from "../bus";
import { purgePhotoByPath } from "../cleanup";
const IMG_EXT = /\.(jpe?g|png|webp|bmp|tiff?)$/i;
const MAX_TAG_ATTEMPTS = 3;
const EXIF_HEAD_BYTES = 512 * 1024;
export interface IngestProgress {
  pending: number;
  busy: boolean;
  done: number;
  current: string | null;
}
/** 只读文件头。EXIF 在 JPEG/TIFF 头部，不必要读完整 20MB 大图 */
function readHead(file: string): Buffer {
  const fd = openSync(file, "r");
  try {
    const size = Math.min(statSync(file).size, EXIF_HEAD_BYTES);
    const buf = Buffer.alloc(size);
    const n = readSync(fd, buf, 0, size, 0);
    return buf.subarray(0, n);
  } finally {
    closeSync(fd);
  }
}
export class IngestPipeline {
  private queue: string[] = [];
  private inQueue = new Set<string>();
  private busy = false;
  private watcher: FSWatcher | null = null;
  private doneCount = 0;
  private lastEmit = 0;
  constructor(
    private db: DB,
    private vlm: VlmService,
    private faces: FaceService,
    private vocab: Vocab,
  ) {}
  scan(root: string): { enqueued: number } {
    const before = this.queue.length;
    const walk = (dir: string) => {
      let entries;
      try { entries = readdirSync(dir, { withFileTypes: true }); }
      catch { return; }
      for (const e of entries) {
        const p = path.join(dir, e.name);
        if (e.isDirectory()) { walk(p); continue; }
        if (!IMG_EXT.test(e.name)) continue;
        let mtime: number;
        try { mtime = statSync(p).mtimeMs; } catch { continue; }
        const row = this.db.prepare(`SELECT mtime FROM photos WHERE path = ?`).get(p) as
          { mtime: number } | undefined;
        if (!row || row.mtime < mtime) this.enqueue(p);
      }
    };
    if (root) walk(root);
    // ★ 补跑前先校验文件还在：用户已删的文件不能白跑必败任务烧 tag_attempts
    for (const r of this.db.prepare(
      `SELECT path FROM photos WHERE stage = 0 AND tag_attempts < ?`
    ).all(MAX_TAG_ATTEMPTS) as Array<{ path: string }>) {
      if (!existsSync(r.path)) { purgePhotoByPath(this.db, r.path); continue; }
      this.enqueue(r.path);
    }
    void this.drain();
    return { enqueued: this.queue.length - before };
  }
  startWatching(root: string): void {
    this.watcher?.close();
    if (!root) return;
    this.watcher = watch(root, {
      ignored: (p: string) => /(^|[\\/])\./.test(p),
      awaitWriteFinish: { stabilityThreshold: 2000 },
      ignoreInitial: true,
    });
    this.watcher.on("add",    (p) => { if (IMG_EXT.test(p)) { this.enqueue(p); void this.drain(); } });
    this.watcher.on("change", (p) => { if (IMG_EXT.test(p)) { this.enqueue(p); void this.drain(); } });
    this.watcher.on("unlink", (p) => { if (IMG_EXT.test(p)) purgePhotoByPath(this.db, p); });
  }
  stopWatching(): void { this.watcher?.close(); this.watcher = null; }
  /** 节流 250ms；busy 状态切换用 force=true 保证最终状态一定送达 */
  private emitProgress(current: string | null = null, force = false): void {
    const now = Date.now();
    if (!force && now - this.lastEmit < 250) return;
    this.lastEmit = now;
    const d: IngestProgress = {
      pending: this.queue.length, busy: this.busy,
      done: this.doneCount, current,
    };
    bus.emit("ingest-progress", d);
  }
  private enqueue(p: string): void {
    if (this.inQueue.has(p)) return;
    this.inQueue.add(p);
    this.queue.push(p);
  }
  private async drain(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    this.emitProgress(null, true);
    while (this.queue.length) {
      const p = this.queue.shift()!;
      this.inQueue.delete(p);
      this.emitProgress(p);
      try {
        await this.processOne(p);
        this.doneCount++;
      } catch (e) {
        console.error("[ingest] 失败", p, e);
        this.db.prepare(
          `UPDATE photos SET tag_attempts = tag_attempts + 1 WHERE path = ? AND stage = 0`
        ).run(p);
      }
    }
    this.busy = false;
    this.emitProgress(null, true);
  }
  private async processOne(file: string): Promise<void> {
    const st = statSync(file);
    let takenAt = st.mtimeMs;
    try {
      const exif = ExifReader.load(readHead(file)) as any;
      const d: unknown = exif?.DateTimeOriginal?.description ?? exif?.DateTime?.description;
      if (typeof d === "string") {
        const t = new Date(d.replace(/^(\d+):(\d+):/, "$1-$2-").replace(" ", "T")).getTime();
        if (Number.isFinite(t)) takenAt = t;
      }
    } catch { /* 无 EXIF 或解析失败 */ }
    const meta = await sharp(file).metadata();
    const ph = await phash(file);
    const thumb = await sharp(file)
      .resize(768, 768, { fit: "inside", withoutEnlargement: true })
      .jpeg({ quality: 80 })
      .toBuffer();
    // ★ 补 taken_at = excluded.taken_at：用户改 EXIF 后重跑全流程要更新拍摄时间
    const row = this.db.prepare(`
      INSERT INTO photos (path, mtime, taken_at, width, height, size_bytes, phash, ingested_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(path) DO UPDATE SET
        mtime        = excluded.mtime,
        taken_at     = excluded.taken_at,
        phash        = excluded.phash,
        width        = excluded.width,
        height       = excluded.height,
        size_bytes   = excluded.size_bytes,
        stage        = CASE WHEN excluded.mtime > photos.mtime THEN 0 ELSE photos.stage END,
        tag_attempts = CASE WHEN excluded.mtime > photos.mtime THEN 0 ELSE photos.tag_attempts END
      RETURNING id
    `).get(file, st.mtimeMs, takenAt,
            meta.width ?? null, meta.height ?? null, st.size, ph, Date.now()) as { id: number };
    const photoId = row.id;
    // ★ faces.process() 在 FaceService 未 ready 时返回 []，人脸路径不阻塞语义路径
    const faceRows = await this.faces.process(thumb);
    const touched = new Set<number>();
    this.db.transaction(() => {
      this.db.prepare(`DELETE FROM vec_faces WHERE face_rowid IN
        (SELECT face_rowid FROM photo_faces WHERE photo_id = ?)`).run(photoId);
      this.db.prepare(`DELETE FROM photo_faces WHERE photo_id = ?`).run(photoId);
      const insFace = this.db.prepare(
        `INSERT INTO photo_faces (photo_id, person_id, bbox, source) VALUES (?, ?, ?, 'face-detector')`);
      const insVec = this.db.prepare(`INSERT INTO vec_faces (face_rowid, embedding) VALUES (?, ?)`);
      for (const f of faceRows) {
        const pid = this.faces.assign(this.db, f.emb) ?? this.faces.createPerson(this.db);
        touched.add(pid);
        const fr = insFace.run(photoId, pid,
          JSON.stringify([f.bbox.x, f.bbox.y, f.bbox.w, f.bbox.h]));
        insVec.run(Number(fr.lastInsertRowid),
          Buffer.from(f.emb.buffer, f.emb.byteOffset, f.emb.byteLength));
      }
    })();
    // ★ capExemplars 是索引维护，抛错不能让已成功的人脸行被计为失败
    for (const pid of touched) {
      try { this.faces.capExemplars(this.db, pid); }
      catch (e) { console.warn("[ingest] capExemplars 失败", pid, e); }
    }
    const result = await tagPhoto(this.vlm, this.vocab, thumb.toString("base64"));
    storeTagResult(this.db, photoId, this.vocab, result);
  }
}
