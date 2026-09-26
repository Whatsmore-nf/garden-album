// 检测：YuNet（opencv_zoo face_detection_yunet_2023mar，~350KB）
// 识别：SFace（opencv_zoo face_recognition_sface_2021dec，112×112 输入 / 512d 输出）
//
// ★ 通道顺序（必须实测确认）：
//   - YuNet 走 cv2.FaceDetectorYN，输入是 BGR Mat → 网络期望 BGR → 我们拿 sharp 的 RGB 需要交换
//   - SFace 的官方 demo 在进网络前 cvtColor(BGR2RGB) → 网络期望 RGB → sharp 的 RGB 直入即可
//   两模型约定不同，必须用独立开关；SIM_THR 也必须用同人/异人 20+ 对做实测校准，
//   通道顺序错的症状是**向量退化成近随机**：不报错、不崩溃，只是聚类全是噪声。
const YUNET_SWAP_RB = true;
const SFACE_SWAP_RB = false;
const DET_SIZE    = 320;
const SCORE_THR   = 0.6;
const NMS_THR     = 0.3;
const SIM_THR     = 0.55;
const KNN_K       = 64;    // 先大范围召回，JOIN 过滤后再取前 16 投票（见 assign 注释）
const VOTE_TOPK   = 16;
const VOTE_RATIO  = 0.5;
const EXEMPLAR_CAP = 32;
const STRIDES   = [8, 16, 32];
const MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96]];
const VARIANCE  = [0.1, 0.2];
export interface FaceBox { x: number; y: number; w: number; h: number }
export interface FaceResult { bbox: FaceBox; emb: Float32Array }
interface Prior { cx: number; cy: number; w: number; h: number }
function toNCHW(px: Buffer, w: number, h: number, swapRb: boolean): Float32Array {
  const n = w * h;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const r = px[3 * i], g = px[3 * i + 1], b = px[3 * i + 2];
    out[i]         = ((swapRb ? b : r) - 127.5) / 128;
    out[n + i]     = (g - 127.5) / 128;
    out[2 * n + i] = ((swapRb ? r : b) - 127.5) / 128;
  }
  return out;
}
function iou(a: number[], b: number[]): number {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const uni = a[2] * a[3] + b[2] * b[3] - inter;
  return uni > 0 ? inter / uni : 0;
}
function nms(cands: Array<{ box: number[]; score: number }>, thr: number) {
  const keep: typeof cands = [];
  for (const c of [...cands].sort((x, y) => y.score - x.score))
    if (keep.every(k => iou(k.box, c.box) <= thr)) keep.push(c);
  return keep;
}
export class FaceService {
  private det!: ort.InferenceSession;
  private rec!: ort.InferenceSession;
  private priors: Prior[] | null = null;
  private ready = false;                       // ★ init 失败时 process 返回 []，绝不拖垮 VLM 标注
  async init(modelsDir: string): Promise<void> {
    this.det = await ort.InferenceSession.create(`${modelsDir}/yunet.onnx`);
    this.rec = await ort.InferenceSession.create(`${modelsDir}/sface.onnx`);
    this.ready = true;
  }
  async process(thumb: Buffer): Promise<FaceResult[]> {
    if (!this.ready) return [];                // ★ 关键：init 未成功时静默降级，人脸丢失但不阻塞语义
    const boxes = await this.detect(thumb);
    const out: FaceResult[] = [];
    for (const b of boxes) out.push({ bbox: b, emb: await this.embed(thumb, b) });
    return out;
  }
  private async detect(img: Buffer): Promise<FaceBox[]> {
    const meta = await sharp(img).metadata();
    const iw = meta.width ?? 0, ih = meta.height ?? 0;
    if (!iw || !ih) return [];
    const scale = Math.min(DET_SIZE / iw, DET_SIZE / ih);
    const rw = Math.max(1, Math.round(iw * scale));
    const rh = Math.max(1, Math.round(ih * scale));
    const ox = Math.floor((DET_SIZE - rw) / 2), oy = Math.floor((DET_SIZE - rh) / 2);
    const resized = await sharp(img).resize(rw, rh).removeAlpha().raw().toBuffer();
    const canvas = Buffer.alloc(3 * DET_SIZE * DET_SIZE, 0);
    for (let y = 0; y < rh; y++)
      resized.copy(canvas, 3 * ((y + oy) * DET_SIZE + ox), 3 * y * rw, 3 * rw);
    const input = new ort.Tensor("float32",
      toNCHW(canvas, DET_SIZE, DET_SIZE, YUNET_SWAP_RB), [1, 3, DET_SIZE, DET_SIZE]);
    const out = await this.det.run({ [this.det.inputNames[0]]: input });
    return this.decodeYunet(out)
      .map(({ box }) => ({
        x: (box[0] - ox) / scale, y: (box[1] - oy) / scale,
        w: box[2] / scale,      h: box[3] / scale,
      }))
      .filter(b => b.w > 4 && b.h > 4 && b.x + b.w > 0 && b.y + b.h > 0 && b.x < iw && b.y < ih);
  }
  private genPriors(): Prior[] {
    if (this.priors) return this.priors;
    const priors: Prior[] = [];
    for (let s = 0; s < STRIDES.length; s++) {
      const stride = STRIDES[s];
      const rows = Math.floor(DET_SIZE / stride), cols = Math.floor(DET_SIZE / stride);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
        for (const m of MIN_SIZES[s])
          priors.push({
            cx: ((c + 0.5) * stride) / DET_SIZE,
            cy: ((r + 0.5) * stride) / DET_SIZE,
            w: m / DET_SIZE, h: m / DET_SIZE,
          });
    }
    this.priors = priors;
    return priors;
  }
  private decodeYunet(out: ort.InferenceResult): Array<{ box: number[]; score: number }> {
    const rec = out as unknown as Record<string, ort.Tensor>;
    let loc = rec.loc ?? rec.boxes;
    let conf = rec.conf ?? rec.scores;
    if (!loc || !conf) {
      for (const t of Object.values(rec)) {
        const last = t.dims[t.dims.length - 1];
        if (!loc && (last === 14 || last === 4)) loc = t;
        else if (!conf && last === 2) conf = t;
      }
    }
    if (!loc || !conf) throw new Error("无法识别 YuNet 输出张量（期望 loc[1,N,14] / conf[1,N,2]）");
    const locD  = loc.data  as Float32Array;
    const confD = conf.data as Float32Array;
    const C = loc.dims[loc.dims.length - 1];
    const N = Math.min(loc.dims[1] ?? 0, conf.dims[1] ?? 0);
    const priors = this.genPriors();
    const n = Math.min(N, priors.length);
    const preActivated = confD.length >= 2 &&
      confD[0] >= 0 && confD[0] <= 1 && confD[1] >= 0 && confD[1] <= 1 &&
      Math.abs(confD[0] + confD[1] - 1) < 1e-3;
    const cands: Array<{ box: number[]; score: number }> = [];
    for (let i = 0; i < n; i++) {
      const bg = confD[2 * i], fg = confD[2 * i + 1];
      let score: number;
      if (preActivated) {
        score = fg;
      } else {
        const m = Math.max(bg, fg);
        score = Math.exp(fg - m) / (Math.exp(bg - m) + Math.exp(fg - m));
      }
      if (score < SCORE_THR) continue;
      const p = priors[i], o = i * C;
      const cx = (p.cx + locD[o]     * VARIANCE[0] * p.w) * DET_SIZE;
      const cy = (p.cy + locD[o + 1] * VARIANCE[0] * p.h) * DET_SIZE;
      const w  = p.w * Math.exp(locD[o + 2] * VARIANCE[1]) * DET_SIZE;
      const h  = p.h * Math.exp(locD[o + 3] * VARIANCE[1]) * DET_SIZE;
      cands.push({ box: [cx - w / 2, cy - h / 2, w, h], score });
    }
    return nms(cands, NMS_THR);
  }
  private async embed(img: Buffer, b: FaceBox): Promise<Float32Array> {
    const meta = await sharp(img).metadata();
    const iw = meta.width ?? 0, ih = meta.height ?? 0;
    if (!iw || !ih) throw new Error("embed: 无效图像");
    let side = Math.max(8, Math.max(b.w, b.h) * 1.3);
    side = Math.min(side, Math.max(iw, ih));
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    let left = Math.round(cx - side / 2), top = Math.round(cy - side / 2);
    left = Math.max(0, Math.min(left, iw - 1));
    top  = Math.max(0, Math.min(top,  ih - 1));
    const sw = Math.min(Math.round(side), iw - left);
    const sh = Math.min(Math.round(side), ih - top);
    if (sw < 8 || sh < 8) throw new Error("embed: 裁剪区域过小");
    const px = await sharp(img).extract({ left, top, width: sw, height: sh })
      .resize(112, 112, { fit: "fill" }).removeAlpha().raw().toBuffer();
    const t = new ort.Tensor("float32",
      toNCHW(px, 112, 112, SFACE_SWAP_RB), [1, 3, 112, 112]);
    const outs = await this.rec.run({ [this.rec.inputNames[0]]: t });
    // 按末维形状选 512 维输出；退化时取第一个张量
    let v: Float32Array | null = null;
    for (const o of Object.values(outs)) {
      if (o.dims[o.dims.length - 1] === 512) {
        v = Float32Array.from(o.data as ArrayLike<number>);
        break;
      }
    }
    if (!v) v = Float32Array.from(Object.values(outs)[0].data as ArrayLike<number>);
    let norm = 0;
    for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= norm;
    return v;
  }
  /** KNN 投票认领。原实现 k=16 且 JOIN 后才过滤——一张多人合影插进去，16 个近邻里
   *  可能大部分是未命名脸 → 真正该认领的人票数不够 → 永远认不出来。
   *  改：k=64 大范围召回，JOIN 过滤到已命名后取前 16 条投票。 */
  assign(db: DB, emb: Float32Array): number | null {
    const blob = Buffer.from(emb.buffer, emb.byteOffset, emb.byteLength);
    const rows = db.prepare(`
      SELECT pf.person_id AS pid, v.distance AS dist
      FROM (SELECT face_rowid, distance FROM vec_faces
            WHERE embedding MATCH ? AND k = ${KNN_K}) v
      JOIN photo_faces pf ON pf.face_rowid = v.face_rowid
      WHERE pf.person_id IS NOT NULL
      ORDER BY v.distance ASC
      LIMIT ${VOTE_TOPK}
    `).all(blob) as Array<{ pid: number; dist: number }>;
    const votes = new Map<number, number>();
    let total = 0, topPid: number | null = null, topSim = -1;
    for (const r of rows) {
      const sim = 1 - (r.dist * r.dist) / 2;
      if (sim < SIM_THR) continue;
      votes.set(r.pid, (votes.get(r.pid) ?? 0) + 1);
      total++;
      if (sim > topSim) { topSim = sim; topPid = r.pid; }
    }
    if (topPid != null && total > 0 && (votes.get(topPid)! / total) >= VOTE_RATIO) return topPid;
    return null;
  }
  createPerson(db: DB): number {
    return Number(db.prepare(`INSERT INTO persons DEFAULT VALUES`).run().lastInsertRowid);
  }
  capExemplars(db: DB, personId: number): void {
    const extra = db.prepare(`
      SELECT v.face_rowid AS rowid FROM vec_faces v
      JOIN photo_faces pf ON pf.face_rowid = v.face_rowid
      WHERE pf.person_id = ? ORDER BY v.face_rowid DESC LIMIT -1 OFFSET ?
    `).all(personId, EXEMPLAR_CAP) as Array<{ rowid: number }>;
    if (extra.length === 0) return;
    db.transaction(() => {
      const del = db.prepare(`DELETE FROM vec_faces WHERE face_rowid = ?`);
      for (const e of extra) del.run(e.rowid);
    })();
  }
}
import ort from "onnxruntime-node";
import sharp from "sharp";
import type { DB } from "../db/schema";
