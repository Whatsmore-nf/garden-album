import type { DB } from "../db/schema";
import type { VlmService } from "../inference/vlm";
export interface VocabEntry { id: number; slug: string; zh: string; aliases: string[]; category: string }
export type Vocab = VocabEntry[];
export interface TagResult {
  tags: Array<{ slug: string; confidence: number }>;
  caption: string;
  quality: number;
  aesthetic: number;
  screenshot: number;
  ocr: string;
  event: string;
}
const TAG_MIN_CONFIDENCE = 0.5;
const MAX_TAGS = 12;
function vocabPrompt(vocab: Vocab): string {
  const byCat = new Map<string, string[]>();
  for (const t of vocab) {
    const line = `${t.slug}=${t.zh}${t.aliases.length ? `(${t.aliases.join("/")})` : ""}`;
    const arr = byCat.get(t.category) ?? (byCat.set(t.category, []).get(t.category)!);
    arr.push(line);
  }
  return [...byCat].map(([c, lines]) => `[${c}] ${lines.join(", ")}`).join("\n");
}
const clamp01 = (x: unknown, d = 0): number => {
  const n = Number(x);
  return Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : d;
};
export async function tagPhoto(
  vlm: VlmService, vocab: Vocab, thumbB64: string
): Promise<TagResult> {
  const system = `你是本地相册的照片索引器。严格遵守：
1. 只能使用<词表>中列出的 slug，禁止发明新标签。
2. 最多选 ${MAX_TAGS} 个标签；confidence 取 0.5~1.0，低于 0.5 的不要选。
3. OCR：提取图中可见文字（发票、票据、屏幕内容尤其重要），无文字输出空串。
4. quality=清晰程度，aesthetic=构图美感，screenshot=是截图的概率，均 0~1。
5. event：所属事件名（如"三亚旅行"），看不出来留空。
6. 只输出一个 JSON 对象，不要任何其他文字：
{"tags":[{"slug":"...","confidence":0.9}],"caption":"中文一句话描述","quality":0.0,"aesthetic":0.0,"screenshot":0.0,"ocr":"","event":""}
<词表>
${vocabPrompt(vocab)}`;
  const raw = await vlm.chatJsonSerial<TagResult>([
    { role: "system", content: system },
    { role: "user", content: [
      { type: "image_url", image_url: { url: `data:image/jpeg;base64,${thumbB64}` } },
      { type: "text", text: "标注这张图片。" },
    ]},
  ], "background");
  const legal = new Set(vocab.map(v => v.slug));
  const rawTags = Array.isArray(raw?.tags) ? (raw.tags as any[]) : [];
  // ★ 按 slug 去重：photo_tags 主键是 (photo_id, tag_id)，模型重复输出同一 slug
  //   会导致 INSERT 抛 constraint → 整个事务回滚 → 该照片永久 stage=0。必须在这里挡住。
  const seen = new Set<string>();
  const tags: Array<{ slug: string; confidence: number }> = [];
  for (const t of rawTags) {
    if (!t || typeof t.slug !== "string" || !legal.has(t.slug)) continue;
    if (!Number.isFinite(Number(t?.confidence)) || Number(t.confidence) < TAG_MIN_CONFIDENCE) continue;
    if (seen.has(t.slug)) continue;
    seen.add(t.slug);
    tags.push({ slug: t.slug as string, confidence: Math.min(1, Number(t.confidence)) });
    if (tags.length >= MAX_TAGS) break;
  }
  return {
    tags,
    caption:   String(raw?.caption ?? "").slice(0, 300),
    quality:   clamp01(raw?.quality),
    aesthetic: clamp01(raw?.aesthetic),
    screenshot: clamp01(raw?.screenshot),
    ocr:       String(raw?.ocr ?? "").slice(0, 5000),
    event:     String(raw?.event ?? "").slice(0, 60),
  };
}
export function storeTagResult(db: DB, photoId: number, vocab: Vocab, r: TagResult): void {
  const slugToId = new Map(vocab.map(v => [v.slug, v.id]));
  const upPhoto = db.prepare(`
    UPDATE photos SET ai_caption=?, ai_quality=?, ai_aesthetic=?, ai_screenshot=?,
                      ocr_text=?, ai_event_name=?, stage=1
    WHERE id=?`);
  const delTags = db.prepare(`DELETE FROM photo_tags WHERE photo_id=?`);
  const delFts  = db.prepare(`DELETE FROM fts_tags WHERE photo_id=?`);
  const delOcr  = db.prepare(`DELETE FROM fts_ocr WHERE photo_id=?`);
  const insTag  = db.prepare(`
    INSERT INTO photo_tags (photo_id, tag_id, confidence, source) VALUES (?,?,?,'vlm')`);
  const insFts  = db.prepare(`INSERT INTO fts_tags (photo_id, slugs) VALUES (?,?)`);
  const insOcr  = db.prepare(`INSERT INTO fts_ocr (photo_id, text) VALUES (?,?)`);
  db.transaction(() => {
    upPhoto.run(r.caption, r.quality, r.aesthetic, r.screenshot, r.ocr, r.event, photoId);
    delTags.run(photoId); delFts.run(photoId); delOcr.run(photoId);
    const slugs: string[] = [];
    for (const t of r.tags) {
      const tid = slugToId.get(t.slug);
      if (tid == null) continue;
      insTag.run(photoId, tid, t.confidence);
      slugs.push(t.slug);
    }
    insFts.run(photoId, slugs.join(" "));
    if (r.ocr.trim()) insOcr.run(photoId, r.ocr.trim());
  })();
}
