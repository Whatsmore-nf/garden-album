import type { DB } from "./schema";
import type { Vocab } from "../ingest/tagging";
/* 示例词表（生产 500–800 条）。
 * 运营要点：alias 覆盖度直接决定“多少查询不必惊动模型”。
 * 强烈建议加一张 query_log 记录未命中的 residual，定期把高频短语补进 alias——
 * 这是这个系统最重要的长期运营动作。
 */
const VOCAB: Array<{ slug: string; zh: string; aliases?: string[]; category: string }> = [
  // scene
  { slug: "beach",     zh: "海边",   aliases: ["海滩", "沙滩"], category: "scene" },
  { slug: "mountain",  zh: "山",     aliases: ["雪山", "爬山"], category: "scene" },
  { slug: "sea",       zh: "海",     category: "scene" },
  { slug: "city",      zh: "城市",   aliases: ["街头", "街拍"], category: "scene" },
  { slug: "night",     zh: "夜晚",   aliases: ["夜里", "晚上"], category: "scene" },
  { slug: "sunset",    zh: "日落",   aliases: ["夕阳"], category: "scene" },
  { slug: "snow",      zh: "雪",     category: "scene" },
  { slug: "home",      zh: "家里",   aliases: ["室内", "房间"], category: "scene" },
  // object / document
  { slug: "food",      zh: "食物",   aliases: ["美食", "吃的"], category: "object" },
  { slug: "car",       zh: "汽车",   aliases: ["车"], category: "object" },
  { slug: "invoice",   zh: "发票",   category: "document" },
  { slug: "receipt",   zh: "收据",   aliases: ["小票"], category: "document" },
  { slug: "screenshot", zh: "截图",  aliases: ["截屏"], category: "document" },
  { slug: "whiteboard", zh: "白板",  category: "document" },
  { slug: "id_card",   zh: "证件",   aliases: ["身份证", "护照"], category: "document" },
  // color / clothing
  { slug: "red_dress", zh: "红裙",   aliases: ["红色裙子"], category: "clothing" },
  { slug: "swimsuit",  zh: "泳衣",   aliases: ["泳装"], category: "clothing" },
  { slug: "wedding_dress", zh: "婚纱", category: "clothing" },
  // emotion
  { slug: "smile",     zh: "笑",     aliases: ["笑容", "微笑", "笑得自然", "开心", "高兴"], category: "emotion" },
  { slug: "crying",    zh: "哭",     aliases: ["哭泣"], category: "emotion" },
  // activity
  { slug: "travel",    zh: "旅行",   aliases: ["旅游", "出游", "度假"], category: "activity" },
  { slug: "wedding",   zh: "婚礼",   aliases: ["结婚", "婚礼现场"], category: "activity" },
  { slug: "birthday",  zh: "生日",   aliases: ["过生日", "生日聚会"], category: "activity" },
  { slug: "selfie",    zh: "自拍",   category: "activity" },
  { slug: "group_photo", zh: "合照", aliases: ["合影", "合影留念"], category: "activity" },
  { slug: "graduation", zh: "毕业",  category: "activity" },
  // people-misc
  { slug: "stranger",  zh: "路人",   aliases: ["陌生人", "行人"], category: "people" },
  { slug: "crowd",     zh: "人群",   aliases: ["拥挤", "人很多"], category: "people" },
  { slug: "baby",      zh: "宝宝",   aliases: ["婴儿", "小孩"], category: "people" },
  { slug: "pet",       zh: "宠物",   aliases: ["猫", "狗"], category: "object" },   // ← 原 people 归类错误
  // quality
  { slug: "blurry",    zh: "模糊",   aliases: ["糊了", "不清晰"], category: "quality" },
];
export function seedVocab(db: DB): void {
  const ins = db.prepare(
    `INSERT OR IGNORE INTO tags (slug, zh, aliases, category) VALUES (?, ?, ?, ?)`
  );
  db.transaction(() => {
    for (const t of VOCAB)
      ins.run(t.slug, t.zh, JSON.stringify(t.aliases ?? []), t.category);
  })();
}
export function loadVocab(db: DB): Vocab {
  return (db.prepare(`SELECT id, slug, zh, aliases, category FROM tags`).all() as Array<{
    id: number; slug: string; zh: string; aliases: string; category: string;
  }>).map(t => ({
    id: t.id, slug: t.slug, zh: t.zh,
    aliases: JSON.parse(t.aliases ?? "[]") as string[],
    category: t.category,
  }));
}
