import type { VlmService } from "../inference/vlm";
const GROUP = 8;
const DEFAULT_SCORE = 50;
export async function rerank(
  query: string, rows: any[],
  vlm: VlmService, thumbOf: (id: number) => Promise<string | null>,
): Promise<any[]> {
  const scored: Array<{ row: any; score: number }> = [];
  for (let i = 0; i < rows.length; i += GROUP) {
    const group = rows.slice(i, i + GROUP);
    const thumbs = await Promise.all(
      group.map(r => Promise.resolve(thumbOf(r.id)).catch(() => null)));
    const usable = thumbs
      .map((t, idx) => ({ t, idx }))
      .filter(x => x.t != null) as Array<{ t: string; idx: number }>;
    const scoreMap = new Map<number, number>();
    if (usable.length > 0) {
      try {
        const out = await vlm.chatJsonSerial<{ scores?: Array<{ index?: unknown; score?: unknown }> }>([
          { role: "system", content:
            `你是搜索重排器。候选照片按【给定数组顺序】从 0 开始编号。按语义相关性打 0~100 分。
只输出 JSON：{"scores":[{"index":0,"score":87}]}，index 必须是照片在数组中的位置，且覆盖每一张。` },
          { role: "user", content: [
            { type: "text", text: `查询：${query}` },
            ...usable.map(x => ({
              type: "image_url",
              image_url: { url: `data:image/jpeg;base64,${x.t}` },
            })),
          ]},
        ], "interactive");
        for (const s of out.scores ?? []) {
          const k = Number(s?.index);
          if (Number.isInteger(k) && k >= 0 && k < usable.length && Number.isFinite(Number(s?.score)))
            scoreMap.set(usable[k].idx, Math.min(100, Math.max(0, Number(s?.score))));
        }
      } catch (e) {
        console.warn("[rerank] 一组评分失败，整组保留默认分", e);
      }
    }
    group.forEach((row, idx) => scored.push({ row, score: scoreMap.get(idx) ?? DEFAULT_SCORE }));
  }
  return scored
    .map((s, i) => ({ ...s.row, _score: s.score, _i: i }))
    .sort((a, b) => (b._score - a._score) || (a._i - b._i))
    .map(({ _score, _i, ...rest }) => ({ ...rest, _rerank: _score }));
}
