import sharp from "sharp";
/** 32×32 灰度 → 2D DCT → 8×8 低频 → 64bit pHash（16 位 hex） */
const N = 32, M = 8;
// cos(u, n) 预表：u∈[0,8) n∈[0,32)，2×256 次浮点乘除。10 万张图省几分钟 CPU
const COS: Float64Array = (() => {
  const t = new Float64Array(M * N);
  for (let u = 0; u < M; u++) for (let n = 0; n < N; n++)
    t[u * N + n] = Math.cos((2 * n + 1) * u * Math.PI / (2 * N));
  return t;
})();
export async function phash(file: string): Promise<string> {
  const { data } = await sharp(file)
    .removeAlpha().greyscale().resize(N, N, { fit: "fill" })
    .raw().toBuffer({ resolveWithObject: true });
  const coef: number[] = [];
  for (let u = 0; u < M; u++) for (let v = 0; v < M; v++) {
    let s = 0;
    for (let x = 0; x < N; x++) {
      const cu = COS[u * N + x];
      for (let y = 0; y < N; y++)
        s += data[x * N + y] * cu * COS[v * N + y];
    }
    coef.push(s);
  }
  const mean = coef.slice(1).reduce((a, b) => a + b, 0) / (coef.length - 1);
  let bits = 0n;
  for (let i = 0; i < 64; i++) if (coef[i] > mean) bits |= 1n << BigInt(63 - i);
  return bits.toString(16).padStart(16, "0");
}
export function hammingHex(a: string, b: string): number {
  let x = BigInt("0x" + a) ^ BigInt("0x" + b), d = 0;
  while (x) { d += Number(x & 1n); x >>= 1n; }
  return d;
}
