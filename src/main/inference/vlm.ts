// llama-server 子进程管理 + 优先级串行队列。
// 用子进程 + OpenAI 兼容 HTTP 的原因：node-llama-cpp 对 MiniCPM-V 的图像注入模板支持不全，
// llama-server 官方支持且模型天然常驻；Electron 管子进程生命周期也更干净。
import { spawn, type ChildProcess } from "node:child_process";
import net from "node:net";
export type Priority = "interactive" | "background";
interface Task {
  fn: () => Promise<unknown>;
  resolve: (v: unknown) => void;
  reject: (e: unknown) => void;
  prio: number; // 0 = interactive（搜索/精排/查询展开），1 = background（索引）
}
const delay = (ms: number) => new Promise<void>(r => setTimeout(r, ms));
export class VlmService {
  private proc: ChildProcess | null = null;
  private port = 0;
  private bin = "";
  private modelGguf = "";
  private mmprojGguf = "";
  private pending: Task[] = [];
  private draining = false;
  private lastRestart = 0;
  private startPromise: Promise<void> | null = null; // 启动互斥：防止“启动中 + 懒重启”双拉起
  private gen = 0;                                   // 代际计数：stop() 打断启动流程
  // ---------------- 生命周期 ----------------
  private static freePort(): Promise<number> {
    return new Promise((res, rej) => {
      const s = net.createServer();
      s.once("error", rej);
      s.listen(0, "127.0.0.1", () => {
        const p = (s.address() as net.AddressInfo).port;
        s.close(() => res(p));
      });
    });
  }
  private async probe(): Promise<boolean> {
    try { return (await fetch(`http://127.0.0.1:${this.port}/health`)).ok; }
    catch { return false; }
  }
  start(bin: string, modelGguf: string, mmprojGguf: string): Promise<void> {
    if (this.startPromise) return this.startPromise;
    this.bin = bin; this.modelGguf = modelGguf; this.mmprojGguf = mmprojGguf;
    this.startPromise = this.doStart().finally(() => { this.startPromise = null; });
    return this.startPromise;
  }
  private async doStart(): Promise<void> {
    const g = this.gen;                                     // 快照：stop() 会 ++gen
    this.port = await VlmService.freePort();
    if (g !== this.gen) return;                             // stop 已发生，放弃本次启动
    const common = [
      "--model", this.modelGguf, "--mmproj", this.mmprojGguf,
      "--port", String(this.port), "--ctx-size", "8192", "--no-webui",
      "--jinja",                                            // response_format 依赖 jinja 模板
    ];
    let proc = await this.trySpawn(common, 99);
    if (g !== this.gen) { try { proc.kill(); } catch {} return; }
    if (!(await this.waitHealthy(proc, 30_000))) {
      proc.kill();
      if (g !== this.gen) return;
      proc = await this.trySpawn(common, 0);
      if (g !== this.gen) { try { proc.kill(); } catch {} return; }
      if (!(await this.waitHealthy(proc, 120_000))) {
        proc.kill();
        throw new Error("llama-server 启动失败（GPU 与 CPU 均未就绪）");
      }
    }
    if (g !== this.gen) { try { proc.kill(); } catch {} return; }  // stop 在等待期间发生
    this.adopt(proc);
  }
  private trySpawn(common: string[], gpuLayers: number): Promise<ChildProcess> {
    return new Promise((res, rej) => {
      const p = spawn(this.bin, [...common, "--n-gpu-layers", String(gpuLayers)], { stdio: "ignore" });
      p.once("error", rej);
      p.once("spawn", () => res(p));
    });
  }
  private async waitHealthy(proc: ChildProcess, timeoutMs: number): Promise<boolean> {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      if (proc.exitCode !== null) return false;
      if (await this.probe()) return true;
      await delay(400);
    }
    return false;
  }
  private adopt(proc: ChildProcess): void {
    this.proc = proc;
    proc.once("exit", (code) => {
      if (this.proc === proc) this.proc = null;
      console.error(`[vlm] llama-server 退出 code=${code}`);
    });
  }
  private async ensureRunning(): Promise<void> {
    if (this.proc && this.proc.exitCode === null) return;
    if (Date.now() - this.lastRestart < 30_000)
      throw new Error("llama-server 刚刚重启失败，30 秒内不再自动重启");
    try {
      await this.start(this.bin, this.modelGguf, this.mmprojGguf);
    } catch (e) {
      this.lastRestart = Date.now();
      throw e;
    }
  }
  /** 优雅关停：SIGTERM → 3s → SIGKILL，保证不留僵尸。
   *  gen++ 会让 in-flight 的 doStart() 在 adopt 之前自杀，避免“启动中退出”漏杀子进程。 */
  async stop(): Promise<void> {
    this.gen++;
    this.startPromise = null;
    const proc = this.proc;
    this.proc = null;
    if (!proc || proc.exitCode !== null) return;
    proc.removeAllListeners("exit");
    const exited = new Promise<void>(r => proc.once("exit", () => r()));
    try { proc.kill("SIGTERM"); } catch { /* ignore */ }
    await Promise.race([exited, delay(3000)]);
    if (proc.exitCode === null) { try { proc.kill("SIGKILL"); } catch { /* ignore */ } }
    await exited;
  }
  /** process.on("exit") 只允许同步代码——最后兜底 */
  killSync(): void { this.gen++; try { this.proc?.kill(); } catch { /* ignore */ } }
  // ---------------- 推理（OpenAI 兼容 HTTP）----------------
  private async chat(body: object): Promise<string> {
    await this.ensureRunning();
    const res = await fetch(`http://127.0.0.1:${this.port}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...body, temperature: 0.1, cache_prompt: true }),
    });
    if (!res.ok) throw new Error(`vlm ${res.status}: ${(await res.text()).slice(0, 2000)}`);
    const json = await res.json() as any;
    return json.choices[0].message.content as string;
  }
  // ---------------- 优先级串行队列 ----------------
  // 注意：run() 的回调 fn 内部**不得再调用 vlm.run()**——drain() 是单线程串行器，
  // 内层任务会被外层阻塞永远得不到调度。当前所有调用点均满足该契约，未来改动要守住。
  run<T>(fn: () => Promise<T>, prio: Priority = "interactive"): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const task: Task = { fn, resolve: resolve as (v: unknown) => void, reject,
                           prio: prio === "interactive" ? 0 : 1 };
      if (task.prio === 0) {
        const i = this.pending.findIndex(t => t.prio > 0);
        if (i < 0) this.pending.push(task); else this.pending.splice(i, 0, task);
      } else {
        this.pending.push(task);
      }
      void this.drain();
    });
  }
  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.pending.length) {
        const t = this.pending.shift()!;
        try { t.resolve(await t.fn()); } catch (e) { t.reject(e); }
      }
    } finally { this.draining = false; }
  }
  chatJsonSerial<T>(messages: object[], prio: Priority = "interactive"): Promise<T> {
    return this.run(async () => {
      const text = await this.chat({ messages, response_format: { type: "json_object" } });
      const m = text.match(/\{[\s\S]*\}/);
      if (!m) throw new Error("非 JSON 输出: " + text.slice(0, 100));
      return JSON.parse(m[0]) as T;
    }, prio);
  }
  expandQueryText(system: string, user: string): Promise<unknown> {
    return this.chatJsonSerial(
      [{ role: "system", content: system }, { role: "user", content: user }],
      "interactive",
    );
  }
}
