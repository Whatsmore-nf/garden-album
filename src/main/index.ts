import { app, BrowserWindow } from "electron";
import path from "node:path";
import sharp from "sharp";
import { openDb } from "./db/schema";
import { seedVocab, loadVocab } from "./db/seed-vocab";
import { VlmService } from "./inference/vlm";
import { FaceService } from "./ingest/faces";
import { IngestPipeline } from "./ingest/pipeline";
import { registerIpc } from "./ipc";
import { bus } from "./bus";
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  let vlmRef: VlmService | null = null;
  let pipelineRef: IngestPipeline | null = null;
  process.on("exit", () => vlmRef?.killSync());
  app.on("second-instance", () => {
    BrowserWindow.getAllWindows().forEach(w => w.focus());
  });
  app.whenReady().then(async () => {
    const db = openDb(path.join(app.getPath("userData"), "gallery.db"));
    seedVocab(db);
    const vocab = loadVocab(db);
    const RES = app.isPackaged
      ? process.resourcesPath
      : path.join(app.getAppPath(), "resources");
    const vlm = new VlmService();
    vlmRef = vlm;
    void vlm.start(
      path.join(RES, "bin/llama-server"),
      path.join(RES, "models/minicpm-v-4.6-Q4_K_M.gguf"),
      path.join(RES, "models/minicpm-v-mmproj.gguf"),
    ).catch(err => console.error("[vlm] 初始启动失败（首次使用时会重试）:", err));
    const faces = new FaceService();
    // ★ init 失败时 FaceService 内部 ready 标志保持 false，process() 返回 []，
    //   人脸路径静默降级，绝不拖垮 VLM 语义标注
    await faces.init(path.join(RES, "models"))
      .catch(err => console.error("[faces] 初始化失败（人脸功能将不可用，语义索引不受影响）:", err));
    const pipeline = new IngestPipeline(db, vlm, faces, vocab);
    pipelineRef = pipeline;
    const thumbOf = async (id: number): Promise<string | null> => {
      const row = db.prepare(`SELECT path FROM photos WHERE id = ?`).get(id) as
        { path: string } | undefined;
      if (!row) return null;
      try {
        return (await sharp(row.path)
          .resize(384, 384, { fit: "inside" })
          .jpeg({ quality: 70 }).toBuffer()).toString("base64");
      } catch { return null; }
    };
    registerIpc({ db, vlm, pipeline, vocab, thumbOf });
    bus.on("ingest-progress", (d) => {
      BrowserWindow.getAllWindows().forEach(w => w.webContents.send("ingest:progress", d));
    });
    createWindow();
    app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
  });
  let quitting = false;
  app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
  app.on("before-quit", (e) => {
    if (quitting) return;
    quitting = true;
    e.preventDefault();
    pipelineRef?.stopWatching();
    Promise.resolve(vlmRef ? vlmRef.stop() : undefined)
      .catch(() => {})
      .finally(() => app.exit(0));
  });
  function createWindow(): void {
    new BrowserWindow({
      width: 1280, height: 800,
      webPreferences: {
        preload: path.join(__dirname, "../preload/index.js"),
        contextIsolation: true,
        sandbox: true,
      },
    }).loadFile(path.join(__dirname, "../renderer/index.html"));
  }
}
