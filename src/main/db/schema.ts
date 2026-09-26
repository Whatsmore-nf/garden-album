import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
/** better-sqlite3 实例类型别名，全项目统一使用 */
export type DB = import("better-sqlite3").Database;
export function openDb(path: string): DB {
  const db = new Database(path);
  db.pragma("journal_mode = WAL");
  db.pragma("synchronous = NORMAL");
  db.pragma("foreign_keys = ON");   // photo_tags / photo_faces 的 ON DELETE CASCADE 依赖此开关
  sqliteVec.load(db);               // vec0 虚拟表
  db.exec(`
    -- ============ 照片主表：事实字段与 AI 推测字段严格分离 ============
    CREATE TABLE IF NOT EXISTS photos (
      id            INTEGER PRIMARY KEY,
      path          TEXT UNIQUE NOT NULL,
      mtime         INTEGER NOT NULL,            -- 文件变更检测
      stage         INTEGER NOT NULL DEFAULT 0,  -- 0=仅元数据(冷启动即可用) 1=AI标注完成
      tag_attempts  INTEGER NOT NULL DEFAULT 0,  -- 标注失败计数，>=3 停止自动重试
      -- ↓ 事实字段（EXIF / 文件系统）
      taken_at      INTEGER,
      width         INTEGER, height INTEGER,
      size_bytes    INTEGER,
      -- ↓ AI 推测字段：每个都带置信度/来源，UI 必须区别呈现，防“伪造记忆”
      ai_caption    TEXT,
      ai_quality    REAL,        -- 清晰度 0-1
      ai_aesthetic  REAL,        -- 美学 0-1
      ai_screenshot REAL,        -- 是截图的概率
      ai_event_name TEXT,
      ocr_text      TEXT,
      phash         TEXT,        -- 16位hex，去重
      ingested_at   INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_photos_time  ON photos(taken_at);
    CREATE INDEX IF NOT EXISTS idx_photos_stage ON photos(stage) WHERE stage = 0;
    -- ============ 受控词表（生产 500-800 条，见 seed-vocab.ts） ============
    CREATE TABLE IF NOT EXISTS tags (
      id       INTEGER PRIMARY KEY,
      slug     TEXT UNIQUE NOT NULL,             -- ASCII 键。FTS 只索引它，绕开中文分词
      zh       TEXT NOT NULL,
      aliases  TEXT NOT NULL DEFAULT '[]',       -- JSON 数组，口语别名
      category TEXT NOT NULL                     -- scene|object|color|clothing|emotion|activity|document|people|quality
    );
    CREATE TABLE IF NOT EXISTS photo_tags (
      photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
      tag_id     INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
      confidence REAL NOT NULL,
      source     TEXT NOT NULL,                  -- 'vlm' | 'rule' | 'exif'
      PRIMARY KEY (photo_id, tag_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pt_tag ON photo_tags(tag_id);
    -- ============ FTS5：标签走 slug（ASCII），OCR 走 trigram，分表 ============
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_tags USING fts5(photo_id UNINDEXED, slugs);
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_ocr  USING fts5(photo_id UNINDEXED, text, tokenize='trigram');
    -- ============ 人物 / 人脸 ============
    CREATE TABLE IF NOT EXISTS persons (
      id       INTEGER PRIMARY KEY,
      name     TEXT,
      relation TEXT                              -- self|father|mother|partner|child|grandparent
    );
    CREATE TABLE IF NOT EXISTS photo_faces (
      face_rowid INTEGER PRIMARY KEY,            -- 与 vec_faces 对齐
      photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
      person_id  INTEGER REFERENCES persons(id),
      bbox       TEXT,
      source     TEXT NOT NULL DEFAULT 'face-detector'
    );
    CREATE INDEX IF NOT EXISTS idx_faces_person ON photo_faces(person_id);
    CREATE INDEX IF NOT EXISTS idx_faces_photo  ON photo_faces(photo_id);
    -- ============ 事件（时间线聚类，本版仅建表占位） ============
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY, name TEXT,
      confidence REAL, source TEXT, start_at INTEGER, end_at INTEGER
    );
    -- ============ 智能相册：存查询 DSL（slug 形式），不物化结果 ============
    CREATE TABLE IF NOT EXISTS smart_albums (
      id INTEGER PRIMARY KEY, name TEXT NOT NULL,
      dsl TEXT NOT NULL, created_at INTEGER NOT NULL
    );
    -- ============ sqlite-vec：人脸向量（512d，SFace），供 KNN 聚类 ============
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_faces USING vec0(
      face_rowid INTEGER PRIMARY KEY,
      embedding  float[512]
    );
  `);
  try { db.exec(`ALTER TABLE photos ADD COLUMN tag_attempts INTEGER NOT NULL DEFAULT 0`); } catch { /* 已存在 */ }
  return db;
}
