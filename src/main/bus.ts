import { EventEmitter } from "node:events";
/** 主进程内部事件总线（ingest 进度等）。index.ts 负责把事件转发给所有窗口。 */
export const bus = new EventEmitter();
