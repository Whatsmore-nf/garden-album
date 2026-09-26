import { contextBridge, ipcRenderer } from "electron";
const api = {
  search: (q: string) => ipcRenderer.invoke("search", q),
  lastSeen: (relation: string) => ipcRenderer.invoke("lastSeen", relation),
  stats: () => ipcRenderer.invoke("stats"),
  createAlbum: (name: string, query: string) => ipcRenderer.invoke("album:create", name, query),
  listAlbums: () => ipcRenderer.invoke("album:list"),
  openAlbum: (id: number) => ipcRenderer.invoke("album:open", id),
  deleteAlbum: (id: number) => ipcRenderer.invoke("album:delete", id),
  listPersons: () => ipcRenderer.invoke("persons:list"),
  updatePerson: (id: number, patch: { name?: string; relation?: string | null }) =>
    ipcRenderer.invoke("persons:update", id, patch),
  scan: (root: string) => ipcRenderer.invoke("ingest:scan", root),
  watch: (root: string) => ipcRenderer.invoke("ingest:watch", root),
  onIngestProgress: (cb: (d: unknown) => void) => {
    const listener = (_e: unknown, d: unknown) => cb(d);
    ipcRenderer.on("ingest:progress", listener);
    return () => ipcRenderer.removeListener("ingest:progress", listener);
  },
  cleanupSuggest: () => ipcRenderer.invoke("cleanup:suggest"),
  cleanupApply: (ids: number[]) => ipcRenderer.invoke("cleanup:apply", ids),
};
contextBridge.exposeInMainWorld("gallery", api);
export type GalleryApi = typeof api;
