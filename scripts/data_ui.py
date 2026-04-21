#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

MAX_TEXT_CHARS = 12000
MAX_JSONL_ROWS = 2000
MAX_NUMERIC_POINTS = 2000


def safe_rel_path(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("path is outside data directory")
    if not candidate.exists() or not candidate.is_file():
        raise ValueError("file does not exist")
    return candidate


def file_tree(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            rows.append(
                {
                    "path": str(p.relative_to(root)),
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                    "ext": p.suffix.lower() or "(none)",
                }
            )
    return rows


def flatten_numeric(obj: Any, prefix: str = "", out: dict[str, list[float]] | None = None) -> dict[str, list[float]]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten_numeric(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten_numeric(v, prefix, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.setdefault(prefix or "value", []).append(float(obj))
    return out


def summarize_values(values: list[float]) -> dict[str, Any]:
    sample = values[:MAX_NUMERIC_POINTS]
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "sample": sample,
    }


def json_overview(obj: Any) -> dict[str, Any]:
    overview: dict[str, Any] = {"kind": type(obj).__name__}
    if isinstance(obj, dict):
        overview["keys"] = list(obj.keys())[:100]
        overview["size"] = len(obj)
    elif isinstance(obj, list):
        overview["size"] = len(obj)
        if obj:
            overview["item_kind"] = type(obj[0]).__name__
    numeric = flatten_numeric(obj)
    top = sorted(((k, v) for k, v in numeric.items() if v), key=lambda kv: len(kv[1]), reverse=True)[:20]
    overview["numeric"] = [{"key": k, **summarize_values(v)} for k, v in top]
    return overview


def load_jsonl(path: Path) -> dict[str, Any]:
    numeric: dict[str, list[float]] = {}
    preview: list[Any] = []
    rows = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            if rows <= 3:
                preview.append(json.loads(line))
            if rows <= MAX_JSONL_ROWS:
                row_numeric = flatten_numeric(json.loads(line))
                for k, vals in row_numeric.items():
                    numeric.setdefault(k, []).extend(vals)
    top = sorted(((k, v) for k, v in numeric.items() if v), key=lambda kv: len(kv[1]), reverse=True)[:20]
    return {
        "kind": "jsonl",
        "rows": rows,
        "preview": preview,
        "numeric": [{"key": k, **summarize_values(v)} for k, v in top],
    }


def load_npy(path: Path) -> dict[str, Any]:
    arr = np.load(path, allow_pickle=False)
    info: dict[str, Any] = {
        "kind": "npy",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if np.issubdtype(arr.dtype, np.number):
        flat = arr.reshape(-1)
        values = flat.astype(float).tolist()
        info["numeric"] = [{"key": "array", **summarize_values(values)}]
    return info


def load_npz(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": "npz", "arrays": []}
    with np.load(path, allow_pickle=False) as z:
        for name in sorted(z.files):
            arr = z[name]
            item: dict[str, Any] = {"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
            if np.issubdtype(arr.dtype, np.number):
                values = arr.reshape(-1).astype(float).tolist()
                item["numeric"] = summarize_values(values)
            out["arrays"].append(item)
    return out


def load_text(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    return {
        "kind": "text",
        "chars": len(text),
        "preview": text[:MAX_TEXT_CHARS],
    }


def inspect_file(path: Path) -> dict[str, Any]:
    ext = path.suffix.lower()
    if ext == ".json":
        with path.open() as f:
            return {"kind": "json", "overview": json_overview(json.load(f))}
    if ext == ".jsonl":
        return load_jsonl(path)
    if ext == ".npy":
        return load_npy(path)
    if ext == ".npz":
        return load_npz(path)
    if ext in {".txt", ".log", ".md", ".csv", ".tsv"}:
        return load_text(path)
    return {"kind": "unsupported", "message": f"No parser for {ext or 'no extension'}"}


HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>
  <title>Corollary Data Viewer</title>
  <style>
    body { font-family: ui-sans-serif, system-ui; margin: 0; background: #0b1020; color: #e7ecff; }
    .wrap { display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }
    .left { border-right: 1px solid #223; padding: 12px; overflow: auto; }
    .right { padding: 14px; overflow: auto; }
    input { width: 100%; padding: 8px; background: #131a31; color: #e7ecff; border: 1px solid #334; border-radius: 8px; }
    .file { padding: 7px; border-radius: 8px; cursor: pointer; margin-top: 6px; border: 1px solid transparent; }
    .file:hover,.file.active { background: #151f3d; border-color: #3c4e86; }
    .muted { color: #9aa4cb; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 12px; }
    .chip { background: #151f3d; border: 1px solid #334; border-radius: 999px; padding: 5px 10px; font-size: 12px; }
    pre { white-space: pre-wrap; background: #10172d; border: 1px solid #334; border-radius: 10px; padding: 10px; }
    canvas { background: #0f1630; border: 1px solid #334; border-radius: 8px; display:block; }
    .metric { margin: 12px 0 16px; }
    h1,h2,h3 { margin: 4px 0 10px; }
    a { color: #9bb4ff; }
  </style>
</head>
<body>
<div class=\"wrap\">
  <div class=\"left\">
    <h2>Data Folder</h2>
    <input id=\"q\" placeholder=\"Filter files...\" />
    <div class=\"chips\" id=\"stats\"></div>
    <div id=\"files\"></div>
  </div>
  <div class=\"right\">
    <h1>Corollary Data Viewer</h1>
    <div class=\"muted\">Select a file in <code>data/</code> to inspect it.</div>
    <div id=\"detail\"></div>
  </div>
</div>
<script>
let allFiles = [];
let selected = null;

function fmtBytes(b){const u=['B','KB','MB','GB'];let i=0,n=b;while(n>1024&&i<u.length-1){n/=1024;i++;}return `${n.toFixed(1)} ${u[i]}`}

function renderStats(){
  const by = {};
  for (const f of allFiles) by[f.ext] = (by[f.ext] || 0) + 1;
  const stats = document.getElementById('stats');
  stats.innerHTML = `<span class='chip'>${allFiles.length} files</span>` +
    Object.entries(by).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([k,v]) => `<span class='chip'>${k}: ${v}</span>`).join('');
}

function renderFiles(){
  const q = document.getElementById('q').value.toLowerCase();
  const rows = allFiles.filter(f => f.path.toLowerCase().includes(q));
  const el = document.getElementById('files');
  el.innerHTML = rows.map(f =>
    `<div class='file ${selected===f.path?'active':''}' data-path='${f.path}'>`+
    `<div>${f.path}</div><div class='muted'>${f.ext} · ${fmtBytes(f.size)}</div></div>`).join('');
  el.querySelectorAll('.file').forEach(n => n.onclick = () => selectFile(n.dataset.path));
}

function spark(canvas, values){
  const ctx = canvas.getContext('2d');
  const w = canvas.width = 600, h = canvas.height = 120;
  ctx.clearRect(0,0,w,h);
  if (!values.length) return;
  let min = Math.min(...values), max = Math.max(...values);
  if (max===min) max = min + 1;
  ctx.strokeStyle = '#8eb1ff'; ctx.lineWidth = 1.5; ctx.beginPath();
  values.forEach((v, i) => {
    const x = i*(w/(values.length-1 || 1));
    const y = h - ((v-min)/(max-min))*h;
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
}

function metricHtml(m){
  return `<div class='metric'><h3>${m.key}</h3><div class='muted'>count=${m.count}, min=${m.min.toFixed(4)}, max=${m.max.toFixed(4)}, mean=${m.mean.toFixed(4)}</div><canvas id='c_${m.key.replace(/[^a-z0-9]/ig,'_')}'></canvas></div>`;
}

async function selectFile(path){
  selected = path;
  renderFiles();
  const res = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
  const data = await res.json();
  const d = document.getElementById('detail');
  let html = `<h2>${path}</h2><div class='muted'>kind: ${data.kind}</div>`;

  if (data.kind === 'json') {
    const ov = data.overview;
    html += `<pre>${JSON.stringify({kind:ov.kind,size:ov.size,keys:ov.keys}, null, 2)}</pre>`;
    if (ov.numeric && ov.numeric.length) html += ov.numeric.slice(0,8).map(metricHtml).join('');
  } else if (data.kind === 'jsonl') {
    html += `<pre>${JSON.stringify({rows:data.rows,preview:data.preview}, null, 2)}</pre>`;
    if (data.numeric && data.numeric.length) html += data.numeric.slice(0,8).map(metricHtml).join('');
  } else if (data.kind === 'npy') {
    html += `<pre>${JSON.stringify({shape:data.shape,dtype:data.dtype}, null, 2)}</pre>`;
    if (data.numeric) html += data.numeric.map(metricHtml).join('');
  } else if (data.kind === 'npz') {
    html += `<pre>${JSON.stringify({arrays:data.arrays.map(a=>({name:a.name,shape:a.shape,dtype:a.dtype}))}, null, 2)}</pre>`;
    const metrics = data.arrays.filter(a=>a.numeric).map(a=>({key:a.name, ...a.numeric}));
    if (metrics.length) html += metrics.slice(0,8).map(metricHtml).join('');
  } else if (data.kind === 'text') {
    html += `<pre>${data.preview.replace(/[<>&]/g, c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}</pre>`;
  } else {
    html += `<pre>${JSON.stringify(data, null, 2)}</pre>`;
  }

  d.innerHTML = html;
  const metrics = [];
  if (data.kind === 'json') metrics.push(...(data.overview.numeric || []).slice(0,8));
  if (data.kind === 'jsonl') metrics.push(...(data.numeric || []).slice(0,8));
  if (data.kind === 'npy') metrics.push(...(data.numeric || []));
  if (data.kind === 'npz') for (const a of data.arrays) if (a.numeric) metrics.push({key:a.name, ...a.numeric});
  for (const m of metrics.slice(0,8)) {
    const id = `c_${m.key.replace(/[^a-z0-9]/ig,'_')}`;
    const c = document.getElementById(id);
    if (c) spark(c, m.sample || []);
  }
}

async function init(){
  const res = await fetch('/api/files');
  allFiles = await res.json();
  renderStats();
  renderFiles();
}

document.getElementById('q').addEventListener('input', renderFiles);
init();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    root: Path

    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_html(HTML)
        if parsed.path == "/api/files":
            return self._send_json(file_tree(self.root))
        if parsed.path == "/api/file":
            q = parse_qs(parsed.query)
            rel = q.get("path", [""])[0]
            try:
                path = safe_rel_path(self.root, rel)
                info = inspect_file(path)
                info["path"] = rel
                return self._send_json(info)
            except Exception as exc:
                return self._send_json({"error": str(exc)}, status=400)
        self.send_error(404)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local UI for browsing data/ artifacts")
    parser.add_argument("--data-dir", default="data", help="Data directory to visualize")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise SystemExit(f"Data directory not found: {data_root}")

    Handler.root = data_root
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Data UI serving {data_root}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
