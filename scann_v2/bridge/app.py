from __future__ import annotations

import html
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode

import requests
from fastapi import FastAPI, HTTPException, Request  # type: ignore[import-not-found]
from fastapi.responses import HTMLResponse  # type: ignore[import-not-found]
from pydantic import BaseModel, Field  # type: ignore[import-not-found]

logger = logging.getLogger("scann_bridge")
logging.basicConfig(level=os.getenv("BRIDGE_LOG_LEVEL", "INFO").upper())

MANIFEST_VERSION = "2.1"
REAL_DETAIL_TYPES = {"asteroid", "supernova", "variable_star"}
BOGUS_DETAIL_TYPES = {
    "satellite_trail",
    "noise",
    "diffraction_spike",
    "cmos_condensation",
    "corresponding",
}
ALL_DETAIL_TYPES = REAL_DETAIL_TYPES | BOGUS_DETAIL_TYPES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _strip_known_prefix(stem: str) -> str:
    for p in ("FW_", "fw_", "Fw_"):
        if stem.startswith(p):
            return stem[len(p) :]
    return stem


def _normalize_key(value: Optional[str]) -> str:
    if not value:
        return ""
    key = Path(value).stem
    if key.lower().endswith("__aligned_crop"):
        key = key[: -len("__aligned_crop")]
    key = _strip_known_prefix(key)
    return key.strip().lower()


def _label_from_detail_type(detail_type: Optional[str]) -> Optional[str]:
    if not detail_type:
        return None
    key = detail_type.strip().lower()
    if key in REAL_DETAIL_TYPES:
        return "real"
    if key in BOGUS_DETAIL_TYPES:
        return "bogus"
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class BridgeConfig:
    dataset_root: Path
    sqlite_path: Path
    new_dir: Path
    old_dir: Path
    new_marked_dir: Path
    preview_cache_dir: Path
    manifest_path: Path

    label_studio_url: str
    label_studio_token: str
    label_studio_project_id: int

    public_data_base_url: str
    js9_base_url: str

    @staticmethod
    def from_env() -> "BridgeConfig":
        dataset_root = Path(os.getenv("BRIDGE_DATASET_ROOT", "/dataset")).resolve()
        sqlite_path = Path(os.getenv("BRIDGE_SQLITE_PATH", str(dataset_root / "annotations.db"))).resolve()

        new_dir = Path(os.getenv("BRIDGE_NEW_DIR", str(dataset_root / "new"))).resolve()
        old_dir = Path(os.getenv("BRIDGE_OLD_DIR", str(dataset_root / "old"))).resolve()
        new_marked_dir = Path(os.getenv("BRIDGE_NEW_MARKED_DIR", str(dataset_root / "new_marked"))).resolve()
        preview_cache_dir = Path(os.getenv("BRIDGE_PREVIEW_CACHE_DIR", str(dataset_root / ".preview_cache"))).resolve()

        label_studio_url = os.getenv("BRIDGE_LABELSTUDIO_URL", "http://labelstudio:8080").rstrip("/")
        raw_token = os.getenv("BRIDGE_LABELSTUDIO_TOKEN", "").strip().strip("'\"")
        if raw_token.lower().startswith("token "):
            raw_token = raw_token[6:].strip()

        label_studio_project_id = int(os.getenv("BRIDGE_PROJECT_ID", "1").strip())
        public_data_base_url = os.getenv("BRIDGE_PUBLIC_DATA_BASE_URL", "http://127.0.0.1:3001/dataset").rstrip("/")
        js9_base_url = os.getenv("BRIDGE_JS9_BASE_URL", "http://127.0.0.1:3001").rstrip("/")

        return BridgeConfig(
            dataset_root=dataset_root,
            sqlite_path=sqlite_path,
            new_dir=new_dir,
            old_dir=old_dir,
            new_marked_dir=new_marked_dir,
            preview_cache_dir=preview_cache_dir,
            manifest_path=dataset_root / "annotations.json",
            label_studio_url=label_studio_url,
            label_studio_token=raw_token,
            label_studio_project_id=label_studio_project_id,
            public_data_base_url=public_data_base_url,
            js9_base_url=js9_base_url,
        )


CONFIG = BridgeConfig.from_env()
app = FastAPI(title="SCANN Label Studio Bridge", version="0.2.0")


class PullRequest(BaseModel):
    import_to_label_studio: bool = True
    overwrite_existing: bool = False
    limit: Optional[int] = Field(default=None, ge=1)


class PullResponse(BaseModel):
    scanned_pairs: int
    tasks_built: int
    tasks_imported: int


class WebhookResponse(BaseModel):
    updated_samples: int


class TaskRecord(BaseModel):
    sample_id: str
    file_name: str
    new_url: str
    old_url: str
    new_marked_url: Optional[str]
    preview_png: str
    js9_embed_url: str
    js9_iframe: str


def _scan_dir(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        return []
    exts = {".fits", ".fit", ".fts", ".fts2", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _build_pair_lookup() -> dict[str, dict[str, Path]]:
    new_files = _scan_dir(CONFIG.new_dir)
    old_files = _scan_dir(CONFIG.old_dir)
    marked_files = _scan_dir(CONFIG.new_marked_dir)

    old_map = {_normalize_key(p.name): p for p in old_files if _normalize_key(p.name)}
    marked_map = {_normalize_key(p.name): p for p in marked_files if _normalize_key(p.name)}

    pairs: dict[str, dict[str, Path]] = {}
    for new_path in new_files:
        key = _normalize_key(new_path.name)
        if not key:
            continue
        old_path = old_map.get(key)
        if old_path is None:
            continue
        pairs[key] = {"new": new_path, "old": old_path}
        if key in marked_map:
            pairs[key]["new_marked"] = marked_map[key]
    return pairs


def _public_url_for(path: Path) -> str:
    rel = path.resolve().relative_to(CONFIG.dataset_root)
    rel_url = "/".join(quote(part) for part in rel.parts)
    return f"{CONFIG.public_data_base_url}/{rel_url}"


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _make_js9_embed_url(new_url: str, old_url: str, sample_id: str, new_marked_url: Optional[str]) -> str:
    params: dict[str, str] = {"new": new_url, "old": old_url, "sample_id": sample_id}
    if new_marked_url:
        params["marked"] = new_marked_url
    query = urlencode(params, quote_via=quote)
    return f"{CONFIG.js9_base_url}/viewer/js9?{query}"


def _make_js9_iframe(
    new_url: str,
    old_url: str,
    new_marked_url: Optional[str],
    js9_embed_url: str,
    sample_id: str,
) -> str:
    safe_embed_url = html.escape(js9_embed_url, quote=True)
    safe_sample_id = html.escape(sample_id, quote=True)

    links = [
        f'<a href="{html.escape(new_url, quote=True)}" target="_blank" rel="noopener noreferrer">新图</a>',
        f'<a href="{html.escape(old_url, quote=True)}" target="_blank" rel="noopener noreferrer">旧图</a>',
    ]
    if new_marked_url:
        links.append(
            f'<a href="{html.escape(new_marked_url, quote=True)}" target="_blank" rel="noopener noreferrer">新图(标注)</a>'
        )

    links_html = " | ".join(links)
    return (
        '<div style="padding:8px;border:1px solid #ddd;border-radius:6px;" '
        f'data-sample-id="{safe_sample_id}">'
        '<div style="margin-bottom:6px;font-weight:600;">SCANN FITS 内嵌判读</div>'
        f'<iframe src="{safe_embed_url}" title="SCANN JS9 Viewer" loading="lazy" '
        'style="width:100%;min-height:520px;border:1px solid #ccc;border-radius:4px;"></iframe>'
        f'<div style="margin-top:6px;"><a href="{safe_embed_url}" target="_blank" rel="noopener noreferrer">在新窗口打开内嵌 Viewer</a></div>'
        f'<div style="margin-top:6px;">{links_html}</div>'
        '</div>'
    )


def _ensure_storage_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            label TEXT,
            detail_type TEXT,
            ai_suggestion TEXT,
            ai_confidence REAL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bboxes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            box_index INTEGER NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            label TEXT,
            detail_type TEXT,
            confidence REAL,
            FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bboxes_image_id ON bboxes(image_id)")


def _write_manifest() -> None:
    payload = {
        "version": MANIFEST_VERSION,
        "storage": "sqlite",
        "db_file": CONFIG.sqlite_path.name,
        "updated_at": _utc_now(),
    }
    CONFIG.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _upsert_sample(sample_id: str, file_name: str, bboxes: list[dict[str, Any]]) -> None:
    detail_type = bboxes[0].get("detail_type") if bboxes else None
    label = _label_from_detail_type(detail_type)

    CONFIG.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CONFIG.sqlite_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        _ensure_storage_schema(conn)

        with conn:
            conn.execute(
                """
                INSERT INTO images (id, file_name, label, detail_type, ai_suggestion, ai_confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    file_name=excluded.file_name,
                    label=excluded.label,
                    detail_type=excluded.detail_type,
                    ai_suggestion=excluded.ai_suggestion,
                    ai_confidence=excluded.ai_confidence,
                    updated_at=excluded.updated_at
                """,
                (sample_id, file_name, label, detail_type, None, None, _utc_now()),
            )
            conn.execute("DELETE FROM bboxes WHERE image_id = ?", (sample_id,))
            for idx, bbox in enumerate(bboxes):
                conn.execute(
                    """
                    INSERT INTO bboxes (image_id, box_index, x, y, width, height, label, detail_type, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        idx,
                        int(bbox["x"]),
                        int(bbox["y"]),
                        int(bbox["width"]),
                        int(bbox["height"]),
                        bbox["label"],
                        bbox["detail_type"],
                        float(bbox.get("confidence", 1.0)),
                    ),
                )
        _write_manifest()
    finally:
        conn.close()


def _parse_ls_result_to_bbox(result: dict[str, Any], image_width: int, image_height: int) -> Optional[dict[str, Any]]:
    if result.get("type") != "rectanglelabels":
        return None

    value = result.get("value") or {}
    labels = value.get("rectanglelabels") or []
    if not labels:
        return None

    detail_type = str(labels[0]).strip().lower()
    if detail_type not in ALL_DETAIL_TYPES:
        return None

    x_pct = _safe_float(value.get("x"))
    y_pct = _safe_float(value.get("y"))
    w_pct = _safe_float(value.get("width"))
    h_pct = _safe_float(value.get("height"))
    if x_pct is None or y_pct is None or w_pct is None or h_pct is None:
        return None

    x = round(x_pct / 100.0 * image_width)
    y = round(y_pct / 100.0 * image_height)
    w = round(w_pct / 100.0 * image_width)
    h = round(h_pct / 100.0 * image_height)
    if w <= 0 or h <= 0:
        return None

    label = _label_from_detail_type(detail_type)
    if label is None:
        return None

    return {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "label": label,
        "detail_type": detail_type,
        "confidence": 1.0,
    }


def _extract_annotations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = payload.get("annotations")
    if isinstance(annotations, list):
        return [a for a in annotations if isinstance(a, dict)]

    annotation = payload.get("annotation")
    if isinstance(annotation, dict):
        return [annotation]

    return []


def _task_data_from_annotation(annotation: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    task_obj = annotation.get("task")
    if isinstance(task_obj, dict):
        data = task_obj.get("data")
        if isinstance(data, dict):
            return data

    data = payload.get("task", {}).get("data")
    if isinstance(data, dict):
        return data
    return {}


@app.get("/viewer/js9", response_class=HTMLResponse)
def js9_viewer_page(new: str, old: str, sample_id: str, marked: Optional[str] = None) -> HTMLResponse:
    urls = {"new": new, "old": old, "marked": marked}
    html_text = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>SCANN JS9 Viewer</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #0f1115; color: #e8e8e8; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px; background: #1a1f2b; border-bottom: 1px solid #323b4f; }}
    .toolbar button, .toolbar input {{ background: #2b3550; color: #e8e8e8; border: 1px solid #4a5a82; border-radius: 6px; padding: 6px 10px; }}
    .toolbar button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    #status {{ margin-left: auto; font-size: 12px; opacity: 0.9; }}
    .viewer {{ height: calc(100vh - 58px); }}
    .viewer iframe {{ width: 100%; height: 100%; border: 0; background: #000; }}
    .fallback {{ padding: 8px 10px; font-size: 12px; border-top: 1px dashed #48536d; }}
  </style>
</head>
<body>
  <div class=\"toolbar\" id=\"scann-js9-toolbar\"> 
    <button type=\"button\" id=\"btn-new\">新图</button>
    <button type=\"button\" id=\"btn-old\">旧图</button>
    <button type=\"button\" id=\"btn-marked\">新图(标注)</button>
    <button type=\"button\" id=\"btn-blink\">闪烁: 关</button>
    <label>速度(ms) <input id=\"blink-ms\" type=\"number\" min=\"100\" step=\"50\" value=\"500\" /></label>
    <button type=\"button\" id=\"btn-invert\">反色</button>
    <button type=\"button\" id=\"btn-stretch-reset\">拉伸重置</button>
    <button type=\"button\" id=\"btn-stretch-auto\">自动拉伸</button>
    <span id=\"status\"></span>
  </div>
  <div class=\"viewer\">
    <iframe id=\"scann-js9-viewer\" title=\"SCANN inline FITS viewer\" loading=\"eager\" referrerpolicy=\"no-referrer\"></iframe>
  </div>
  <div class=\"fallback\" id=\"fallback-links\"></div>

  <script>
    const urls = {_json_for_script(urls)};
    const sampleId = {_json_for_script(sample_id)};
    const iframe = document.getElementById("scann-js9-viewer");
    const status = document.getElementById("status");
    const btnMarked = document.getElementById("btn-marked");
    const btnBlink = document.getElementById("btn-blink");
    const blinkMsInput = document.getElementById("blink-ms");
    const fallback = document.getElementById("fallback-links");

    let active = "new";
    let blinkTimer = null;
    let invertOn = false;

    function setStatus(text) {{
      status.textContent = `sample: ${{sampleId}} | ${{text}}`;
    }}

    function setFrame(which) {{
      const url = urls[which];
      if (!url) return;
      active = which;
      iframe.src = url;
      setStatus(`当前: ${{which}}`);
    }}

    function stopBlink() {{
      if (blinkTimer) {{
        clearInterval(blinkTimer);
        blinkTimer = null;
      }}
      btnBlink.textContent = "闪烁: 关";
    }}

    function startBlink() {{
      const ms = Math.max(100, Number.parseInt(blinkMsInput.value || "500", 10));
      let next = active === "old" ? "new" : "old";
      blinkTimer = setInterval(() => {{
        setFrame(next);
        next = next === "new" ? "old" : "new";
      }}, ms);
      btnBlink.textContent = "闪烁: 开";
    }}

    function postViewerAction(action) {{
      try {{
        iframe.contentWindow?.postMessage({{ source: "scann-bridge", action }}, "*");
      }} catch (_err) {{
      }}
    }}

    document.getElementById("btn-new").addEventListener("click", () => setFrame("new"));
    document.getElementById("btn-old").addEventListener("click", () => setFrame("old"));
    document.getElementById("btn-marked").addEventListener("click", () => setFrame("marked"));

    btnBlink.addEventListener("click", () => {{
      if (blinkTimer) stopBlink();
      else startBlink();
    }});

    document.getElementById("btn-invert").addEventListener("click", () => {{
      invertOn = !invertOn;
      iframe.style.filter = invertOn ? "invert(1)" : "none";
      postViewerAction(invertOn ? "invertOn" : "invertOff");
    }});
    document.getElementById("btn-stretch-reset").addEventListener("click", () => postViewerAction("stretchReset"));
    document.getElementById("btn-stretch-auto").addEventListener("click", () => postViewerAction("stretchAuto"));

    if (!urls.marked) {{
      btnMarked.disabled = true;
      btnMarked.title = "当前样本缺少 new_marked 图像";
    }}

    const fallbackLinks = [
      `<a href="${{urls.new}}" target="_blank" rel="noopener noreferrer">新图</a>`,
      `<a href="${{urls.old}}" target="_blank" rel="noopener noreferrer">旧图</a>`
    ];
    if (urls.marked) fallbackLinks.push(`<a href="${{urls.marked}}" target="_blank" rel="noopener noreferrer">新图(标注)</a>`);
    fallback.innerHTML = `降级链接: ${{fallbackLinks.join(" | ")}}`;

    setFrame(active);
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_text)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "dataset_root": str(CONFIG.dataset_root),
        "sqlite_path": str(CONFIG.sqlite_path),
        "project_id": CONFIG.label_studio_project_id,
    }


@app.post("/tasks/pull", response_model=PullResponse)
def pull_tasks(req: PullRequest) -> PullResponse:
    pairs = _build_pair_lookup()

    tasks: list[dict[str, Any]] = []
    for key, v in pairs.items():
        sample_id = key
        new_path = v["new"]
        old_path = v["old"]
        marked_path = v.get("new_marked")

        preview_source = marked_path or new_path

        try:
            preview_url = _public_url_for(preview_source)
            new_url = _public_url_for(new_path)
            old_url = _public_url_for(old_path)
            marked_url = _public_url_for(marked_path) if marked_path else None
        except ValueError:
            logger.warning("文件不在 dataset_root 下，跳过 sample=%s", sample_id)
            continue

        js9_embed_url = _make_js9_embed_url(new_url, old_url, sample_id, marked_url)
        js9_iframe = _make_js9_iframe(new_url, old_url, marked_url, js9_embed_url, sample_id)

        tasks.append(
            {
                "data": TaskRecord(
                    sample_id=sample_id,
                    file_name=new_path.name,
                    new_url=new_url,
                    old_url=old_url,
                    new_marked_url=marked_url,
                    preview_png=preview_url,
                    js9_embed_url=js9_embed_url,
                    js9_iframe=js9_iframe,
                ).model_dump()
            }
        )

    tasks.sort(key=lambda t: t["data"]["file_name"])
    if req.limit is not None:
        tasks = tasks[: req.limit]

    imported = 0
    if req.import_to_label_studio:
        if not CONFIG.label_studio_token:
            raise HTTPException(status_code=400, detail="BRIDGE_LABELSTUDIO_TOKEN 未配置")

        headers = {"Authorization": f"Token {CONFIG.label_studio_token}"}
        import_url = f"{CONFIG.label_studio_url}/api/projects/{CONFIG.label_studio_project_id}/import"

        safe_token = f"{CONFIG.label_studio_token[:5]}...{CONFIG.label_studio_token[-5:]}" if len(CONFIG.label_studio_token) > 10 else "***"
        logger.info("Pushing %s tasks to %s with Token [Token %s]", len(tasks), import_url, safe_token)

        resp = requests.post(import_url, headers=headers, json=tasks, timeout=120)
        if resp.status_code >= 300:
            raise HTTPException(status_code=502, detail=f"导入 Label Studio 失败: {resp.status_code} {resp.text}")

        body = resp.json() if resp.content else {}
        if isinstance(body, dict):
            imported = int(body.get("task_count") or body.get("created") or len(tasks))
        else:
            imported = len(tasks)

    return PullResponse(scanned_pairs=len(pairs), tasks_built=len(tasks), tasks_imported=imported)


@app.post("/webhook/labelstudio", response_model=WebhookResponse)
async def labelstudio_webhook(request: Request) -> WebhookResponse:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload 必须是 JSON 对象")

    updated = 0
    for ann in _extract_annotations(payload):
        data = _task_data_from_annotation(ann, payload)
        sample_id = str(data.get("sample_id") or "").strip()
        file_name = str(data.get("file_name") or sample_id or "").strip()
        if not sample_id:
            continue

        results = ann.get("result") or []
        if not isinstance(results, list):
            continue

        first_result = results[0] if results else {}
        image_width = int((first_result or {}).get("original_width") or (first_result or {}).get("image_width") or data.get("image_width") or 0)
        image_height = int((first_result or {}).get("original_height") or (first_result or {}).get("image_height") or data.get("image_height") or 0)

        if image_width <= 0 or image_height <= 0:
            logger.warning("缺少图像尺寸信息，跳过 sample=%s", sample_id)
            continue

        bboxes: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            box = _parse_ls_result_to_bbox(result, image_width, image_height)
            if box is not None:
                bboxes.append(box)

        _upsert_sample(sample_id=sample_id, file_name=file_name or sample_id, bboxes=bboxes)
        updated += 1

    return WebhookResponse(updated_samples=updated)


