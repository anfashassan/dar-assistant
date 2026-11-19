"""Background data refresh.

The image ships with a scraped snapshot (data/raw) so the bot works instantly.
Every REFRESH_INTERVAL_H hours the container re-runs the scraper in a subprocess
into a scratch directory, validates the result and hot-swaps the knowledge base.
If a refresh fails or looks incomplete, the last good data keeps serving.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .knowledge import KnowledgeBase

log = logging.getLogger("app.refresh")


class Refresher:
    def __init__(self, data_dir: Path, on_new_kb, interval_h: float | None = None,
                 work_dir: Path | None = None, size_fn=lambda: 0):
        self.data_dir = data_dir
        self.on_new_kb = on_new_kb  # callback(KnowledgeBase)
        self.size_fn = size_fn      # current number of pages, used to reject partial scrapes
        self.interval_h = float(os.getenv("REFRESH_INTERVAL_H", "24")) if interval_h is None else interval_h
        self.work_dir = work_dir or Path(os.getenv("REFRESH_WORK_DIR", "/tmp/dar-refresh"))
        self.timeout_s = int(os.getenv("REFRESH_TIMEOUT_S", "2400"))
        self.extra_args = os.getenv("REFRESH_ARGS", "--max-listings 400 --max-wasalt-pages 550").split()
        self.min_ratio = float(os.getenv("REFRESH_MIN_RATIO", "0.6"))
        self.status = {"enabled": self.interval_h > 0, "running": False, "last_run": None,
                       "last_status": None, "last_message": "", "next_run": None, "active_dataset": "bundled"}
        self.cmd_override = None  # tests can inject a fake scraper command
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    # ---------------------------------------------------------- scheduling
    def start(self, current: KnowledgeBase):
        if self.interval_h <= 0:
            return
        self._task = asyncio.create_task(self._loop(current))

    async def stop(self):
        if self._task:
            self._task.cancel()

    def _initial_delay(self, kb: KnowledgeBase) -> float:
        scraped = kb.manifest.get("scraped_at")
        age_h = 1e9
        if scraped:
            try:
                age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(scraped)).total_seconds() / 3600
            except ValueError:
                pass
        # stale snapshot -> refresh soon after boot (but not instantly, let the app warm up)
        return 300.0 if age_h >= self.interval_h else (self.interval_h - age_h) * 3600

    async def _loop(self, kb: KnowledgeBase):
        delay = self._initial_delay(kb)
        while True:
            self.status["next_run"] = datetime.fromtimestamp(time.time() + delay, timezone.utc).isoformat(timespec="seconds")
            await asyncio.sleep(delay)
            await self.run_once()
            delay = self.interval_h * 3600

    # ------------------------------------------------------------- one run
    async def run_once(self, current_pages: int | None = None) -> dict:
        if self._lock.locked():
            return {"status": "already running"}
        async with self._lock:
            self.status.update(running=True, last_run=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            out_dir = self.work_dir / "raw"
            try:
                if self.work_dir.exists():
                    shutil.rmtree(self.work_dir)
                out_dir.mkdir(parents=True)
                # start from current data: failed sites keep theirs, records only merge
                current_raw = self.data_dir / "raw"
                for name in ("pages.jsonl", "projects.json", "listings.json", "embeddings.npz"):
                    if (current_raw / name).exists():
                        shutil.copy2(current_raw / name, out_dir / name)
                cmd = (self.cmd_override(out_dir) if self.cmd_override else
                       [sys.executable, "-m", "scraper.run", "--out", str(out_dir), *self.extra_args])
                log.info("refresh: %s", " ".join(cmd))
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=str(Path(__file__).resolve().parent.parent),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                try:
                    out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
                except asyncio.TimeoutError:
                    proc.kill()
                    raise RuntimeError(f"scraper timed out after {self.timeout_s}s")
                tail = out.decode("utf-8", "replace")[-600:]
                if proc.returncode != 0:
                    raise RuntimeError(f"scraper exited {proc.returncode}: {tail}")

                manifest = json.loads((out_dir / "manifest.json").read_text())
                baseline = current_pages if current_pages is not None else self.size_fn()
                if manifest.get("pages", 0) < max(20, baseline * self.min_ratio):
                    raise RuntimeError(f"refresh looks incomplete ({manifest.get('pages')} pages vs "
                                       f"{baseline} currently); keeping existing data")

                kb = await asyncio.to_thread(KnowledgeBase.load, self.data_dir, out_dir)
                self.on_new_kb(kb)
                self.status.update(last_status="ok", active_dataset="live refresh",
                                   last_message=f"{manifest.get('pages')} pages, {manifest.get('projects')} projects, "
                                                f"{manifest.get('listings')} listings")
                log.info("refresh ok: %s", self.status["last_message"])
            except Exception as e:
                self.status.update(last_status="failed", last_message=str(e)[:400])
                log.warning("refresh failed: %s", e)
            finally:
                self.status["running"] = False
            return dict(self.status)
