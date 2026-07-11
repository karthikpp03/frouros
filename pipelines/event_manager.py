"""
pipelines/event_manager.py
===========================
Event Manager — the bridge between the real-time tracking loop and the
(slow) AI pipeline.

Problem this solves
--------------------
The original single-event pipeline (see pipelines/event_pipeline.py and
the pre-v5 main.py) kept ONE global recording slot: only one Event could
exist at a time, and close_event() ran the full AI pipeline (VideoMAE ->
Qwen summary -> attribute extraction -> Telegram) *inline* in the main
loop, blocking video capture/tracking for several seconds every time an
event ended. It also meant two people in the ROI at once were merged
into a single shared event.

This module introduces two classes:

  TrackEvent   — one independent event owned by exactly one Track ID.
                 Holds its own VideoWriter, its own best-frame / best-crop
                 trackers, and its own event record. Nothing here is
                 module-level/shared, so N TrackEvents run fully in
                 parallel without interfering with each other — this is
                 exactly what utils/image_utils.py and utils/crop_utils.py
                 could NOT do on their own, since those hold a single
                 shared best_frame / best_crops dict sized for one
                 concurrent event.

  EventManager — owns the dict of currently-open TrackEvents (keyed by
                 track_id), decides when to open/close them each frame,
                 and hands every closed event to a background worker
                 queue for AI processing so the tracking loop never
                 blocks on it.

Design notes
------------
* The real-time side (`update_detections()` / `tick()`) is called once
  per frame from main.py. Both are O(active tracks) and do no model
  inference — they just write a video frame per active event and check
  timestamps. This is what keeps detection/tracking real-time.

* AI processing (VideoMAE, Qwen, Groq, Telegram) is delegated to a
  single dedicated background thread reading off a queue.Queue.
  `_finalize_event()` re-uses every one of the existing pipeline calls
  verbatim (`extract_smart_frames`, `generate_summary`,
  `extract_person_attributes`, `send_telegram_alert`, `tg_send_message`,
  `save_memory_append`, `gallery_save`) — nothing about *how* an event
  gets summarised has changed, only *when* and *on which thread*, and
  each event now supplies its own isolated best_frame/crop instead of
  reading shared globals (which is what makes concurrent events safe).

* One dedicated worker thread (not a pool) is intentional: Qwen2.5-VL
  and VideoMAE are singleton models loaded once (models/qwen_vl.py,
  models/videomae.py). Calling .generate() on the same GPU model from
  multiple threads at once risks CUDA contention/OOM, and
  event_memory.json / reid_gallery.json are read-modify-write files
  that aren't safe to write from multiple threads concurrently.
  Serialising finalisation keeps GPU access and disk writes safe while
  staying fully asynchronous relative to the tracking loop: closed
  events simply queue up and are summarised one at a time in the
  background, never blocking detection/tracking, and never blocking
  each other's *recording* (only their AI write-up is serialised).

* GPU MEMORY OPTIMISATION: Qwen2.5-VL-7B (~5-6 GB in 4-bit) used to be
  loaded once at startup and held on the GPU permanently, alongside
  YOLO — leaving little headroom for the activation/KV-cache spike
  during `.generate()`, which is what caused CUDA OOM. `_finalize_event()`
  now loads VideoMAE, extracts smart frames, fully releases VideoMAE,
  and only THEN loads Qwen for the summary + attribute-extraction calls,
  releasing Qwen again immediately after. So at any given moment the GPU
  holds at most: YOLO (permanently, since it runs continuously) + ONE of
  {VideoMAE, Qwen} — never both heavy models at once, and neither of
  them outside of the brief window they're actually being used in.

* Each TrackEvent = exactly one Track ID, per the requirement. If the
  tracker reassigns a person's Track ID (e.g. after occlusion), the old
  Event just times out and closes normally, and a new Event opens for
  the new Track ID — the ReID identity (`reid_id`) is still attached to
  the event's person record, so the ReID gallery and query_pipeline
  keep working exactly as before.
"""

import cv2
import time
import queue
import gc
import threading
import itertools
from datetime import datetime

import torch

from config.settings import (
    EVENTS_DIR, PERSON_CROPS_DIR, SMART_FRAMES_DIR, CHAT_ID,
    FRAME_WIDTH, FRAME_HEIGHT,
    REID_GRACE_PERIOD, NO_PERSON_TIMEOUT,
    USE_OPENAI, ENABLE_FACE_RECOGNITION,
)
from memory.event_memory        import empty_event_record, empty_person_record, save_memory_append
from memory.gallery             import gallery_was_recent, gallery_purge_stale, gallery_save
from models.videomae            import extract_smart_frames, load_videomae, unload_videomae
from models.qwen_vl             import load_qwen, unload_qwen
#from models.smolvlm             import load_qwen, unload_qwen
from pipelines.summary_pipeline import extract_person_attributes
from telegram.alerts            import send_telegram_alert, tg_send_message
from utils.event_logger         import log_event_header, log_block, log_completion

# Single routing decision point for "Qwen vs OpenAI" — see
# services/summary_router.py. event_manager never decides the provider
# itself, it just asks the router for a summary.
from services.summary_router    import generate_event_summary

# Phase 2 — SQLite persistence. Purely additive: db_writer.persist_event()
# is called once finalisation already has everything it needs (summary +
# structured data), after the existing JSON-memory/Telegram steps. Never
# raises — see services/db_writer.py docstring.
from services.db_writer         import persist_event, build_structured_from_qwen

# Single default camera id until multi-camera support / config/settings.py
# CAMERA_ID is introduced — kept local to this module so no other file
# needs to change to support Phase 2.
_DEFAULT_CAMERA_ID = "cam_default"


_event_id_counter = itertools.count(0)


def _next_event_id():
    """Monotonically increasing event id, mirroring the original
    global `event_id` counter. Only ever called from the main tracking
    thread (inside TrackEvent.__init__), so no lock is required."""
    return next(_event_id_counter)


_NOT_VISIBLE = "Not Clearly Visible"


def _persons_attrs_from_structured(structured):
    """
    Build the same list[dict] shape that
    pipelines.summary_pipeline.extract_person_attributes() returns
    (keys: appearance, actions, objects, movement, waiting) directly
    from the OpenAI structured payload.

    Used ONLY on the OpenAI branch, so JSON memory (person["appearance"]
    etc.) still gets populated even though Qwen never ran for this
    event — see the ROUTER note in _finalize_event() for why Qwen must
    never load here.
    """
    persons  = structured.get("persons") or []
    actions  = [a.get("action") for a in (structured.get("actions") or []) if a.get("action")]
    objects  = [o.get("object_type") for o in (structured.get("objects") or []) if o.get("object_type")]
    movement = structured.get("movement") or {}
    movement_desc = movement.get("direction") if movement.get("direction") != _NOT_VISIBLE else None
    waiting  = bool(movement.get("loitering"))

    if not persons:
        return [{
            "appearance": None,
            "actions":    actions,
            "objects":    objects,
            "movement":   movement_desc,
            "waiting":    waiting,
        }]

    result = []
    for p in persons:
        parts = [
            p.get(k) for k in ("top_clothing", "bottom_clothing", "headwear")
            if p.get(k) and p.get(k) != _NOT_VISIBLE
        ]
        appearance = ", ".join(parts) if parts else (
            p.get("body_build") if p.get("body_build") != _NOT_VISIBLE else None
        )
        result.append({
            "appearance": appearance,
            "actions":    actions,
            "objects":    objects,
            "movement":   movement_desc,
            "waiting":    waiting,
        })
    return result


class TrackEvent:
    """
    One independent event, owned by exactly one Track ID, from the
    moment it enters the ROI until it leaves / times out.

    All state is instance-level (recording, best frame, best crop,
    timeline, record) — this is what allows many TrackEvents to run at
    once without one clobbering another's data.
    """

    def __init__(self, track_id, reid_id, fps):
        self.event_id = _next_event_id()
        self.track_id = track_id
        self.reid_id  = reid_id
        self.fps      = fps

        self.start_time = time.time()
        self.last_seen  = self.start_time

        # ── own recording ────────────────────────────────────────────
        self.output_path  = f"{EVENTS_DIR}/event_{self.event_id}.mp4"
        fourcc             = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer  = cv2.VideoWriter(
            self.output_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # ── own best-frame tracker (replaces utils/image_utils.py's
        #    shared best_frame/best_score for the concurrent case) ────
        self.best_frame = None
        self.best_score = 0

        # ── own best-crop tracker (replaces utils/crop_utils.py's
        #    shared best_crops dict for the concurrent case) ──────────
        self.best_crop       = None
        self.best_crop_score = 0

        # ── own timeline ─────────────────────────────────────────────
        self.timeline = [
            {"t": datetime.now().strftime("%H:%M:%S"), "note": "entered ROI"}
        ]

        # ── own event record (same schema as empty_event_record) ────
        self.record = empty_event_record(
            self.event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        pid   = f"event{self.event_id}_person1"
        p_rec = empty_person_record(pid, track_id, self.event_id)
        p_rec["reid_id"]    = reid_id
        p_rec["first_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.record["persons"].append(p_rec)

        print(f"[EVENT START] Event {self.event_id} (track {track_id})")

        log_event_header(self.event_id)
        log_block("YOLO", "Detecting persons...", "1 person detected.")
        log_block("TRACKING", f"Track ID : {track_id}")

    # ------------------------------------------------------------------
    def update(self, frame, box, confidence, reid_id, now):
        """Called every frame this Track ID is detected in the ROI.
        Writes to this event's own VideoWriter and updates this event's
        own best-frame/best-crop trackers — never touches any other
        event's state."""
        x1, y1, x2, y2 = box
        self.last_seen = now
        if reid_id is not None:
            self.reid_id = reid_id
            self.record["persons"][0]["reid_id"] = reid_id

        area  = (x2 - x1) * (y2 - y1)
        score = area * confidence

        if score > self.best_score:
            self.best_score = score
            self.best_frame = frame.copy()

        H, W = frame.shape[:2]
        px1, py1 = max(0, x1 - 10), max(0, y1 - 10)
        px2, py2 = min(W, x2 + 10), min(H, y2 + 10)
        crop = frame[py1:py2, px1:px2]
        if crop.size and score > self.best_crop_score:
            self.best_crop_score = score
            self.best_crop       = crop.copy()

        if self.video_writer is not None:
            self.video_writer.write(frame)

    # ------------------------------------------------------------------
    def elapsed_since_seen(self, now):
        return now - self.last_seen

    def grace_period(self):
        """Same grace-period rule as the original monolith
        (REID_GRACE_PERIOD vs NO_PERSON_TIMEOUT), scoped to this
        event's own identity instead of a shared active_reid_ids set."""
        return REID_GRACE_PERIOD if self.reid_id else NO_PERSON_TIMEOUT

    # ------------------------------------------------------------------
    def to_finalize_job(self):
        """Release this event's local resources (video writer) and
        package everything the background worker needs into a plain
        dict — the worker never touches `self` directly, so there's no
        shared mutable state between the tracking thread and the
        finalisation thread."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.timeline.append({
            "t": datetime.now().strftime("%H:%M:%S"),
            "note": "left ROI / event closed",
        })
        self.record["timeline"] = self.timeline

        return {
            "event_id":    self.event_id,
            "track_id":    self.track_id,
            "reid_id":     self.reid_id,
            "output_path": self.output_path,
            "start_time":  self.start_time,
            "record":      self.record,
            "best_frame":  self.best_frame,
            "best_crop":   self.best_crop,
        }


class EventManager:
    """
    Owns every currently-open TrackEvent plus the background
    finalisation worker.

    main.py only ever needs two calls per frame:
      event_manager.update_detections(frame, detections, fps, now)
      event_manager.tick(now)
    Both are cheap and non-blocking; AI work happens later, off-thread.
    """

    def __init__(self):
        self.events          = {}   # track_id -> TrackEvent
        self._finalize_queue = queue.Queue()
        self._worker = threading.Thread(target=self._finalize_worker, daemon=True)
        self._worker.start()

    # ==================================================================
    # REAL-TIME SIDE — runs on the main tracking thread. No AI, no
    # blocking I/O beyond writing this frame to each open event's video.
    # ==================================================================
    def update_detections(self, frame, detections, fps, now):
        """
        detections: list of (x1, y1, x2, y2, track_id, confidence, reid_id)

        Opens a brand-new TrackEvent the instant a Track ID is seen for
        the first time (satisfies "immediately create a new Event"),
        and feeds every already-open TrackEvent its own frame.
        """
        for (x1, y1, x2, y2, track_id, confidence, reid_id) in detections:
            event = self.events.get(track_id)
            if event is None:
                event = TrackEvent(track_id, reid_id, fps)
                self.events[track_id] = event
            event.update(frame, (x1, y1, x2, y2), confidence, reid_id, now)

    def tick(self, now):
        """
        Close any TrackEvent whose Track ID hasn't been seen for longer
        than its own grace period. Each event is evaluated and closed
        completely independently of every other event — one person
        leaving (or their track timing out) never blocks or delays
        anyone else's event.
        """
        closed_any = False
        for track_id in list(self.events.keys()):
            event   = self.events[track_id]
            elapsed = event.elapsed_since_seen(now)
            grace   = event.grace_period()

            if elapsed <= grace:
                continue

            # Same "identity briefly reappeared" leeway the original
            # monolith gave, now scoped to this event's own reid_id
            # instead of a shared active_reid_ids set.
            if event.reid_id and gallery_was_recent(event.reid_id) and elapsed < grace * 2:
                continue

            del self.events[track_id]
            self._close_event(event)
            closed_any = True

        if closed_any:
            gallery_purge_stale()

    def _close_event(self, event):
        print(f"[EVENT END] Event {event.event_id} (track {event.track_id})")
        job = event.to_finalize_job()
        if job["best_frame"] is None:
            print(f"[WARNING] Event {job['event_id']} has no best_frame — skipping.")
            return
        self._finalize_queue.put(job)   # hand-off is non-blocking

    def close_all(self):
        """Flush every still-open event at shutdown — generalises the
        original 'finalize last event' step to N concurrent events."""
        for track_id in list(self.events.keys()):
            event = self.events.pop(track_id)
            self._close_event(event)

    # ==================================================================
    # BACKGROUND SIDE — runs on the single dedicated worker thread.
    # All the slow AI calls live here, fully off the tracking loop.
    # ==================================================================
    def _finalize_worker(self):
        while True:
            job = self._finalize_queue.get()
            try:
                self._finalize_event(job)
            except Exception as e:
                print(f"[EventManager] Finalisation error for event "
                      f"{job.get('event_id')}: {e}")
            finally:
                self._finalize_queue.task_done()

    def _finalize_event(self, job):
        """
        Same 9-step finalisation sequence as the original close_event()
        in pipelines/event_pipeline.py, re-using every one of its
        underlying calls verbatim. The only change is that inputs come
        from this event's own job dict instead of shared module globals,
        which is what makes it safe to run one finalisation after
        another (or, if this were ever widened to a worker pool, in
        parallel) without cross-event interference.
        """
        ev_id      = job["event_id"]
        ev_record  = job["record"]
        ev_output  = job["output_path"]
        ev_start   = job["start_time"]
        best_frame = job["best_frame"]
        best_crop  = job["best_crop"]

        ev_end      = time.time()
        ev_duration = int(ev_end - ev_start)
        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # STEP 1: snapshot
        snapshot_path = f"{SMART_FRAMES_DIR}/event_{ev_id}.jpg"
        write_ok      = cv2.imwrite(snapshot_path, best_frame)
        if not write_ok:
            print(f"[WARNING] Snapshot write failed for Event {ev_id}")

        # STEP 2: immediate Telegram alert (fire-and-forget, own thread)
        if write_ok:
            send_telegram_alert(snapshot_path, "⏳ Processing event summary...",
                                 ev_duration, timestamp, ev_id)

        # STEP 3: VideoMAE — sparse "smart frame" selection. Only these
        # ~10 sampled frames (never every raw frame) are ever sent to
        # the AI, satisfying "don't send every frame to the AI".
        #
        # GPU MEMORY OPTIMISATION: VideoMAE is loaded on-demand right
        # here (instead of once at startup) and fully released again
        # immediately below, BEFORE Qwen is loaded. This guarantees
        # VideoMAE and Qwen are never resident in memory "at the same
        # time" — VideoMAE already ran on CPU in this codebase, so this
        # step mainly frees system RAM, but the load→use→release shape
        # is kept identical to Qwen's so the pattern holds regardless
        # of which device either model ends up running on.
        load_videomae()
        smart_frames = extract_smart_frames(ev_output, ev_id)
        unload_videomae()
        # Force VideoMAE's tensors/model to actually be freed before we
        # (maybe) load Qwen next, and clear CUDA's cache so Qwen gets
        # the largest possible contiguous block of free GPU memory.
        gc.collect()
        torch.cuda.empty_cache()

        # STEP 4: summary — provider (Qwen vs OpenAI) decided ENTIRELY
        # by services/summary_router.py, based on USE_OPENAI /
        # ENABLE_FACE_RECOGNITION in .env. event_manager never decides
        # the provider itself, it only asks the router for a summary
        # and reacts to which provider it got back.
        #
        # ROUTING FIX: Qwen must NEVER load when USE_OPENAI=True and
        # ENABLE_FACE_RECOGNITION=False (pure OpenAI testing mode) —
        # previously Qwen was pre-loaded unconditionally here and then
        # reused for attribute extraction even after OpenAI had already
        # returned the full structured result, which is not the
        # intended architecture. Qwen is now only pre-loaded when it
        # can actually be selected: USE_OPENAI=False (always Qwen), or
        # ENABLE_FACE_RECOGNITION=True (a known face may still route to
        # Qwen — face_status isn't known until the router runs).
        qwen_may_run = (not USE_OPENAI) or ENABLE_FACE_RECOGNITION
        if qwen_may_run:
            load_qwen()
        try:
            summary, provider, face_status, structured = generate_event_summary(smart_frames, ev_id)
            header = f"{provider.upper()} SUMMARY" + (f" (face: {face_status})" if face_status else "")
            print("\n" + "=" * 50 + f"\n{header}\n" + "=" * 50)
            print(summary)

            # STEP 5: full summary message
            if write_ok:
                tg_send_message(CHAT_ID, f"📋 *Event #{ev_id} Full Report*\n\n{summary[:1000]}")

            # STEP 6: save this event's own best crop (no shared dict needed
            # since best_crop already belongs to only this event)
            person = ev_record["persons"][0]
            if best_crop is not None:
                crop_path = f"{PERSON_CROPS_DIR}/event{ev_id}_person1_{job['reid_id']}.jpg"
                cv2.imwrite(crop_path, best_crop)
                person["crop_image"] = crop_path
            person["last_seen"] = timestamp
            person["frames"]    = smart_frames[:3]

            # STEP 7: attribute extraction. Only ever calls into Qwen
            # when Qwen did NOT already run the summary AND OpenAI did
            # NOT already return a full structured result — i.e. Qwen
            # extraction only ever runs on the Qwen branch. On the
            # OpenAI branch, OpenAI already produced everything
            # (summary + person attributes + movement + objects +
            # actions + keywords) in its one API call, so Qwen must
            # never load a second time just to re-derive attributes
            # OpenAI already gave us.
            if provider == "openai" and structured:
                persons_attrs = _persons_attrs_from_structured(structured)
            else:
                persons_attrs = extract_person_attributes(summary)
        finally:
            # Always release Qwen, even if summarisation/extraction
            # raised, and even if it was only speculatively loaded
            # above (unload_qwen() is a no-op when nothing was loaded)
            # — we never want the 5-6 GB model left dangling on the GPU
            # after a failed event.
            unload_qwen()

        if persons_attrs and isinstance(persons_attrs[0], dict):
            attrs = persons_attrs[0]
            person["appearance"] = attrs.get("appearance")
            person["actions"]    = attrs.get("actions", [])
            person["objects"]    = attrs.get("objects", [])
            person["movement"]   = attrs.get("movement")
            person["waiting"]    = attrs.get("waiting", False)
        if not person["first_seen"]:
            person["first_seen"] = timestamp

        # STEP 8: finalise + persist
        ev_record["summary"]   = summary
        ev_record["duration"]  = ev_duration
        ev_record["timestamp"] = timestamp
        ev_record["snapshot"]  = snapshot_path
        ev_record["video"]     = ev_output

        save_memory_append(ev_record)
        gallery_save()

        print(f"[INFO] Event {ev_id} saved with {len(ev_record['persons'])} person record(s).")

        # ------------------------------------------------------------
        # STEP 9 (Phase 2, additive): mirror this event into SQLite so
        # Telegram/Groq (pipelines/query_pipeline.py) can query it.
        # Never blocks/breaks finalisation — persist_event() catches
        # and logs its own errors. JSON memory (above) remains the
        # source of truth regardless of whether this succeeds.
        # ------------------------------------------------------------
        if provider == "openai" and structured:
            # OpenAI already returned the full rich structure in its
            # one API call — use it as-is.
            from config.settings import MERGED_EVENTS_DIR
            merged_image_path = f"{MERGED_EVENTS_DIR}/event_{ev_id}.jpg"
        else:
            # Qwen branch (or OpenAI branch that somehow returned no
            # structured data) — build an equivalent minimal structure
            # from the attribute-extraction output we already have, so
            # every event is queryable from SQLite regardless of which
            # provider generated it. No extra AI calls are made here.
            structured = build_structured_from_qwen(summary, persons_attrs)
            merged_image_path = None

        db_ok = persist_event(
            event_id=ev_id,
            camera_id=_DEFAULT_CAMERA_ID,
            provider=provider,
            summary_text=summary,
            structured=structured,
            start_time=datetime.fromtimestamp(ev_start).strftime("%Y-%m-%d %H:%M:%S"),
            end_time=timestamp,
            duration_seconds=float(ev_duration),
            video_path=ev_output,
            merged_image_path=merged_image_path,
        )

        telegram_status = "DISPATCHED" if write_ok else "SKIPPED"
        log_block("TELEGRAM", "Alert " + ("Dispatched" if write_ok else "Skipped (no snapshot)"))

        log_completion(
            pipeline_used=provider.upper(),
            summary_ok=bool(summary),
            db_ok=db_ok,
            telegram_status=telegram_status,
            total_seconds=time.time() - ev_end,
        )

        # (no reset_event_state() equivalent needed — this job dict and
        # its TrackEvent are already discarded; there is no shared
        # module-level state left to reset)


# Module-level singleton — main.py imports and uses this directly,
# the same way `pipelines.event_pipeline` was imported as `ep` before.
event_manager = EventManager()
