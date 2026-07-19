"""
pipelines/event_manager.py
===========================
Event Manager — the bridge between the real-time tracking loop and the
(slow) AI pipeline.

SCENE-BASED EVENTS (this revision)
------------------------------------
Previously this module created one independent TrackEvent per Track
ID, so three people interacting in front of the camera produced three
fragmented events/summaries. This revision replaces that with SCENE
EVENTS: a Scene Event represents everything that happens inside the
ROI while at least one tracked person remains inside it. One person,
two persons, three persons, known, unknown, or any combination all
share the SAME Scene Event, the same recording, and get ONE combined
summary of the complete scene — never one summary per person.

    One Scene -> One Event -> Multiple Persons

The AI pipeline itself (YOLO, Tracking, VideoMAE, Qwen, OpenAI, Groq,
Face Recognition, the Router, Retrieval) is completely untouched —
only the event lifecycle/ownership model changed, exactly as
requested. Face Recognition and Tracking both remain PERSON based; the
EVENT is what became SCENE based.

Two independent background workers, same reasoning as before:

  Worker 1 (_join_worker / _process_join) — Real-Time Scene Manager's
    background half. Runs the instant a NEW tracked person enters the
    (possibly already-active) scene: saves their crop, writes/updates
    their `persons` row in SQLite, and fires the Stage 1 Telegram
    alert (🟢 scene started for the first person, ➕ update for every
    person after that) — WITHOUT ever performing Face Recognition or
    reporting any identity (no "Unknown", no "Known Person", no
    name). NEVER waits on Face Recognition, VideoMAE, OpenAI, or Qwen
    — those, and all identity resolution, only ever run on Worker 2,
    once the whole scene has ended (Stage 2).

  Worker 2 (_ai_worker / _process_ai) — AI Processing (Stage 2). Runs
    once per Scene Event, only after the ROI has become completely
    empty (every tracked person has left): VideoMAE-driven smart-frame
    selection over the FULL scene recording (VideoMAE's own output
    picks the 3 frames, not a handcrafted heuristic) -> Face
    Recognition (the ONLY place identity is ever resolved, using each
    participant's best crop across the whole scene) -> Router ->
    Qwen/OpenAI (unchanged, single call per scene, exactly like
    before) -> structured data -> SQLite -> one Telegram "scene
    summary" message covering every participant, with real identities
    -> take the next queued scene.

Real-time side (`update_detections()` / `tick()`) is still called once
per frame from main.py, is still O(active tracks), and still does no
model inference — it only writes a frame to the scene's single shared
recording and tracks per-person timestamps, so live capture/tracking
stays real-time regardless of how much or how little the two workers
have queued up.

NOTE on the Router / Qwen naming (ISSUE 3 — Qwen must become scene
aware): services/summary_router.py's ROUTING DECISION (Qwen vs OpenAI)
is unchanged — it is still made from a single face/recognizer.py check,
exactly as before. What Qwen is TOLD once it is the branch taken has
changed: it used to only ever receive one `person_name`, so a
multi-person scene could only ever be described as if one person were
present. Qwen now receives the Scene Event's COMPLETE participant list
(every known real name, every "Unknown"/"Unknown visitor N" label, and
each participant's join/leave time — see _scene_participants_payload()
below and prompts/summary_prompts.py) and is instructed to describe the
whole interaction in one combined summary. OpenAI's structured contract
already returns a `persons` list and needs no such change.
"""

import cv2
import time
import queue
import gc
import threading
import itertools
from datetime import datetime

import numpy as np
import torch

from config.settings import (
    EVENTS_DIR, PERSON_CROPS_DIR, SMART_FRAMES_DIR, CHAT_ID,
    FRAME_WIDTH, FRAME_HEIGHT,
    REID_GRACE_PERIOD, NO_PERSON_TIMEOUT,
    USE_OPENAI, ENABLE_FACE_RECOGNITION,
)
from memory.event_memory        import empty_event_record, empty_person_record, save_memory_append
from memory.gallery              import gallery_was_recent, gallery_purge_stale, gallery_save
from models.videomae              import (
    extract_smart_frames, load_videomae, unload_videomae,
    extract_smart_frames_fallback,
)
from models.qwen_vl                import load_qwen, unload_qwen
from models                        import model_manager
from utils.debug_artifacts        import EventDebug
#from models.smolvlm               import load_qwen, unload_qwen
from pipelines.summary_pipeline  import extract_person_attributes
from telegram.alerts             import (
    send_scene_started_alert, send_person_joined_alert,
    tg_send_message, _format_telegram_datetime,
)
from utils.event_logger          import log_event_header, log_block, log_completion
from utils.device                import empty_cache

# Single routing decision point for "Qwen vs OpenAI" — see
# services/summary_router.py. event_manager never decides the provider
# itself, it just asks the router for a summary of the whole scene.
from services.summary_router     import generate_event_summary

# Phase 2 — SQLite persistence. Purely additive.
from services.db_writer          import (
    persist_event, persist_initial_event, persist_join_person,
    build_structured_from_qwen,
)

# Worker 2 reads the event back from SQLite (the source of truth)
# before starting AI processing — see EventManager._process_ai().
from pipelines                   import retrieval

# Single default camera id until multi-camera support is introduced —
# kept local to this module so no other file needs to change.
_DEFAULT_CAMERA_ID = "cam_default"


_event_id_counter = itertools.count(0)


def _next_event_id():
    """Monotonically increasing Scene Event id. Only ever called from
    the main tracking thread (inside SceneEvent.__init__), so no lock
    is required."""
    return next(_event_id_counter)


_NOT_VISIBLE = "Not Clearly Visible"


def _persons_attrs_from_structured(structured):
    """
    Build the same list[dict] shape that
    pipelines.summary_pipeline.extract_person_attributes() returns
    (keys: appearance, actions, objects, movement, waiting) directly
    from the OpenAI structured payload — OpenAI's own contract already
    returns one entry per detected person in the scene, so this needs
    no scene-specific change.

    Used ONLY on the OpenAI branch, so JSON memory (person["appearance"]
    etc.) still gets populated even though Qwen never ran for this
    event.
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


class ScenePerson:
    """
    One tracked person's participation inside a Scene Event. A Scene
    Event may hold any combination of one, two, three, known,
    unknown, or unrecognized ScenePersons at once — Tracking/ReID
    identity (track_id/reid_id) is unchanged; only OWNERSHIP of the
    event moved from "one event per person" to "one entry inside a
    shared Scene Event".
    """

    def __init__(self, track_id, reid_id, person_index, event_id, now):
        self.track_id      = track_id
        self.reid_id        = reid_id
        self.person_index   = person_index      # 1-based -> "personN"
        self.event_id        = event_id

        self.joined_at       = now
        self.last_seen        = now

        self.best_frame        = None
        self.best_score        = 0
        self.best_crop          = None
        self.best_crop_score    = 0
        self.crop_path           = None

        # Filled in asynchronously by EventManager._process_join() once
        # Face Recognition completes for THIS specific person — never
        # blocks the real-time loop.
        self.name        = None
        self.confidence  = None

        self.first_seen_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_seen_str  = self.first_seen_str

    # ------------------------------------------------------------------
    def update(self, frame, box, confidence, reid_id, now):
        """Called every frame this Track ID is detected in the ROI.
        Updates only THIS person's own best-frame/best-crop trackers —
        never touches any other participant's state or the scene's
        shared recording (SceneEvent.write_frame() handles that once
        per frame, regardless of how many people are in it)."""
        x1, y1, x2, y2 = box
        self.last_seen     = now
        self.last_seen_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if reid_id is not None:
            self.reid_id = reid_id

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

    # ------------------------------------------------------------------
    def elapsed_since_seen(self, now):
        return now - self.last_seen

    def grace_period(self):
        """Same grace-period rule as before (REID_GRACE_PERIOD vs
        NO_PERSON_TIMEOUT), scoped to this person's own identity."""
        return REID_GRACE_PERIOD if self.reid_id else NO_PERSON_TIMEOUT

    def to_person_record(self):
        pid = f"event{self.event_id}_person{self.person_index}"
        rec = empty_person_record(pid, self.track_id, self.event_id)
        rec["reid_id"]    = self.reid_id
        rec["first_seen"] = self.first_seen_str
        rec["last_seen"]  = self.last_seen_str
        rec["crop_image"] = self.crop_path
        return rec


class SceneEvent:
    """
    One Scene Event: everything that happens inside the ROI while at
    least one tracked person remains inside it. Owns ONE shared
    recording/timeline for however many people pass through it,
    replacing the old one-recording-per-Track-ID model.
    """

    def __init__(self, fps):
        self.event_id = _next_event_id()
        self.fps      = fps
        self.start_time = time.time()

        self.output_path = f"{EVENTS_DIR}/event_{self.event_id}.mp4"
        fourcc            = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # track_id -> ScenePerson, for every person who has EVER been
        # part of this scene — kept even after they individually leave,
        # since the scene stays open until ALL of them have.
        self.participants = {}
        # track_ids currently still inside the ROI right now.
        self.active_track_ids = set()

        # ISSUE 3 (reid-aware de-duplication): `self.participants` above
        # is keyed by raw Track ID, but Tracking can briefly lose and
        # reassign a NEW Track ID to the SAME physical person (occlusion,
        # a missed frame, walking behind something). Without this, that
        # shows up as a second ScenePerson — a second Worker 1 job, a
        # second Telegram "joined" alert, and a second (possibly
        # "Unknown") participant for someone who was already recognized
        # a moment earlier. `reid_index` maps ReID identity -> the ONE
        # ScenePerson already tracking that physical person, so a
        # reappearing Track ID gets aliased onto the existing
        # participant instead of minting a new one. `person_list` is the
        # de-duplicated, one-entry-per-real-person ordering used
        # everywhere a unique participant list is needed (self.
        # participants.values() cannot be used for that once aliasing is
        # in play, since several keys can point at the same object).
        self.reid_index = {}
        self.person_list = []

        self.timeline = [{
            "t": datetime.now().strftime("%H:%M:%S"),
            "note": "scene started",
        }]

        # ISSUE 4/5 (smart frame selection): index-aligned with every
        # frame actually written to self.output_path — one float per
        # write_frame() call, using data update_detections() already
        # computes for free (no extra model inference). 0.0 marks a
        # frame with no confirmed detection (e.g. a grace-period gap) so
        # extract_smart_frames() can prefer real, on-target frames over
        # empty ones without ever re-decoding/re-detecting anything.
        self.frame_quality = []

        print(f"[SCENE START] Event {self.event_id}")
        log_event_header(self.event_id)
        log_block("YOLO", "Detecting persons...", "Scene created.")

    # ------------------------------------------------------------------
    def add_person(self, track_id, reid_id, now):
        person = ScenePerson(track_id, reid_id, len(self.person_list) + 1, self.event_id, now)
        self.participants[track_id] = person
        self.active_track_ids.add(track_id)
        self.person_list.append(person)
        if reid_id is not None:
            self.reid_index.setdefault(reid_id, person)
        log_block("TRACKING", f"Track ID : {track_id} (person {person.person_index})")
        return person

    def mark_left(self, track_id):
        self.active_track_ids.discard(track_id)
        self.timeline.append({
            "t": datetime.now().strftime("%H:%M:%S"),
            "note": f"track {track_id} left ROI",
        })

    def is_empty(self):
        return len(self.active_track_ids) == 0

    def write_frame(self, frame, quality=0.0):
        if self.video_writer is not None:
            self.video_writer.write(frame)
            self.frame_quality.append(quality)

    # ------------------------------------------------------------------
    def to_finalize_job(self):
        """Release the scene's recording and package everything Worker
        2 needs into a plain dict — the worker never touches `self`
        directly, so there's no shared mutable state between the
        tracking thread and the AI-processing thread."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.timeline.append({
            "t": datetime.now().strftime("%H:%M:%S"),
            "note": "ROI empty / scene closed",
        })

        record = empty_event_record(
            self.event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        record["timeline"] = self.timeline
        record["persons"]  = [p.to_person_record() for p in self.person_list]

        # Best frame across every participant, used for the scene
        # snapshot / VideoMAE seed.
        best_frame, best_score = None, -1
        for p in self.person_list:
            if p.best_frame is not None and p.best_score > best_score:
                best_frame, best_score = p.best_frame, p.best_score

        return {
            "event_id":     self.event_id,
            "output_path":  self.output_path,
            "start_time":   self.start_time,
            "record":       record,
            "best_frame":   best_frame,
            "participants": list(self.person_list),
            "frame_quality": list(self.frame_quality),
        }


class EventManager:
    """
    Owns the single currently-active SceneEvent (a Scene Event
    represents the whole ROI, not one Track ID) plus TWO independent
    background workers:

      Worker 1 (_join_worker)  — fires once per NEW person entering the
        (possibly already-active) scene: crop -> Face Recognition ->
        SQLite (initial row for the first person, an added participant
        row for every person after that) -> immediate Telegram alert.
        NEVER waits on VideoMAE/OpenAI/Qwen.

      Worker 2 (_ai_worker)    — fires once per Scene Event, only after
        the ROI becomes completely empty: VideoMAE -> Router ->
        Qwen/OpenAI -> structured data -> SQLite -> one Telegram scene
        summary -> take the next queued scene.

    main.py only ever needs two calls per frame:
      event_manager.update_detections(frame, detections, fps, now)
      event_manager.tick(now)
    Both are cheap and non-blocking; all AI work happens off the
    tracking thread, on one of the two workers above.
    """

    def __init__(self):
        self.active_scene = None   # SceneEvent | None — one ROI, one scene, at a time

        # Worker 1's input — one job per NEW person joining the scene.
        self._join_queue = queue.Queue()

        # Worker 2's input — one job per CLOSED scene (queue.Queue is
        # strictly FIFO, which is all "process scenes in order" needs).
        self._ai_queue = queue.Queue()

        # In-memory-only status tracker (ACTIVE / PROCESSING / COMPLETED
        # / FAILED) — deliberately not a SQLite column, purely an
        # execution-flow/observability concern.
        self._status_lock  = threading.Lock()
        self._event_status = {}

        self._join_worker_thread = threading.Thread(
            target=self._join_worker, daemon=True
        )
        self._join_worker_thread.start()

        self._ai_worker_thread = threading.Thread(
            target=self._ai_worker, daemon=True
        )
        self._ai_worker_thread.start()

    # ==================================================================
    # REAL-TIME SIDE — runs on the main tracking thread. No AI, no
    # blocking I/O beyond writing this frame to the scene's recording.
    # ==================================================================
    def update_detections(self, frame, detections, fps, now):
        """
        detections: list of (x1, y1, x2, y2, track_id, confidence, reid_id)
        for every tracked person currently inside the ROI this frame.

        If the ROI is empty (detections is falsy), there is nothing to
        do here — tick() below is what eventually closes an active
        scene once every participant has actually left.

        Otherwise: open a brand-new Scene Event the instant the FIRST
        tracked person enters an empty ROI, or fold a NEW Track ID into
        the ALREADY-active scene instead of ever starting a second one.

        BUGFIX (frame recording / smart frame pipeline): a frame with no
        CONFIRMED detection (occlusion, motion blur, a momentary
        confidence dip below MIN_CONFIDENCE, etc.) does NOT mean the
        scene has ended — tick()'s grace-period logic is the only thing
        that ever closes a scene. Previously this function returned
        immediately whenever `detections` was empty, which silently
        skipped writing that camera frame to the scene's recording. Over
        a multi-second scene that gap adds up to a video with far fewer
        frames than the scene's real (wall-clock) duration, which is
        exactly why Smart Frame Selection later reports "Not enough
        frames" even though the event duration itself is correct. The
        scene's shared recording must keep capturing every camera frame
        for as long as the scene stays open, regardless of whether this
        exact frame produced a confirmed detection — never start a NEW
        scene from an empty frame though, only keep an already-active
        one recording.
        """
        if not detections:
            if self.active_scene is not None:
                self.active_scene.write_frame(frame)
            return

        if self.active_scene is None:
            self.active_scene = SceneEvent(fps)

        scene = self.active_scene

        frame_quality = 0.0
        for (x1, y1, x2, y2, track_id, confidence, reid_id) in detections:
            person = scene.participants.get(track_id)

            # ISSUE 3 (reid-aware de-duplication): this exact Track ID
            # hasn't been seen in THIS scene before, but if ReID already
            # recognizes it as a physical person who IS already a
            # participant (their track briefly dropped and got
            # reassigned a new id), alias this Track ID onto that same
            # ScenePerson instead of starting a brand-new one — avoids a
            # duplicate Worker 1 job (duplicate Telegram "joined" alert,
            # a second possibly-"Unknown" participant for someone
            # already recognized).
            if person is None and reid_id is not None:
                person = scene.reid_index.get(reid_id)
                if person is not None:
                    scene.participants[track_id] = person

            is_new = person is None
            if is_new:
                person = scene.add_person(track_id, reid_id, now)
            else:
                # Handles a track briefly reappearing before its grace
                # period actually closed it out of the active set.
                scene.active_track_ids.add(track_id)

            person.update(frame, (x1, y1, x2, y2), confidence, reid_id, now)

            # A reid_id can arrive a few frames AFTER a person first
            # joins (gallery matching needs a moment) — register it the
            # instant it's known so a later dropped/reassigned track for
            # this same person can still be aliased above.
            if reid_id is not None:
                scene.reid_index.setdefault(reid_id, person)

            if is_new:
                self._queue_join(scene, person, frame, (x1, y1, x2, y2))

            # ISSUE 4/5 (smart frame selection): cheap, inference-free
            # per-frame quality signal — area * confidence, same values
            # already computed above for best-crop tracking — the
            # largest, highest-confidence detection wins this frame.
            area  = (x2 - x1) * (y2 - y1)
            score = area * confidence
            if score > frame_quality:
                frame_quality = score

        # ONE shared recording for the whole scene, written once per
        # frame regardless of how many people are in it.
        scene.write_frame(frame, frame_quality)

    def _queue_join(self, scene, person, frame, box):
        """Hand a brand-new participant off to Worker 1 (background
        thread) for Face Recognition + SQLite + the immediate Telegram
        alert — never done inline on the tracking thread."""
        x1, y1, x2, y2 = box
        H, W = frame.shape[:2]
        px1, py1 = max(0, x1 - 10), max(0, y1 - 10)
        px2, py2 = min(W, x2 + 10), min(H, y2 + 10)
        crop_region = frame[py1:py2, px1:px2]
        crop = crop_region.copy() if crop_region.size else None

        is_first = (len(scene.participants) == 1)   # this new person is the only one so far

        self._join_queue.put({
            "event_id":    scene.event_id,
            "person":      person,
            "crop":        crop,
            "is_first":    is_first,
            "start_time":  scene.start_time,
            "output_path": scene.output_path,
            # ISSUE 1 (two-stage Telegram alerts): how many distinct
            # participants this scene has ever had as of THIS join —
            # used only for the "Persons Detected : N" line in the
            # Stage 1 alert (see telegram/alerts.py). Never used for
            # identity — Stage 1 never resolves who anyone is.
            "persons_detected": len(scene.participants),
        })

    def tick(self, now):
        """
        Remove any participant who hasn't been seen for longer than
        their own grace period from the active set. The Scene Event
        itself only ends once NO tracked persons remain inside the
        ROI — one person leaving while others stay never closes it.
        """
        scene = self.active_scene
        if scene is None:
            return

        for track_id in list(scene.active_track_ids):
            person  = scene.participants[track_id]
            elapsed = person.elapsed_since_seen(now)
            grace   = person.grace_period()

            if elapsed <= grace:
                continue

            # Same "identity briefly reappeared" leeway as before, now
            # scoped to this person's own reid_id.
            if person.reid_id and gallery_was_recent(person.reid_id) and elapsed < grace * 2:
                continue

            scene.mark_left(track_id)
            print(f"[SCENE] Track {track_id} left Event {scene.event_id} "
                  f"({len(scene.active_track_ids)} still inside).")

        if scene.is_empty():
            self.active_scene = None
            self._close_scene(scene)
            gallery_purge_stale()

    def _close_scene(self, scene):
        print(f"[SCENE END] Event {scene.event_id} (ROI empty)")
        job = scene.to_finalize_job()
        if job["best_frame"] is None:
            print(f"[WARNING] Scene Event {job['event_id']} has no best_frame — "
                  f"skipping AI processing.")
            return
        self._ai_queue.put(job)   # hand-off is non-blocking

    def close_all(self):
        """Flush the still-open scene (if any) at shutdown."""
        if self.active_scene is not None:
            scene = self.active_scene
            self.active_scene = None
            self._close_scene(scene)

    # ------------------------------------------------------------------
    # Status tracking (in-memory only — see __init__ note above)
    # ------------------------------------------------------------------
    def _set_status(self, event_id, status):
        with self._status_lock:
            self._event_status[event_id] = status

    def get_event_status(self, event_id):
        """ACTIVE -> PROCESSING -> COMPLETED (or FAILED). Returns None
        for an event_id Worker 1 hasn't created yet."""
        with self._status_lock:
            return self._event_status.get(event_id)

    # ==================================================================
    # WORKER 1 — Real-Time Scene Manager's background half. One job per
    # NEW person joining the (possibly already-active) scene: SQLite ->
    # immediate Stage 1 Telegram alert. NEVER touches Face Recognition,
    # OpenAI/Qwen/Groq/VideoMAE — see _process_join(). Identity is
    # resolved ONLY on Worker 2, once the whole scene has ended (Stage
    # 2 — see _process_ai() below).
    # ==================================================================

    def _join_worker(self):
        while True:
            job = self._join_queue.get()
            try:
                self._process_join(job)
            except Exception as e:
                print(f"[EventManager] Join processing error for "
                      f"event {job.get('event_id')}: {e}")
            finally:
                self._join_queue.task_done()

    def _process_join(self, job):
        """
        Worker 1's entire job for ONE new participant — snapshot their
        crop, persist them to SQLite, and IMMEDIATELY fire the correct
        Stage 1 Telegram alert (🟢 scene started for the first person
        in the scene, ➕ update for everyone after). This is Stage 1 of
        the two-stage alert system: it NEVER performs Face Recognition
        and NEVER waits for one — identity is deliberately left
        unresolved (person.name stays None) here.

        Why: the only frame(s) available this early in an event are
        often a back view, a side view, a partial body, or motion
        blur — running recognition on that and reporting the result
        (even as "Unknown") is exactly what produced false "UNKNOWN
        PERSON DETECTED" alerts before. Stage 2 (EventManager.
        _process_ai(), after the whole scene has ended and VideoMAE
        has selected the representative Smart Frames) is now the ONLY
        place identity is ever resolved and reported, using the best
        face available across the participant's entire time in the
        ROI instead of whatever the first second or two happened to
        capture.
        """
        ev_id            = job["event_id"]
        person           = job["person"]
        is_first         = job["is_first"]
        persons_detected = job["persons_detected"]

        self._set_status(ev_id, "ACTIVE")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Use whatever the main thread has produced by now — almost
        # always better than the single frame available the instant
        # this person was first seen (job["crop"]), which is only ever
        # a fallback for the (rare) case best_crop never got set. No
        # waiting/polling here (unlike before) — Stage 1 must fire
        # immediately, and it needs A crop to attach, not specifically
        # a face-visible one.
        crop = person.best_crop if person.best_crop is not None else job["crop"]

        # STEP 1: this person's crop snapshot.
        crop_path = f"{PERSON_CROPS_DIR}/event{ev_id}_person{person.person_index}_{person.reid_id}.jpg"
        write_ok  = bool(crop is not None and crop.size and cv2.imwrite(crop_path, crop))
        if write_ok:
            person.crop_path = crop_path
        else:
            print(f"[WARNING] No usable crop for event {ev_id} "
                  f"person{person.person_index} — skipping their alert.")

        # STEP 2: identity is intentionally left unresolved here.
        # person.name / person.confidence stay None (their __init__
        # default) — Stage 2 (_process_ai() below) is the only place
        # they are ever set, after VideoMAE has selected the Smart
        # Frames for the complete scene recording.

        # STEP 3: Create Event / Add Participant. SQLite remains the
        # running source of truth for exactly who is currently part of
        # this Scene Event. name/confidence are None until Stage 2
        # resolves them.
        if is_first:
            persist_initial_event(
                event_id=ev_id,
                camera_id=_DEFAULT_CAMERA_ID,
                start_time=datetime.fromtimestamp(job["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                video_path=job["output_path"],
                persons=[{"name": None, "confidence": None}],
            )
        else:
            persist_join_person(
                event_id=ev_id,
                camera_id=_DEFAULT_CAMERA_ID,
                person_name=None,
                confidence=None,
            )

        # STEP 4: Immediate Telegram Alert — HIGH PRIORITY, Stage 1.
        # Fires the instant this person's crop/SQLite write are done,
        # well before the eventual scene-wide AI summary. Never reports
        # an identity — see the docstring above. The Stage 2 scene
        # summary always arrives afterward as its own, separate
        # message (see Worker 2 / _process_ai() below).
        if write_ok:
            if is_first:
                send_scene_started_alert(crop_path, ev_id, timestamp, persons_detected)
            else:
                send_person_joined_alert(crop_path, ev_id, timestamp, persons_detected)

        log_block(
            "TELEGRAM",
            ("Scene Started Alert " if is_first else "Person Joined Alert ") +
            ("Dispatched" if write_ok else "Skipped (no crop)")
        )

    # ==================================================================
    # WORKER 2 — AI Processing. Fires once per CLOSED Scene Event:
    # VideoMAE -> Router -> Qwen/OpenAI -> structured data -> SQLite ->
    # one Telegram scene summary -> take the next queued scene. Reuses

    # every existing pipeline call verbatim — no routing/model changes
    # live here.
    # ==================================================================
    def _ai_worker(self):
        while True:
            job = self._ai_queue.get()
            ev_id = job.get("event_id")
            try:
                self._set_status(ev_id, "PROCESSING")
                self._process_ai(job)
                self._set_status(ev_id, "COMPLETED")
            except Exception as e:
                self._set_status(ev_id, "FAILED")
                print(f"[EventManager] AI processing error for scene {ev_id}: {e}")
            finally:
                self._ai_queue.task_done()

    def _process_ai(self, job):
        """
        Worker 2's entire job for one Scene Event — VideoMAE smart-frame
        selection over the FULL scene recording, then the unchanged
        Router -> Qwen/OpenAI -> attribute extraction -> SQLite ->
        Telegram summary pipeline, describing the COMPLETE SCENE rather
        than any one participant.
        """
        ev_id        = job["event_id"]
        ev_record    = job["record"]
        ev_output    = job["output_path"]
        ev_start     = job["start_time"]
        participants = job["participants"]

        ev_duration = int(time.time() - ev_start)
        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ev_end      = time.time()

        # Retrieve this event back from SQLite before starting AI
        # processing — confirms Worker 1's initial row is really there
        # (SQLite is the source of truth), without ever blocking AI
        # processing on it.
        try:
            existing = retrieval.get_event_by_id(str(ev_id))
            if existing is None:
                print(f"[EventManager] WARNING — Scene Event {ev_id} not found "
                      f"in SQLite at AI-processing time (initial save may "
                      f"have failed); continuing anyway.")
        except Exception as e:
            print(f"[EventManager] Could not retrieve Scene Event {ev_id} from "
                  f"SQLite before AI processing: {e}")

        # STEP 1: scene snapshot — best frame across every participant.
        snapshot_path = f"{SMART_FRAMES_DIR}/event_{ev_id}.jpg"
        write_ok      = cv2.imwrite(snapshot_path, job["best_frame"])
        if not write_ok:
            print(f"[WARNING] Snapshot write failed for Scene Event {ev_id}")

        # DEBUGGING REQUIREMENTS — one debug/event_<id>/ folder per
        # event (original_video.mp4, all_frames/, selected_frames/,
        # videomae_scores.csv, frame_mapping.csv, processing_log.txt).
        # Every write in here is best-effort — see utils/debug_artifacts.py
        # — a debug-write failure must never affect real processing.
        debug = EventDebug(ev_id)
        debug.log(f"[INFO]\nStarting AI processing for Scene Event {ev_id} "
                   f"(duration ~{ev_duration}s, {len(participants)} participant(s)).")

        # STEP 2: VideoMAE — sparse "smart frame" selection over the
        # ENTIRE scene recording (every participant included), fully
        # loaded/used/released here — Worker 2 never touches VideoMAE
        # outside this window. The combined score (see models/videomae.py)
        # now drives WHICH 3 frames get selected — `frame_quality`
        # (index-aligned per-frame scores computed for free during
        # capture — see SceneEvent.write_frame()/update_detections())
        # is still passed through, but only as a validity gate (skip a
        # frame with no confirmed detection at all), never as the thing
        # that picks the winner.
        #
        # ISSUE 2 — ROBUST MODEL LOADING: VideoMAE is loaded through
        # model_manager.safe_load(), which NEVER raises — a failed load
        # (model not found, corrupt checkpoint, OOM, ...) is logged in
        # full and marked FAILED, and Smart Frame Selection falls back
        # to extract_smart_frames_fallback() (sharpness + detection-
        # quality only, no VideoMAE) instead of taking down Face
        # Recognition / AI Summary / the Telegram alert for this event.
        videomae_ready = model_manager.safe_load("videomae", load_videomae)
        try:
            if videomae_ready:
                smart_frames = model_manager.run_stage(
                    "videomae_extract",
                    extract_smart_frames,
                    ev_output, ev_id,
                    quality_scores=job.get("frame_quality"), debug=debug,
                    fallback=lambda: extract_smart_frames_fallback(
                        ev_output, ev_id,
                        quality_scores=job.get("frame_quality"), debug=debug,
                    ),
                )
            else:
                smart_frames = extract_smart_frames_fallback(
                    ev_output, ev_id,
                    quality_scores=job.get("frame_quality"), debug=debug,
                )
        finally:
            if videomae_ready:
                model_manager.safe_unload("videomae", unload_videomae)
            gc.collect()
            empty_cache()

        # STEP 2.5 (ISSUE 1 — Face Recognition must use the best Smart
        # Frame): Worker 1 (Stage 1) never attempts Face Recognition at
        # all — every participant's identity is still unresolved
        # (person.name is None) at this point. This is the FIRST and
        # ONLY time identity is resolved, and it happens strictly AFTER
        # VideoMAE has finished selecting the Smart Frames above.
        #
        # Previously this ran Face Recognition on person.best_crop —
        # the largest/highest-confidence frame captured DURING
        # real-time tracking. That crop can easily be a side face, a
        # back view, or a partial face (whatever the largest/most
        # confident detection happened to look like), even though
        # VideoMAE went on to select much clearer Smart Frames later.
        # Now: every detected face across the 3 Smart Frames is pooled
        # into one candidate list, and each participant is matched to
        # THEIR OWN face within that pool using their best_crop only as
        # an identity ANCHOR (to tell participants apart in a
        # multi-person scene) — the actual recognition always runs on
        # the clearer Smart Frame face, never on the anchor crop
        # itself, whenever a Smart Frame face is available.
        face_recognition_active = USE_OPENAI and ENABLE_FACE_RECOGNITION
        # ISSUE 2 — this ENTIRE stage is wrapped so any failure here
        # (InsightFace not loaded, a corrupt face database, an
        # unexpected detector error, ...) is logged in full and simply
        # leaves participants unresolved (p.name stays None), instead
        # of crashing _process_ai() and losing AI Summary + the
        # Telegram alert for the whole event.
        if face_recognition_active:
            try:
                from face.recognizer import has_usable_face, detect_faces, match_embedding

                unresolved = [p for p in participants if p.name is None]

                # Face Detection on every selected Smart Frame, ONCE per
                # event (not once per participant) — every face InsightFace
                # finds across the 3 VideoMAE Smart Frames, pooled together.
                smart_face_candidates = []
                for frame_path in smart_frames:
                    smart_face_candidates.extend(detect_faces(frame_path))

                for p in unresolved:
                    # Identity ANCHOR: this participant's own best_crop,
                    # used only to tell THEM apart from other participants
                    # in the same Smart Frames — never used for the final
                    # recognition call when a Smart Frame face is found.
                    anchor_embedding = None
                    crop = p.best_crop
                    if crop is not None and crop.size and has_usable_face(crop):
                        anchor_faces = detect_faces(crop)
                        if anchor_faces:
                            anchor_embedding = max(anchor_faces, key=lambda f: f["area"])["embedding"]

                    chosen = None
                    if smart_face_candidates:
                        if anchor_embedding is not None:
                            # The Smart Frame face that is the SAME physical
                            # person as this participant's anchor (highest
                            # embedding similarity) — "the clearest visible
                            # face" for THIS participant specifically.
                            best_i, best_sim = None, -1.0
                            for i, cand in enumerate(smart_face_candidates):
                                sim = float(np.dot(anchor_embedding, cand["embedding"]))
                                if sim > best_sim:
                                    best_i, best_sim = i, sim
                            chosen = smart_face_candidates.pop(best_i)
                        elif len(unresolved) == 1:
                            # Only one unresolved participant and no usable
                            # anchor (their best_crop never showed a face at
                            # all) — unambiguous: the single largest
                            # remaining Smart Frame face must be them.
                            best_i = max(
                                range(len(smart_face_candidates)),
                                key=lambda i: smart_face_candidates[i]["area"],
                            )
                            chosen = smart_face_candidates.pop(best_i)

                    if chosen is not None:
                        name, conf = match_embedding(chosen["embedding"])
                    elif anchor_embedding is not None:
                        # No Smart Frame face available/left to match this
                        # participant to (e.g. they had already left before
                        # any Smart Frame was captured) — fall back to
                        # their own best_crop rather than reporting nothing.
                        name, conf = match_embedding(anchor_embedding)
                    else:
                        continue

                    if name not in (None, "Unknown"):
                        p.name, p.confidence = name, conf
                        print(f"[EventManager] Scene {ev_id} person{p.person_index} "
                              f"identified as {name} from Smart Frame.")
            except Exception as e:
                model_manager.mark_failed("face_recognition", e)
                print(f"[ERROR]\nFace Recognition failed for event {ev_id}.\n"
                      f"Reason:\n{type(e).__name__}: {e}")
                debug.log(f"[ERROR]\nFace Recognition failed.\nReason:\n{type(e).__name__}: {e}")
                # Participants simply stay unresolved (name=None) —
                # AI Summary/Telegram/DB below still proceed normally.

        # ISSUE 3 — Qwen must become scene aware: the COMPLETE
        # participant list (every known real name, every unrecognized
        # participant, and each one's join/leave time), index-aligned
        # with ev_record["persons"]/persons_attrs below (both are built
        # from this exact same self.person_list order in
        # SceneEvent.to_finalize_job()). Forwarded to the router so
        # Qwen can describe the whole scene, not just the first known
        # participant.
        participants_payload = [
            {
                "name":       p.name,
                "track_id":   p.track_id,
                "joined_at":  p.first_seen_str,
                "left_at":    p.last_seen_str,
                "confidence": p.confidence,
            }
            for p in participants
        ]

        # Kept for the legacy Qwen-branch db_writer fallback signature
        # (single primary name) — build_structured_from_qwen() now
        # prefers `participants_payload` (below) when it's supplied,
        # so this pair is effectively unused on that path, but stays
        # accurate in case anything else still reads it.
        known_participants = [p for p in participants if p.name]
        primary_known_name = known_participants[0].name if known_participants else None
        primary_known_conf = known_participants[0].confidence if known_participants else None

        # ROUTING (unchanged): Qwen must never load when
        # USE_OPENAI=True and ENABLE_FACE_RECOGNITION=False.
        qwen_may_run = (not USE_OPENAI) or ENABLE_FACE_RECOGNITION
        if qwen_may_run:
            # ISSUE 2: a Qwen load failure is logged and marked FAILED
            # but never raises — generate_event_summary() below is
            # still attempted (it may route to OpenAI instead, or
            # surface its own clear error) rather than crashing here.
            model_manager.safe_load("qwen", load_qwen)
        try:
            summary, provider, face_result, structured = model_manager.run_stage(
                "summary_generation",
                generate_event_summary,
                smart_frames, ev_id, participants=participants_payload,
                fallback=lambda: (
                    "AI summary unavailable for this event (summary "
                    "generation failed — see logs).",
                    "unavailable", None, None,
                ),
            )
            header = f"{provider.upper()} SCENE SUMMARY" + (
                f" (person: {face_result['name']}, {face_result['confidence']:.1f}%)"
                if face_result else ""
            )
            print("\n" + "=" * 50 + f"\n{header}\n" + "=" * 50)
            print(summary)

            if provider == "openai" and structured:
                persons_attrs = _persons_attrs_from_structured(structured)
            elif provider == "unavailable":
                persons_attrs = []
            else:
                persons_attrs = model_manager.run_stage(
                    "attribute_extraction", extract_person_attributes, summary,
                    fallback=[],
                )
        finally:
            # Always release Qwen, even if summarisation/extraction
            # raised, and even if it was only speculatively loaded.
            if qwen_may_run:
                model_manager.safe_unload("qwen", unload_qwen)

        # Apply extracted attributes onto every participant record —
        # index-aligned where possible, falling back to the first
        # extracted entry (same "best effort" mapping the single-
        # person code already relied on).
        for i, person_rec in enumerate(ev_record["persons"]):
            attrs = None
            if persons_attrs:
                attrs = persons_attrs[i] if i < len(persons_attrs) else persons_attrs[0]
            if isinstance(attrs, dict):
                person_rec["appearance"] = attrs.get("appearance")
                person_rec["actions"]    = attrs.get("actions", [])
                person_rec["objects"]    = attrs.get("objects", [])
                person_rec["movement"]   = attrs.get("movement")
                person_rec["waiting"]    = attrs.get("waiting", False)

        # AI scene-summary Telegram message — always a separate
        # message from every immediate join alert already sent.
        participant_names = [p.name or "Unknown" for p in participants]
        event_date_str, event_time_str = _format_telegram_datetime(timestamp)

        if write_ok:
            bar = "=" * 50
            header_lines = [bar, f"EVENT #{ev_id}"]
            header_lines.append("Persons : " + ", ".join(participant_names))
            header_lines.append(f"Date    : {event_date_str}")
            header_lines.append(f"Time    : {event_time_str}")
            header_lines.append(bar)
            header_block = "\n".join(header_lines)
            tg_send_message(CHAT_ID, f"{header_block}\n\n{summary[:1000]}")

        # finalise + persist JSON memory (unchanged — still the source
        # of truth for the JSON record, SQLite below remains its
        # additive, queryable mirror).
        ev_record["summary"]   = summary
        ev_record["duration"]  = ev_duration
        ev_record["timestamp"] = timestamp
        ev_record["snapshot"]  = snapshot_path
        ev_record["video"]     = ev_output

        save_memory_append(ev_record)
        gallery_save()

        print(f"[INFO] Scene Event {ev_id} saved with "
              f"{len(ev_record['persons'])} participant(s): {', '.join(participant_names)}.")

        # ------------------------------------------------------------
        # Phase 2 (additive): replace Worker 1's initial SQLite row
        # with the finished scene summary + structured data (every
        # participant included), same event_id.
        # ------------------------------------------------------------
        if provider == "openai" and structured:
            from config.settings import MERGED_EVENTS_DIR
            merged_image_path = f"{MERGED_EVENTS_DIR}/event_{ev_id}.jpg"
        else:
            structured = build_structured_from_qwen(
                summary, persons_attrs,
                participants=participants_payload,
                person_name=primary_known_name,
                confidence=primary_known_conf,
            )
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
        log_block("TELEGRAM", "Scene Summary " + ("Dispatched" if write_ok else "Skipped (no snapshot)"))

        log_completion(
            pipeline_used=provider.upper(),
            summary_ok=bool(summary),
            db_ok=db_ok,
            telegram_status=telegram_status,
            total_seconds=time.time() - ev_end,
        )

        # ISSUE 2 — one-line-per-model status summary for this event
        # (LOADED / FAILED + reason for every model that was touched:
        # videomae, qwen, face_recognition, summary_generation, ...),
        # printed to stdout AND persisted into this event's
        # processing_log.txt for later debugging.
        model_manager.log_report(ev_id)
        debug.log("[INFO]\nFinal model status:\n" + model_manager.report())
        debug.log(f"[INFO]\nEvent {ev_id} finished. Summary: {'OK' if summary else 'UNAVAILABLE'}, "
                   f"DB: {'OK' if db_ok else 'FAILED'}, Telegram: {telegram_status}.")
        debug.flush_log()

        # (no reset_event_state() equivalent needed — this job dict and
        # its SceneEvent are already discarded; there is no shared
        # module-level state left to reset)


# Module-level singleton — main.py imports and uses this directly.
event_manager = EventManager()