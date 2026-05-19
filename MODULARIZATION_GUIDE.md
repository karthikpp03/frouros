# CCTV v4 — Modularization Guide
## Production-Grade Multi-File Project Structure

---

## 1. Final Folder Structure

```
cctv_v4/
├── src/
│   ├── main.py                          ← orchestration + main loop
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                  ← ALL constants, credentials, paths, thresholds
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py             ← YOLO load + run_tracking()
│   │   ├── qwen_vl.py                   ← Qwen2.5-VL-7B load + _qwen_infer()
│   │   ├── videomae.py                  ← VideoMAE load + extract_smart_frames()
│   │   ├── reid.py                      ← FastReID/OSNet/ResNet18 chain + reid_fn
│   │   └── groq_query_engine.py         ← Groq client + _llama_infer()
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── event_memory.py              ← JSON persistence, empty record builders
│   │   ├── faiss_index.py               ← FAISS index state + search/add/rebuild
│   │   └── gallery.py                   ← gallery_data state + match/update/purge
│   │
│   ├── telegram/
│   │   ├── __init__.py
│   │   ├── alerts.py                    ← tg helpers + send_telegram_alert()
│   │   └── bot.py                       ← polling loop + start_bot_thread()
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── event_pipeline.py            ← per-event state + close_event()
│   │   ├── summary_pipeline.py          ← generate_summary() + extract_person_attributes()
│   │   └── query_pipeline.py            ← classify_intent() + query_memory()
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py               ← best_frame/best_score state + helpers
│   │   ├── crop_utils.py                ← best_crops state + crop_update/save/clear
│   │   ├── roi_utils.py                 ← is_inside_roi() + draw_roi()
│   │   └── debug_utils.py               ← save_rejected_detection()
│   │
│   └── prompts/
│       ├── __init__.py
│       ├── grounding_rules.py           ← _GROUNDING_RULES constant
│       ├── summary_prompts.py           ← build_summary_messages() + build_attribute_extraction_prompt()
│       └── query_prompts.py             ← LLAMA_SYSTEM_PROMPT + INTENT_PATTERNS
│
└── data/                                ← runtime-generated data (gitignore this)
    ├── events/
    ├── smart_frames/
    ├── person_crops/
    ├── debug_rejected/
    ├── event_memory.json
    └── reid_gallery.json
```

---

## 2. File-by-File Split Plan

| Original Section | Destination File |
|---|---|
| `BOT_TOKEN`, `CHAT_ID`, `BASE_URL` | `config/settings.py` |
| `roi_points` | `config/settings.py` → `ROI_POINTS` |
| `video_path`, `FRAME_WIDTH/HEIGHT` | `config/settings.py` |
| `YOLO_MODEL_PATH` | `config/settings.py` |
| `QWEN_MODEL_ID`, `BNB_CONFIG` | `config/settings.py` |
| `GROQ_API_KEY`, `GROQ_MODEL` | `config/settings.py` |
| `REID_SIMILARITY_THRESHOLD` + all REID constants | `config/settings.py` |
| `MEMORY_FILE`, `REID_GALLERY_FILE`, `*_DIR` | `config/settings.py` |
| `_GROUNDING_RULES` | `prompts/grounding_rules.py` |
| Summary prompt text | `prompts/summary_prompts.py` |
| Attribute extraction prompt text | `prompts/summary_prompts.py` |
| `LLAMA_SYSTEM_PROMPT` | `prompts/query_prompts.py` |
| `INTENT_PATTERNS` | `prompts/query_prompts.py` |
| `YOLO(...)` + `model.track(...)` | `models/yolo_detector.py` |
| `Qwen2_5_VLForConditionalGeneration`, `_qwen_infer()` | `models/qwen_vl.py` |
| `VideoMAEModel`, `extract_smart_frames()` | `models/videomae.py` |
| `load_fastreid()`, `_load_osnet_fallback()`, `reid_fn` | `models/reid.py` |
| `groq_client`, `_llama_infer()` | `models/groq_query_engine.py` |
| `_faiss_index`, `_faiss_id_map`, `_faiss_rebuild/add/search` | `memory/faiss_index.py` |
| `gallery_data`, `gallery_counter`, `gallery_*` functions | `memory/gallery.py` |
| `empty_person_record`, `empty_event_record`, `load/save_memory` | `memory/event_memory.py` |
| `tg_send_message`, `tg_send_photo`, `tg_get_updates` | `telegram/alerts.py` |
| `_alert_worker`, `send_telegram_alert` | `telegram/alerts.py` |
| `bot_poll_loop`, `bot_offset`, `bot_chat_history` | `telegram/bot.py` |
| `generate_summary()` | `pipelines/summary_pipeline.py` |
| `extract_person_attributes()`, `_heuristic_extract()` | `pipelines/summary_pipeline.py` |
| `classify_intent()`, `build_structured_context()` | `pipelines/query_pipeline.py` |
| `find_matching_crops()`, `query_memory()` | `pipelines/query_pipeline.py` |
| `close_event()` | `pipelines/event_pipeline.py` |
| `track_to_reid_id`, `reid_to_person_idx`, `person_counter` | `pipelines/event_pipeline.py` |
| `best_frame`, `best_score` | `utils/image_utils.py` |
| `best_crops`, `crop_update/save/clear` | `utils/crop_utils.py` |
| ROI polygon test | `utils/roi_utils.py` |
| Debug rejected block | `utils/debug_utils.py` |
| Main video loop + startup + bot start | `main.py` |

---

## 3. Functions → Files

### `config/settings.py`
All constants — no functions.

### `prompts/grounding_rules.py`
- `_GROUNDING_RULES` (constant string)

### `prompts/summary_prompts.py`
- `build_summary_messages(pil_images)` → messages list for _qwen_infer
- `build_attribute_extraction_prompt(summary)` → messages list for _qwen_infer

### `prompts/query_prompts.py`
- `LLAMA_SYSTEM_PROMPT` (constant string)
- `INTENT_PATTERNS` (constant dict)

### `models/yolo_detector.py`
- `load_yolo()` → YOLO model instance
- `run_tracking(model, frame)` → results list

### `models/qwen_vl.py`
- `load_qwen()` → sets module-level `qwen_model`, `processor`
- `_qwen_infer(messages, pil_images, max_new_tokens)` → str

### `models/videomae.py`
- `load_videomae()` → sets module-level `videomae_model`, `videomae_processor`
- `extract_smart_frames(video_path_arg, ev_id)` → [str]

### `models/reid.py`
- `load_reid()` → sets `reid_fn`, `REID_DIM`; propagates dim to `config.settings`
- `_load_fastreid()` → (extract_fn, dim)
- `_load_osnet_fallback()` → (extract_fn, dim)

### `models/groq_query_engine.py`
- `load_groq()` → sets module-level `groq_client`
- `_llama_infer(system_prompt, user_prompt, max_new_tokens)` → str

### `memory/faiss_index.py`
- `_faiss_rebuild(gallery_data)` ← takes gallery_data as arg (avoids circular import)
- `_faiss_add(reid_id, embedding)`
- `faiss_search(query_embedding, k)` → [(reid_id, similarity)]

### `memory/gallery.py`
- `gallery_load()`
- `gallery_save()`
- `gallery_match(embedding)` → (reid_id, similarity)
- `gallery_update(reid_id, embedding)`
- `gallery_was_recent(reid_id)` → bool
- `gallery_purge_stale()`

### `memory/event_memory.py`
- `empty_person_record(person_id, track_id, ev_id)` → dict
- `empty_event_record(ev_id, timestamp)` → dict
- `load_memory()` → list
- `save_memory_append(event_record)`

### `telegram/alerts.py`
- `tg_send_message(chat_id, text)`
- `tg_send_photo(chat_id, image_path, caption)`
- `tg_get_updates(offset)` → list
- `_alert_worker(image_path, caption, ev_id)`
- `send_telegram_alert(image_path, summary, duration, timestamp, ev_id)`

### `telegram/bot.py`
- `bot_poll_loop()` — daemon thread body
- `start_bot_thread()` → Thread
- `stop_bot()`

### `pipelines/summary_pipeline.py`
- `generate_summary(frame_paths)` → str
- `extract_person_attributes(summary)` → [dict]
- `_heuristic_extract(summary)` → [dict]

### `pipelines/query_pipeline.py`
- `classify_intent(query)` → [str]
- `_parse_hour(ts_str)` → int | None
- `_time_matches_query(hour, query)` → bool
- `build_structured_context(memory, query, intents)` → str
- `find_matching_crops(query, memory, query_embedding)` → [(path, desc)]
- `query_memory(query, query_embedding)` → (str, [str])

### `pipelines/event_pipeline.py`
- `close_event(ev_id, ev_output_path, ev_start_time, ev_record)`
- `reset_event_state()`

### `utils/image_utils.py`
- `reset_best_frame()`
- `try_update_best_frame(frame, score)` → bool
- `get_best_frame()` → np.ndarray | None

### `utils/crop_utils.py`
- `crop_update(frame, track_id, reid_id, x1, y1, x2, y2, confidence)`
- `crop_save_by_reid(reid_id, ev_id, person_index)` → str | None
- `crop_clear()`

### `utils/roi_utils.py`
- `is_inside_roi(cx, cy)` → bool
- `draw_roi(frame)` → frame

### `utils/debug_utils.py`
- `save_rejected_detection(frame, frame_index, track_id, confidence, area, x1, y1, x2, y2)`

---

## 4. Import Changes (what moved and where)

| Original import site | New import |
|---|---|
| `import faiss` (top-level) | `memory/faiss_index.py` only |
| `from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor` | `models/qwen_vl.py` only |
| `from transformers import VideoMAEModel, VideoMAEImageProcessor` | `models/videomae.py` only |
| `from transformers import BitsAndBytesConfig` | `config/settings.py` only |
| `from ultralytics import YOLO` | `models/yolo_detector.py` only |
| `from groq import Groq` | `models/groq_query_engine.py` only |
| `torch.backends.cudnn.benchmark = True` | `main.py` only |
| `INTENT_PATTERNS` dict | `prompts/query_prompts.py` |
| `LLAMA_SYSTEM_PROMPT` | `prompts/query_prompts.py` |
| `_GROUNDING_RULES` | `prompts/grounding_rules.py` |

---

## 5. Initialization Order

```
main.py startup sequence:
  1.  sys.path.insert(0, src/)          # path setup
  2.  import config.settings            # pure constants, no side effects
  3.  import prompts.*                  # pure constants, no side effects
  4.  load_qwen()                       # GPU memory — load first
  5.  load_groq()                       # network handshake
  6.  load_videomae()                   # CPU model
  7.  load_reid()                       # CPU model; WRITES config.settings.REID_DIM
  8.  gallery_load()                    # reads REID_DIM → builds FAISS index
  9.  from models.reid import reid_fn   # resolved AFTER load_reid()
  10. model = load_yolo()               # GPU model
  11. start_bot_thread()                # daemon thread; imports query_pipeline lazily
  12. cap = cv2.VideoCapture(VIDEO_PATH)
  13. [main loop]
```

**Critical ordering constraint:** `gallery_load()` (step 8) must come **after** `load_reid()` (step 7) because `load_reid` writes `REID_DIM` to `config.settings`, and `faiss_index._faiss_rebuild()` reads that value when constructing the index.

---

## 6. main.py Structure

```
main.py
  ├── [imports — all from modular subpackages]
  ├── [torch.backends.cudnn.benchmark = True]
  ├── [section 1]  mkdir data dirs, seed MEMORY_FILE
  ├── [section 2]  load_qwen / load_groq / load_videomae / load_reid / gallery_load
  ├── [section 3]  load_yolo
  ├── [section 4]  start_bot_thread
  ├── [section 5]  open video capture
  ├── [section 6]  declare event/recording state vars
  ├── [section 7]  while True: main tracking loop
  ├── [section 8]  finalize last event
  ├── [section 9]  bot keepalive + KeyboardInterrupt handler
  └── [section 10] test query
```

---

## 7. Shared-State Handling Strategy

| State | Owner module | How main.py accesses it |
|---|---|---|
| `best_frame`, `best_score` | `utils/image_utils.py` | via `try_update_best_frame()`, `get_best_frame()`, `reset_best_frame()` |
| `best_crops` | `utils/crop_utils.py` | via `crop_update()`, `crop_save_by_reid()`, `crop_clear()` |
| `gallery_data`, `gallery_counter` | `memory/gallery.py` | via `gallery_match()`, `gallery_save()`, etc. |
| `_faiss_index`, `_faiss_id_map` | `memory/faiss_index.py` | via `faiss_search()`, `_faiss_add()`, `_faiss_rebuild()` |
| `qwen_model`, `processor` | `models/qwen_vl.py` | via `_qwen_infer()` only |
| `groq_client` | `models/groq_query_engine.py` | via `_llama_infer()` only |
| `videomae_model` | `models/videomae.py` | via `extract_smart_frames()` only |
| `reid_fn`, `REID_DIM` | `models/reid.py` | `reid_fn` imported after `load_reid()` |
| `current_event`, `track_to_*`, `person_counter` | `pipelines/event_pipeline.py` | via `ep.*` (module aliased as `ep`) |
| `bot_running`, `bot_chat_history` | `telegram/bot.py` | via `stop_bot()`, internal to thread |
| `REID_DIM` (runtime) | `config/settings.py` | written by `models/reid.py`; read by `memory/faiss_index.py` |

---

## 8. Avoiding Circular Imports

The one structural risk in this codebase is the FAISS ↔ gallery cycle:
- `gallery.py` calls `_faiss_rebuild(gallery_data)` — it **passes `gallery_data` as an argument** rather than importing it from `faiss_index`, so `faiss_index.py` never imports `gallery.py`.
- `faiss_index.py` reads `REID_DIM` from `config.settings` at **call time** (not import time), which means it sees the updated value written by `models/reid.py` after `load_reid()` returns.
- `telegram/bot.py` imports `query_memory` **inside** `bot_poll_loop()` (lazy import), avoiding any import-time dependency on models that haven't been loaded yet.
- `models/reid.py` writes back to `config.settings.REID_DIM` at load time via `import config.settings as _cfg; _cfg.REID_DIM = REID_DIM` — a one-way write, no circular dependency.

```
Dependency graph (→ means "imports from"):
  main.py → everything (leaf consumer)
  config/settings.py → (nothing in src/)
  prompts/* → config/settings (only grounding_rules; summary_prompts → grounding_rules)
  models/* → config/settings
  models/reid.py → config/settings (writes REID_DIM back)
  memory/faiss_index.py → config/settings (reads REID_DIM at call time)
  memory/gallery.py → config/settings, memory/faiss_index
  memory/event_memory.py → config/settings
  utils/* → config/settings
  telegram/alerts.py → config/settings
  telegram/bot.py → config/settings, telegram/alerts (lazy: pipelines/query_pipeline)
  pipelines/summary_pipeline.py → models/qwen_vl, prompts/summary_prompts
  pipelines/query_pipeline.py → config/settings, memory/event_memory,
                                 memory/faiss_index, models/groq_query_engine,
                                 prompts/query_prompts
  pipelines/event_pipeline.py → config/settings, memory/event_memory,
                                  memory/gallery, models/videomae,
                                  pipelines/summary_pipeline,
                                  telegram/alerts, utils/crop_utils,
                                  utils/image_utils
```

No cycles exist in this graph.

---

## 9. What Should Remain Global (Module-Level State)

| Variable | Reason |
|---|---|
| `gallery_data`, `gallery_counter` in `memory/gallery.py` | shared across every tracking frame and every query |
| `_faiss_index`, `_faiss_id_map` in `memory/faiss_index.py` | single index instance, mutated in-place |
| `best_crops` in `utils/crop_utils.py` | accumulated per frame during an event |
| `best_frame`, `best_score` in `utils/image_utils.py` | accumulated per frame during an event |
| `qwen_model`, `processor` in `models/qwen_vl.py` | heavy GPU model, load once |
| `groq_client` in `models/groq_query_engine.py` | persistent API connection |
| `videomae_model` in `models/videomae.py` | CPU model, load once |
| `reid_fn`, `REID_DIM` in `models/reid.py` | resolved at startup, used every frame |
| `bot_running`, `bot_chat_history` in `telegram/bot.py` | daemon thread state |
| `current_event`, `track_to_*`, `person_counter` in `pipelines/event_pipeline.py` | per-event accumulation |

---

## 10. Code That Must Remain Untouched

The following function bodies are **copied verbatim** — zero logic change:

- `_qwen_infer()` — exact generation parameters
- `_llama_infer()` — exact Groq API call
- `extract_smart_frames()` — VideoMAE logic + frame selection
- `gallery_match()` — FAISS threshold + EMA update
- `gallery_update()` — EMA formula + normalisation
- `_faiss_rebuild()` — index construction logic
- `faiss_search()` — query normalisation + k-NN lookup
- `close_event()` — 9-step finalisation sequence (Steps 1-9)
- `bot_poll_loop()` — anti-spam, retry, chat history window
- `send_telegram_alert()` — threading + retry + caption truncation
- `_alert_worker()` — exponential backoff retry
- `generate_summary()` — prompt + image list construction
- `extract_person_attributes()` — JSON parse + fallback chain
- `_heuristic_extract()` — keyword extraction logic
- `classify_intent()` — regex matching against INTENT_PATTERNS
- `build_structured_context()` — time-filter + context builder
- `find_matching_crops()` — FAISS + keyword dual-path retrieval
- `query_memory()` — intent → context → Llama flow
- `empty_person_record()` / `empty_event_record()` — exact JSON schema
- All prompt strings — reproduced character-for-character

---

## 11. Minimal-Diff Migration Strategy

The migration follows a strict **outside-in** order — leaves first, root last:

```
Phase 1 (no dependencies on other src modules):
  config/settings.py
  prompts/grounding_rules.py

Phase 2 (depend only on Phase 1):
  prompts/summary_prompts.py
  prompts/query_prompts.py
  utils/roi_utils.py
  utils/image_utils.py
  utils/crop_utils.py
  utils/debug_utils.py
  models/yolo_detector.py
  models/qwen_vl.py
  models/videomae.py
  models/reid.py
  models/groq_query_engine.py
  memory/event_memory.py

Phase 3 (depend on Phase 1 + 2):
  memory/faiss_index.py       (reads config.settings.REID_DIM)
  memory/gallery.py           (uses faiss_index)
  telegram/alerts.py          (uses config.settings)
  pipelines/summary_pipeline.py (uses models/qwen_vl + prompts)

Phase 4 (depend on Phase 1-3):
  pipelines/query_pipeline.py (uses memory + models + prompts)
  pipelines/event_pipeline.py (uses memory + models + pipelines/summary + telegram + utils)
  telegram/bot.py             (lazy-imports pipelines/query_pipeline)

Phase 5 (root):
  main.py                     (imports from everything)
```

---

## 12. Safe Modularization Order

Follow Phase 1 → 5 above. After each phase, verify with a dry-run import:

```bash
cd cctv_v4/src
python -c "from config import settings; print('Phase 1 OK')"
python -c "from prompts import summary_prompts, query_prompts; print('Phase 2 prompts OK')"
python -c "from utils import roi_utils, crop_utils; print('Phase 2 utils OK')"
# ... continue per phase
python -c "from pipelines import query_pipeline; print('Phase 4 OK')"
python main.py   # full run
```

---

## 13. Functions That Must Remain Exactly Unchanged

Every function listed in Section 10 above. In addition, these helper functions in `main.py` remain as inline logic (not extracted further):

- The per-detection `for` loop body (embedding + gallery_match + crop_update)
- The event-start block (VideoWriter init + empty_event_record)
- The person-dedup block (reid_to_person_idx check)
- The event-end grace-period check (elapsed > grace)

---

## 14. Dependency Flow Between Modules

```
config/settings.py
       ↑
       │ (all modules read from here)
       │
  ┌────┴────────────────────────────────────────────┐
  │                                                  │
prompts/*          utils/*           models/*        memory/*
  │                  │                  │               │
  │                  │         models/reid.py ──writes─►REID_DIM
  │                  │                  │               │
  └──────────────────┴──────────────────┴───────────────┤
                                                         │
                                            memory/faiss_index.py
                                                         │
                                            memory/gallery.py
                                                         │
                              ┌──────────────────────────┤
                              │                           │
                   pipelines/summary_pipeline    pipelines/query_pipeline
                              │                           │
                              └────────┐    ┌─────────────┘
                                       │    │
                              pipelines/event_pipeline
                                       │
                               telegram/alerts
                                       │
                               telegram/bot ──(lazy)──► pipelines/query_pipeline
                                       │
                                   main.py
```

---

## 15. Final Execution Flow Diagram

```
python src/main.py
        │
        ├─ [dirs + memory seed]
        │
        ├─ load_qwen()          → qwen_model, processor     [GPU]
        ├─ load_groq()          → groq_client               [network]
        ├─ load_videomae()      → videomae_model             [CPU]
        ├─ load_reid()          → reid_fn, REID_DIM          [CPU]
        │                         (writes config.settings.REID_DIM)
        ├─ gallery_load()       → gallery_data + FAISS index [disk]
        ├─ reid_fn (import)     → resolved module-level fn
        ├─ load_yolo()          → YOLO model                 [GPU]
        ├─ start_bot_thread()   ──────────────────────────── [daemon thread]
        │                                                          │
        │                                              bot_poll_loop()
        │                                              tg_get_updates()
        │                                              query_memory()
        │                                                _llama_infer()
        │                                              tg_send_message()
        │
        ├─ cap = VideoCapture(VIDEO_PATH)
        │
        └─ while True:  ← MAIN LOOP
               │
               ├─ cap.read()  →  frame
               ├─ resize + draw_roi(frame)
               ├─ run_tracking(model, frame)  →  results
               │
               ├─ [for each detection]
               │       ├─ filter: MIN_AREA, MIN_CONFIDENCE
               │       │     └─ save_rejected_detection()
               │       ├─ is_inside_roi(cx, cy)
               │       ├─ try_update_best_frame(frame, score)
               │       └─ cv2.rectangle(frame, ...)
               │
               ├─ [if person_in_roi]
               │       ├─ reid_fn(crop)           →  embedding
               │       ├─ gallery_match(embedding) →  (reid_id, sim)
               │       ├─ crop_update(...)
               │       └─ [if not recording]
               │             ├─ VideoWriter init
               │             └─ empty_event_record()
               │
               │       └─ [person dedup loop]
               │             ├─ reid_to_person_idx check
               │             └─ empty_person_record() + append
               │
               ├─ [if recording] video_writer.write(frame)
               │
               └─ [if recording + elapsed > grace]
                       ├─ video_writer.release()
                       └─ close_event()
                               ├─ cv2.imwrite(snapshot)
                               ├─ send_telegram_alert()   [thread]
                               ├─ extract_smart_frames()  [VideoMAE]
                               ├─ generate_summary()      [Qwen2.5-VL]
                               ├─ tg_send_message()       [full report]
                               ├─ crop_save_by_reid()
                               ├─ extract_person_attributes() [Qwen2.5-VL]
                               ├─ save_memory_append()    [JSON]
                               ├─ gallery_save()          [JSON]
                               └─ reset_event_state() + reset_best_frame()

        [loop ends]
        ├─ close_event() for final open event (if any)
        ├─ cap.release()
        ├─ gallery_save()
        ├─ bot keepalive (Ctrl+C → stop_bot())
        └─ query_memory("Who carried a bag?")  [test]
```

---

## Running the Modularized System

```bash
# From the project root:
cd cctv_v4
python src/main.py

# Or with explicit path:
PYTHONPATH=src python src/main.py
```

All behavior, outputs, Telegram messages, JSON schemas, FAISS operations, 
ReID logic, and event lifecycles are **identical** to the original monolith.
