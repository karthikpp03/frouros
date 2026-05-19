# frouros



src/
│
├── main.py
│
├── config/
│ ├── settings.py
│
├── models/
│ ├── yolo_detector.py
│ ├── qwen_vl.py
│ ├── videomae.py
│ ├── reid.py
│ ├── groq_query_engine.py
│
├── memory/
│ ├── event_memory.py
│ ├── faiss_index.py
│ ├── gallery.py
│
├── telegram/
│ ├── bot.py
│ ├── alerts.py
│
├── pipelines/
│ ├── event_pipeline.py
│ ├── summary_pipeline.py
│ ├── query_pipeline.py
│
├── utils/
│ ├── image_utils.py
│ ├── crop_utils.py
│ ├── roi_utils.py
│ ├── debug_utils.py
│
├── prompts/
│ ├── grounding_rules.py
│ ├── summary_prompts.py
│ ├── query_prompts.py
│
└── data/
├── events/
├── smart_frames/
├── person_crops/
├── debug_rejected/
