"""
utils/event_logger.py
======================
Pure print()-formatting helpers so every module in the project logs in
one consistent shape. No logic, no state, no side effects beyond
printing — safe to import from anywhere without risk of circular
imports.

Format:
==================================================
[TAG]
line 1
line 2
--------------------------------------------------
"""

_BAR = "=" * 50
_SEP = "-" * 50


def log_bar():
    """Top/bottom event or query boundary."""
    print(_BAR)


def log_event_header(event_id):
    print(f"\n{_BAR}\nEVENT #{event_id}\n{_BAR}")


def log_block(tag, *lines):
    """Print one tagged block followed by a separator line, e.g.

        [YOLO]
        Detecting persons...
        --------------------------------------------------
    """
    print(f"[{tag}]")
    for line in lines:
        print(line)
    print(_SEP)


def log_completion(pipeline_used, summary_ok, db_ok, telegram_status, total_seconds):
    print(_BAR)
    print("EVENT COMPLETED")
    print(f"Pipeline Used          : {pipeline_used}")
    print(f"Summary                : {'SUCCESS' if summary_ok else 'FAILED'}")
    print(f"Database               : {'SUCCESS' if db_ok else 'FAILED'}")
    print(f"Telegram               : {telegram_status}")
    print(f"Total Processing Time  : {total_seconds:.2f} sec")
    print(_BAR)
