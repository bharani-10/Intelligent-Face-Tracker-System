import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.config import ConfigurationError, load_config
from app.database import DatabaseConfigurationError, DatabaseManager


SUPPORTED_EXTENSIONS = ["mp4", "avi", "mov", "mkv"]
UPLOAD_DIR = ROOT / "data" / "uploads"
RUN_DIR = ROOT / "data" / "runs"


def discover_sample_videos() -> list[Path]:
    return sorted(
        path for path in (ROOT / "videos").iterdir()
        if path.is_file() and path.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS
    )


def save_upload(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower()
    target = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    with target.open("wb") as output_file:
        output_file.write(uploaded_file.getbuffer())
    return target


def load_stats(stats_path: Path):
    if not stats_path.exists():
        return None
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def event_id_floor(database: DatabaseManager) -> int:
    events = database.get_event_history()
    return max((int(event["event_id"]) for event in events), default=0)


st.set_page_config(page_title="Intelligent Face Tracker", layout="wide")
st.title("Intelligent Face Tracker")
st.caption("YOLO · ByteTrack · InsightFace · PostgreSQL")

try:
    config = load_config("config.json")
    database = DatabaseManager(env_var=config["database"].get("url_env", "DATABASE_URL"))
except (ConfigurationError, DatabaseConfigurationError):
    st.error("Database connection failed. Check DATABASE_URL and PostgreSQL availability.")
    st.stop()
except Exception:
    st.error("Database connection failed. Check the backend logs.")
    st.stop()

for key, default in {
    "pipeline": None,
    "run": None,
    "selected_path": None,
    "uploaded_path": None,
}.items():
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.subheader("Video Source")
    source_mode = st.radio("Choose source", ["Use sample video", "Upload your own video"])
    selected_path = None

    if source_mode == "Use sample video":
        samples = discover_sample_videos()
        if not samples:
            st.warning("No supported videos were found in videos/.")
        else:
            selected_path = st.selectbox(
                "Available videos",
                samples,
                format_func=lambda path: path.name,
            )
    else:
        uploaded_file = st.file_uploader(
            "Upload a video to test the face tracker",
            type=SUPPORTED_EXTENSIONS,
        )
        if uploaded_file is not None:
            if uploaded_file.size == 0:
                st.error("The uploaded video is empty.")
            else:
                if (
                    st.session_state.uploaded_path is None
                    or st.session_state.uploaded_path.name != uploaded_file.name
                ):
                    st.session_state.uploaded_path = save_upload(uploaded_file)
                selected_path = st.session_state.uploaded_path

    if selected_path is not None:
        st.session_state.selected_path = str(selected_path)
        st.write(f"Selected: `{Path(selected_path).name}`")
        st.video(str(selected_path))

    with st.expander("Runtime settings", expanded=False):
        st.write(f"Detection confidence: `{config['detection']['confidence']}`")
        st.write(
            "Recognition threshold: `"
            f"{config['recognition'].get('threshold', config['recognition'].get('similarity_threshold'))}`"
        )
        st.write(f"Frame skip: `{config['detection']['frame_skip']}`")
        st.write(f"Missing-frame timeout: `{config['tracking']['max_age']}`")
        st.caption("Settings are read-only for this test session.")

    can_start = bool(st.session_state.selected_path) and st.session_state.pipeline is None
    if st.button("Start Processing", type="primary", disabled=not can_start):
        source = Path(st.session_state.selected_path)
        if not source.exists():
            st.error("Video could not be opened.")
        else:
            RUN_DIR.mkdir(parents=True, exist_ok=True)
            run_id = uuid4().hex
            stats_path = RUN_DIR / f"{run_id}.json"
            run = {
                "id": run_id,
                "filename": source.name,
                "source": str(source),
                "stats_path": str(stats_path),
                "event_id_floor": event_id_floor(database),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            st.session_state.run = run
            st.session_state.pipeline = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "app.main",
                    "--source",
                    str(source),
                    "--stats-path",
                    str(stats_path),
                    "--no-display",
                ],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            st.rerun()

    if st.button("New Test"):
        if st.session_state.pipeline is not None:
            st.session_state.pipeline.terminate()
        st.session_state.pipeline = None
        st.session_state.run = None
        st.session_state.selected_path = None
        st.session_state.uploaded_path = None
        st.rerun()

run = st.session_state.run
if st.session_state.pipeline is not None:
    if st.session_state.pipeline.poll() is None:
        st.info("Processing video... Please wait.")
        time.sleep(1)
        st.rerun()
    else:
        return_code = st.session_state.pipeline.returncode
        st.session_state.pipeline = None
        if return_code != 0:
            st.session_state.run["error"] = "Processing failed. Check the backend logs."
        else:
            st.session_state.run["stats"] = load_stats(Path(st.session_state.run["stats_path"]))
            if st.session_state.run["stats"] is None:
                st.session_state.run["error"] = "Processing failed. No result statistics were produced."
        st.rerun()

if run is not None:
    st.subheader(f"Current Test: {run['filename']}")
    if "error" in run:
        st.error(run["error"])
    elif "stats" in run:
        stats = run["stats"]
        st.success("Processing completed successfully.")
        cards = st.columns(4)
        cards[0].metric("Unique Visitors", stats["unique_visitors"])
        cards[1].metric("Total Entries", stats["total_entries"])
        cards[2].metric("Total Exits", stats["total_exits"])
        cards[3].metric("Currently Present", stats["currently_present"])
        cards = st.columns(3)
        cards[0].metric("Frames Processed", stats["frames_processed"])
        cards[1].metric("Processing FPS", f"{stats['processing_fps']:.2f}")
        cards[2].metric("Processing Time", f"{stats['processing_time_seconds']:.2f}s")
        if stats.get("entries_this_run", 0) == 0:
            st.info("No faces were detected in this video.")
        else:
            st.caption(
                f"This run: {stats.get('visitors_registered_this_run', 0)} new visitors, "
                f"{stats['entries_this_run']} entries, {stats['exits_this_run']} exits."
            )
        st.caption(f"Test completed at {datetime.now().astimezone().isoformat(timespec='seconds')}")
    else:
        st.info("Processing video... Please wait.")

events = database.get_event_history()
if run is not None:
    events = [event for event in events if int(event["event_id"]) > run["event_id_floor"]]
events = list(reversed(events[-50:]))

st.subheader("Recent Events")
if events:
    st.dataframe(
        [
            {
                "Visitor ID": event["visitor_id"],
                "Event Type": event["event_type"],
                "Track ID": event["track_id"],
                "Timestamp": event["timestamp"],
                "Image Path": event["image_path"],
            }
            for event in events
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No events are available for this test.")

for event_type, heading in (("ENTRY", "Entry Images"), ("EXIT", "Exit Images")):
    st.subheader(heading)
    image_events = [event for event in events if event["event_type"] == event_type]
    if not image_events:
        st.info(f"No {event_type.lower()} images are available for this test.")
        continue
    columns = st.columns(min(4, len(image_events)))
    for index, event in enumerate(image_events[:12]):
        image_path = Path(event["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        with columns[index % len(columns)]:
            if image_path.exists():
                st.image(str(image_path), caption=f"{event['visitor_id']} · Track {event['track_id']}")
            else:
                st.warning(f"Image unavailable: {event['image_path']}")

st.subheader("Current Visitors")
present_ids = set(database.get_currently_present_visitors())
visitor_rows = [
    {
        "Visitor ID": visitor["visitor_id"],
        "First Seen": visitor["first_seen"],
        "Last Seen": visitor["last_seen"],
        "Status": "PRESENT" if visitor["visitor_id"] in present_ids else "EXITED",
    }
    for visitor in database.retrieve_visitors()
]
st.dataframe(visitor_rows, use_container_width=True, hide_index=True)