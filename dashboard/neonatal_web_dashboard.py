
"""
BabyPulse Headless Web Dashboard
For Raspberry Pi without desktop/HDMI.

Keeps these files unchanged:
    /home/pi/neonatal_monitor.py
    /home/pi/cry_predict.py

Run:
    source ~/cry-env/bin/activate
    streamlit run ~/neonatal_web_dashboard.py --server.address 0.0.0.0 --server.port 8501
"""

import os
import re
import json
import csv
import time
import signal
import threading
import subprocess
import datetime as dt
from pathlib import Path

import streamlit as st
import requests

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO = None
    GPIO_AVAILABLE = False

# ---------------- CONFIG ----------------
BASE = Path("/home/pi")
DATA_FILE = BASE / "babypulse_web_data.json"
TELEGRAM_FILE = BASE / "babypulse_telegram.json"
MONITOR_SCRIPT = BASE / "neonatal_monitor.py"
PREDICT_SCRIPT = BASE / "cry_predict.py"
UPLOAD_DIR = BASE / "manual_uploads"

GREEN_GPIO = 17
RED_GPIO = 22
ESCALATION_SECONDS = 60
CRITICAL = {"pain", "distress"}

DEFAULT_PROFILE = {
    "name": "Baby 1",
    "dob": "2026-06-10",
    "age_months": "3",
    "mother_name": "Mother",
    "father_name": "Father",
    "birth_weight_kg": "2.9",
    "current_weight_kg": "4.6",
    "blood_group": "O+",
    "notes": "No known allergies."
}

DEFAULT_SCHEDULE = {
    "last_feed": "",
    "next_feed": "",
    "last_diaper": "",
    "last_medication": "",
    "sleep_start": ""
}

LABELS = ["Normal", "Hunger", "Pain", "Discomfort", "Sleep", "Distress"]

# ---------------- DATA ----------------
def default_data():
    return {
        "profile": dict(DEFAULT_PROFILE),
        "schedule": dict(DEFAULT_SCHEDULE),
        "alerts": [],
        "cry_history": [],
        "care_history": [],
        "vital_history": [],
        "monitor_history": [],
        "uploads": [],
        "care_center": {
            "active": True,
            "name": "NICU / Care Center",
            "contact": "Primary Caregiver",
            "phone": "",
        },
        "checklist": {
            "feeding": False,
            "diaper": False,
            "medication": False,
            "sleep": False,
            "vitals": False,
            "alerts": False,
        },
    }


def load_data():
    data = default_data()
    try:
        if DATA_FILE.exists():
            saved = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            for k in ("profile", "schedule", "care_center", "checklist"):
                if isinstance(saved.get(k), dict):
                    data[k].update(saved[k])
            for k in ("alerts", "cry_history", "care_history",
                      "vital_history", "monitor_history", "uploads"):
                if isinstance(saved.get(k), list):
                    data[k] = saved[k]
    except Exception:
        pass
    return data


def save_data(data):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(DATA_FILE)


def telegram_config():
    cfg = {"bot_token": "", "primary_chat_id": "", "secondary_chat_id": ""}
    try:
        if TELEGRAM_FILE.exists():
            saved = json.loads(TELEGRAM_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                cfg.update(saved)
    except Exception:
        pass
    return cfg


DATA = load_data()
TG = telegram_config()

# ---------------- RUNTIME ----------------
if "runtime" not in st.session_state:
    st.session_state.runtime = {
        "process": None,
        "running": False,
        "reader": None,
        "lines": [],
        "last_label": "No analysis yet",
        "confidence": 0.0,
        "status": "READY",
        "baby_temp": "--",
        "room_temp": "--",
        "green_led": False,
        "red_led": False,
        "alert_id": None,
        "alert_label": None,
        "alert_started": None,
        "escalation_due": None,
        "primary_sent": False,
        "secondary_sent": False,
        "last_signature": None,
        "lock": threading.Lock(),
    }

R = st.session_state.runtime

# ---------------- STYLE ----------------
st.set_page_config(
    page_title="BabyPulse",
    page_icon="🍼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background:#08101f; color:#f5f7ff; }
[data-testid="stSidebar"] { background:#0e1830; }
.block-container { padding-top:1.5rem; }
.card {
    background:#121e38; border:1px solid #26395f; border-radius:16px;
    padding:18px; margin-bottom:14px;
}
.metric {
    background:#121e38; border:1px solid #26395f; border-radius:14px;
    padding:16px;
}
.danger { color:#ff5c7a; font-weight:800; }
.good { color:#45d6a0; font-weight:800; }
.info { color:#6c8cff; font-weight:800; }
.warn { color:#ffc857; font-weight:800; }
.small { color:#9ba7c7; font-size:.88rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- HARDWARE ----------------
def gpio_setup():
    if not GPIO_AVAILABLE:
        return
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(GREEN_GPIO, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(RED_GPIO, GPIO.OUT, initial=GPIO.LOW)
    except Exception:
        pass


def set_leds(green=False, red=False):
    R["green_led"] = bool(green)
    R["red_led"] = bool(red)
    if GPIO_AVAILABLE:
        try:
            GPIO.output(GREEN_GPIO, GPIO.HIGH if green else GPIO.LOW)
            GPIO.output(RED_GPIO, GPIO.HIGH if red else GPIO.LOW)
        except Exception:
            pass


gpio_setup()
set_leds(R["green_led"], R["red_led"])

# ---------------- TELEGRAM ----------------
def telegram_send(chat_id, message):
    token = str(TG.get("bot_token", "")).strip()
    chat_id = str(chat_id or "").strip()
    if not token or not chat_id or token == "YOUR_BOT_TOKEN":
        return False, "Telegram configuration missing"
    try:
        res = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        if res.ok:
            return True, "sent"
        return False, f"HTTP {res.status_code}: {res.text[:150]}"
    except Exception as e:
        return False, str(e)


def record_alert(kind, status, message):
    DATA["alerts"].append({
        "time": dt.datetime.now().strftime("%d %b %Y %H:%M:%S"),
        "type": kind,
        "status": status,
        "message": message,
    })
    DATA["alerts"] = DATA["alerts"][-200:]
    save_data(DATA)


def primary_alert(label):
    baby = DATA["profile"].get("name", "Baby 1")
    message = (
        "🚨 BABYPULSE CRITICAL ALERT\n\n"
        f"Baby: {baby}\n"
        f"Classification: {label}\n"
        "Priority: HIGH\n"
        "Red LED: ON\n"
        "Action: Immediate caregiver attention required.\n"
        f"Time: {dt.datetime.now().strftime('%d %b %Y %H:%M:%S')}"
    )
    ok, detail = telegram_send(TG.get("primary_chat_id"), message)
    record_alert(label, "Primary Telegram sent" if ok else f"Primary Telegram failed: {detail}", message)
    R["primary_sent"] = ok


def secondary_alert(label):
    if R["alert_id"] is None or R["alert_label"] != label:
        return
    baby = DATA["profile"].get("name", "Baby 1")
    message = (
        "⚠️ BABYPULSE ESCALATION ALERT\n\n"
        f"Baby: {baby}\n"
        f"Classification: {label}\n"
        "Primary caregiver has not cleared the alert.\n"
        "Secondary caregiver / care center notified.\n"
        f"Time: {dt.datetime.now().strftime('%d %b %Y %H:%M:%S')}"
    )
    ok, detail = telegram_send(TG.get("secondary_chat_id"), message)
    record_alert(label, "Secondary Telegram sent" if ok else f"Secondary Telegram failed: {detail}", message)
    R["secondary_sent"] = ok


def start_alert(label):
    label = str(label).strip().title()
    if label.lower() not in CRITICAL:
        clear_alert()
        return

    signature = label.lower()
    if R["alert_id"] is not None and R["last_signature"] == signature:
        return

    R["last_signature"] = signature
    R["alert_id"] = time.time_ns()
    R["alert_label"] = label
    R["alert_started"] = time.time()
    R["escalation_due"] = time.time() + ESCALATION_SECONDS
    R["primary_sent"] = False
    R["secondary_sent"] = False
    set_leds(green=False, red=True)

    record_alert(label, "Critical alert active",
                 f"Critical classification detected: {label}")
    threading.Thread(target=primary_alert, args=(label,), daemon=True).start()


def clear_alert():
    R["alert_id"] = None
    R["alert_label"] = None
    R["alert_started"] = None
    R["escalation_due"] = None
    R["last_signature"] = None
    R["primary_sent"] = False
    R["secondary_sent"] = False
    set_leds(green=R["running"], red=False)


def check_escalation():
    if (
        R["alert_id"] is not None
        and R["alert_label"]
        and not R["secondary_sent"]
        and R["escalation_due"]
        and time.time() >= R["escalation_due"]
    ):
        threading.Thread(
            target=secondary_alert,
            args=(R["alert_label"],),
            daemon=True
        ).start()
        R["escalation_due"] = None

# ---------------- MONITOR PROCESS ----------------
def append_line(line):
    R["lines"].append(line.rstrip())
    R["lines"] = R["lines"][-300:]


def parse_label(text):
    matches = re.findall(
        r"Cry classification\s*:\s*([A-Za-z]+)",
        text,
        flags=re.I
    )
    return matches[-1].title() if matches else None


def parse_confidence(text):
    matches = re.findall(
        r"Confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        text,
        flags=re.I
    )
    return float(matches[-1]) / 100 if matches else None


def parse_temp(text):
    m = re.findall(
        r"(?:baby\s+temperature|temperature)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.I
    )
    return float(m[-1]) if m else None


def process_monitor_line(line):
    append_line(line)
    lower = line.lower()

    label = parse_label(line)
    conf = parse_confidence(line)
    temp = parse_temp(line)

    if label:
        R["last_label"] = label
        R["status"] = "DANGER" if label.lower() in CRITICAL else "STABLE"
        if conf is not None:
            R["confidence"] = conf

        DATA["cry_history"].append({
            "time": dt.datetime.now().strftime("%d %b %Y %H:%M:%S"),
            "label": label,
            "confidence": R["confidence"],
            "source": "Raspberry Pi live monitor",
        })
        DATA["cry_history"] = DATA["cry_history"][-200:]
        DATA["monitor_history"].append({
            "time": dt.datetime.now().strftime("%d %b %Y %H:%M:%S"),
            "classification": label,
            "confidence": R["confidence"],
            "status": R["status"],
        })
        DATA["monitor_history"] = DATA["monitor_history"][-200:]
        save_data(DATA)

        if label.lower() in CRITICAL:
            start_alert(label)
        else:
            clear_alert()

    if temp is not None:
        R["baby_temp"] = f"{temp:.2f}"

    if "room temperature" in lower:
        rt = parse_temp(line)
        if rt is not None:
            R["room_temp"] = f"{rt:.2f}"

    if "green led" in lower:
        R["green_led"] = "on" in lower

    if "red led" in lower:
        R["red_led"] = "on" in lower


def monitor_reader(proc):
    try:
        for raw in iter(proc.stdout.readline, ""):
            if not raw:
                break
            process_monitor_line(raw)
    except Exception as e:
        append_line(f"Reader error: {e}")
    finally:
        R["running"] = False


def start_monitor():
    if R["running"]:
        return True, "Already running"
    if not MONITOR_SCRIPT.exists():
        return False, f"Missing {MONITOR_SCRIPT}"

    try:
        set_leds(green=True, red=False)
        R["status"] = "MONITORING"
        R["lines"] = []
        R["last_label"] = "Listening..."
        R["confidence"] = 0.0
        R["last_signature"] = None
        R["process"] = subprocess.Popen(
            ["python3", str(MONITOR_SCRIPT)],
            cwd=str(BASE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        R["running"] = True
        R["reader"] = threading.Thread(
            target=monitor_reader,
            args=(R["process"],),
            daemon=True,
        )
        R["reader"].start()
        return True, "Monitoring started"
    except Exception as e:
        R["running"] = False
        set_leds(False, False)
        return False, str(e)


def stop_monitor():
    proc = R.get("process")
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass

    R["process"] = None
    R["running"] = False
    R["status"] = "READY"

    # Stop must always physically switch both LEDs off.
    set_leds(False, False)

    # Do not silently send a new Telegram message just because monitoring stopped.
    # An already-active critical alert is allowed to finish its escalation timer.
    append_line("LIVE MONITORING STOPPED")
    return True

# ---------------- MANUAL UPLOAD ----------------
def save_uploaded(uploaded):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded.name).name.replace(" ", "_")
    target = UPLOAD_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{safe_name}"
    target.write_bytes(uploaded.getvalue())

    DATA["uploads"].append({
        "time": dt.datetime.now().strftime("%d %b %Y %H:%M:%S"),
        "file": str(target),
        "original_name": uploaded.name,
        "size_bytes": target.stat().st_size,
    })
    DATA["uploads"] = DATA["uploads"][-100:]
    save_data(DATA)
    return target

# ---------------- NAV ----------------
PAGES = [
    "Dashboard",
    "Live Monitor",
    "Manual Upload",
    "Baby Profile",
    "Care Center",
    "Alerts",
    "Weekly Report",
    "System",
]

with st.sidebar:
    st.markdown("## 🍼 BabyPulse")
    st.caption("Smart AI-Enabled Neonatal Distress Monitoring")
    page = st.radio("Navigation", PAGES, index=0)

    st.divider()
    state = "🟢 ACTIVE" if R["running"] else "⚪ READY"
    st.markdown(f"**Monitor:** {state}")
    led = "🔴 RED" if R["red_led"] else ("🟢 GREEN" if R["green_led"] else "⚪ OFF")
    st.markdown(f"**LED:** {led}")
    if R["alert_label"]:
        st.markdown(f"**Alert:** 🚨 {R['alert_label']}")

# ---------------- HEADER ----------------
st.title("🍼 BabyPulse")
st.caption("Headless Raspberry Pi dashboard — controlled from your laptop browser")

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    check_escalation()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Baby", DATA["profile"].get("name", "Baby 1"))
    with c2:
        st.metric("Monitor", "LIVE" if R["running"] else "READY")
    with c3:
        st.metric("Latest Cry", R["last_label"])
    with c4:
        st.metric("Confidence", f"{R['confidence']*100:.1f}%")

    st.subheader("Current status")
    if R["alert_label"]:
        st.error(f"🚨 CRITICAL — {R['alert_label']}")
    elif R["running"]:
        st.info("🔵 LIVE MONITORING")
    else:
        st.success("🟢 READY / STABLE")

    a, b, c, d = st.columns(4)
    a.metric("Baby Temperature", f"{R['baby_temp']} °C")
    b.metric("Room Temperature", f"{R['room_temp']} °C")
    c.metric("Green LED", "ON" if R["green_led"] else "OFF")
    d.metric("Red LED", "ON" if R["red_led"] else "OFF")

    st.subheader("Latest AI history")
    if DATA["cry_history"]:
        st.dataframe(DATA["cry_history"][-10:][::-1], use_container_width=True)
    else:
        st.info("No AI classifications recorded yet.")

# ---------------- LIVE MONITOR ----------------
elif page == "Live Monitor":
    st.subheader("🎙️ Live Monitor")
    st.write("This page starts your existing `neonatal_monitor.py`. It does not replace it.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ Start Monitoring", type="primary",
                     disabled=R["running"], use_container_width=True):
            ok, msg = start_monitor()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    with c2:
        if st.button("■ Stop Monitoring",
                     disabled=not R["running"], use_container_width=True):
            stop_monitor()
            st.warning("Monitoring stopped. Both LEDs were switched OFF.")
    with c3:
        if st.button("🧪 Test Red Alert", use_container_width=True):
            start_alert("Distress")
            st.warning("Test critical alert started.")

    st.divider()

    x1, x2, x3 = st.columns(3)
    x1.metric("AI Result", R["last_label"])
    x2.metric("Confidence", f"{R['confidence']*100:.1f}%")
    x3.metric("Status", R["status"])

    st.write("### Physical LEDs")
    l1, l2 = st.columns(2)
    l1.success("🟢 GREEN LED ON" if R["green_led"] else "⚪ GREEN LED OFF")
    l2.error("🔴 RED LED ON" if R["red_led"] else "⚪ RED LED OFF")

    st.write("### Monitor console")
    st.code("\n".join(R["lines"][-80:]) or "No monitor output yet.", language="text")

    # Live rerun. Streamlit 1.64 supports fragments with run_every.
    try:
        st.fragment(run_every="2s")(lambda: None)()
    except Exception:
        pass

# ---------------- MANUAL UPLOAD ----------------
elif page == "Manual Upload":
    st.subheader("📁 Manual Audio / Video Upload")
    st.info(
        "Uploading a file stores it on the Pi. It does NOT automatically create "
        "a critical Telegram alert. This prevents the previous upload-only alert problem."
    )

    uploaded = st.file_uploader(
        "Upload a cry audio/video file",
        type=["wav", "mp3", "m4a", "mp4", "avi", "mov", "webm"],
        max_upload_size=200,
    )

    if uploaded:
        st.audio(uploaded) if uploaded.type and uploaded.type.startswith("audio") else None
        if uploaded.type and uploaded.type.startswith("video"):
            st.video(uploaded)

        if st.button("Save file to Raspberry Pi", type="primary"):
            path = save_uploaded(uploaded)
            st.success(f"Saved: {path}")

            # Explicitly no Telegram alert here.
            st.caption("No Telegram alert was generated by the upload itself.")

    if DATA["uploads"]:
        st.write("### Uploaded files")
        st.dataframe(DATA["uploads"][-20:][::-1], use_container_width=True)

    st.warning(
        "Your existing cry_predict.py is designed for live 5-second recording. "
        "Because you asked not to modify cry_predict.py, this page does not pretend "
        "that an uploaded video has been classified by that script."
    )

# ---------------- PROFILE ----------------
elif page == "Baby Profile":
    st.subheader("👶 Editable Baby Profile")

    fields = [
        ("name", "Baby Name"),
        ("dob", "Date of Birth"),
        ("age_months", "Age (months)"),
        ("mother_name", "Mother Name"),
        ("father_name", "Father Name"),
        ("birth_weight_kg", "Birth Weight (kg)"),
        ("current_weight_kg", "Current Weight (kg)"),
        ("blood_group", "Blood Group"),
    ]

    with st.form("profile_form"):
        values = {}
        cols = st.columns(2)
        for i, (key, label) in enumerate(fields):
            with cols[i % 2]:
                values[key] = st.text_input(
                    label,
                    value=str(DATA["profile"].get(key, "")),
                )
        notes = st.text_area(
            "Notes",
            value=str(DATA["profile"].get("notes", "")),
        )
        if st.form_submit_button("💾 Save Profile", type="primary"):
            DATA["profile"].update(values)
            DATA["profile"]["notes"] = notes
            save_data(DATA)
            st.success("Baby profile saved.")

    st.subheader("Care Schedule")
    with st.form("schedule_form"):
        schedule = {}
        for key, label in [
            ("last_feed", "Last Feed"),
            ("next_feed", "Next Feed"),
            ("last_diaper", "Last Diaper"),
            ("last_medication", "Last Medication"),
            ("sleep_start", "Sleep Started"),
        ]:
            schedule[key] = st.text_input(
                label, value=str(DATA["schedule"].get(key, ""))
            )
        if st.form_submit_button("💾 Save Schedule"):
            DATA["schedule"] = schedule
            save_data(DATA)
            st.success("Schedule saved.")

# ---------------- CARE CENTER ----------------
elif page == "Care Center":
    st.subheader("🏥 Care Center")
    st.caption("Functional care-center controls, not a decorative screen.")

    cc = DATA["care_center"]

    with st.form("care_center_form"):
        active = st.checkbox("Care Center Active", value=bool(cc.get("active", True)))
        name = st.text_input("Care Center Name", value=cc.get("name", "NICU / Care Center"))
        contact = st.text_input("Primary Contact", value=cc.get("contact", "Primary Caregiver"))
        phone = st.text_input("Contact Phone", value=cc.get("phone", ""))
        if st.form_submit_button("Save Care Center"):
            DATA["care_center"] = {
                "active": active,
                "name": name,
                "contact": contact,
                "phone": phone,
            }
            save_data(DATA)
            st.success("Care center settings saved.")

    st.divider()

    st.write("### Notification status")
    if TG.get("bot_token") and TG.get("primary_chat_id"):
        st.success("Telegram primary channel configured.")
    else:
        st.error("Telegram primary channel is not configured.")

    if TG.get("secondary_chat_id"):
        st.success("Telegram secondary escalation channel configured.")
    else:
        st.warning("Secondary escalation chat ID is not configured.")

    if st.button("📱 Test Primary Telegram"):
        msg = (
            "🧪 BABYPULSE CARE CENTER TEST\n"
            f"Baby: {DATA['profile'].get('name', 'Baby 1')}\n"
            "This is a notification test."
        )
        ok, detail = telegram_send(TG.get("primary_chat_id"), msg)
        if ok:
            st.success("Primary Telegram message sent.")
        else:
            st.error(detail)

    if st.button("📱 Test Secondary Telegram"):
        msg = (
            "🧪 BABYPULSE SECONDARY TEST\n"
            f"Baby: {DATA['profile'].get('name', 'Baby 1')}\n"
            "This is an escalation-channel test."
        )
        ok, detail = telegram_send(TG.get("secondary_chat_id"), msg)
        if ok:
            st.success("Secondary Telegram message sent.")
        else:
            st.error(detail)

    st.write("### GPIO")
    st.write(f"Green LED: GPIO {GREEN_GPIO}")
    st.write(f"Red LED: GPIO {RED_GPIO}")
    st.write(f"RPi.GPIO available: {'YES' if GPIO_AVAILABLE else 'NO'}")

# ---------------- ALERTS ----------------
elif page == "Alerts":
    check_escalation()
    st.subheader("🚨 Alerts & Escalation")

    if R["alert_label"]:
        elapsed = int(time.time() - (R["alert_started"] or time.time()))
        remaining = max(0, ESCALATION_SECONDS - elapsed)
        st.error(f"ACTIVE CRITICAL ALERT: {R['alert_label']}")
        st.write(f"Primary sent: {'YES' if R['primary_sent'] else 'NO'}")
        st.write(f"Secondary sent: {'YES' if R['secondary_sent'] else 'NO'}")
        st.write(f"Escalation countdown: {remaining} seconds")

        if st.button("✓ Clear Critical Alert"):
            clear_alert()
            record_alert(
                "Alert cleared",
                "Cleared by caregiver",
                "Critical alert cleared manually."
            )
            st.success("Alert cleared. Red LED OFF.")
    else:
        st.success("No active critical alert.")

    if st.button("🧪 Start Test Distress Alert"):
        start_alert("Distress")
        st.rerun()

    st.write("### Alert history")
    if DATA["alerts"]:
        st.dataframe(DATA["alerts"][-50:][::-1], use_container_width=True)
    else:
        st.info("No alerts recorded.")

# ---------------- REPORT ----------------
elif page == "Weekly Report":
    st.subheader("📊 Weekly Report")

    cutoff = dt.datetime.now() - dt.timedelta(days=7)
    recent = []

    for item in DATA["cry_history"]:
        try:
            t = dt.datetime.strptime(item["time"], "%d %b %Y %H:%M:%S")
            if t >= cutoff:
                recent.append(item)
        except Exception:
            recent.append(item)

    counts = {x: 0 for x in LABELS}
    for item in recent:
        label = item.get("label", "")
        if label in counts:
            counts[label] += 1

    c1, c2, c3 = st.columns(3)
    c1.metric("Classifications", len(recent))
    c2.metric("Pain", counts["Pain"])
    c3.metric("Distress", counts["Distress"])

    st.bar_chart(counts)
    st.dataframe(recent[::-1], use_container_width=True)

    if st.button("Export Report CSV"):
        report_path = BASE / "babypulse_weekly_report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["time", "label", "confidence", "source"]
            )
            writer.writeheader()
            writer.writerows(recent)
        st.success(f"Saved {report_path}")

# ---------------- SYSTEM ----------------
elif page == "System":
    st.subheader("⚙️ System")

    st.write("### Files")
    for p in [MONITOR_SCRIPT, PREDICT_SCRIPT, DATA_FILE, TELEGRAM_FILE]:
        st.write(f"{'✅' if p.exists() else '❌'} {p}")

    st.write("### Telegram")
    st.write(
        "Configured" if TG.get("bot_token") else "Not configured"
    )

    st.write("### Hardware")
    st.write(f"RPi.GPIO: {'available' if GPIO_AVAILABLE else 'not available'}")
    st.write(f"Green LED GPIO: {GREEN_GPIO}")
    st.write(f"Red LED GPIO: {RED_GPIO}")

    st.write("### Current process")
    proc = R.get("process")
    st.write(
        "Running" if proc is not None and proc.poll() is None else "Stopped"
    )

    if st.button("Turn both LEDs OFF"):
        set_leds(False, False)
        st.success("Both LEDs switched OFF.")

    if st.button("Set GREEN LED"):
        set_leds(True, False)
        st.success("Green LED switched ON.")

    if st.button("Set RED LED"):
        set_leds(False, True)
        st.error("Red LED switched ON.")

# ---------------- AUTO ESCALATION / REFRESH ----------------
check_escalation()

# Streamlit 1.64 supports fragments. This keeps the live page updating
# without requiring the user to click around.
try:
    @st.fragment(run_every="2s")
    def runtime_refresh():
        check_escalation()
        if R["alert_label"]:
            st.caption(
                f"Alert active: {R['alert_label']} | "
                f"Primary: {'sent' if R['primary_sent'] else 'pending'} | "
                f"Secondary: {'sent' if R['secondary_sent'] else 'pending'}"
            )
    runtime_refresh()
except Exception:
    pass
