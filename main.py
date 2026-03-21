"""
main.py  –  Care Home Monthly Calendar v2

Restructured with tabbed UI, structured rule editor,
persistent image library, and deduplicated PDF generation.
"""

import streamlit as st
import hashlib
import pandas as pd
import datetime as dt
import calendar
from io import BytesIO
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.utils import ImageReader
import re
import json
import os
import base64
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, black, white
from reportlab.lib.units import mm
import requests
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Care Home Activities Calendar",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Pexels API ───────────────────────────────────────────
PEXELS_API_KEY = st.secrets.get("PEXELS_API_KEY", "")
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

# ── Directory setup ──────────────────────────────────────
IMAGE_CACHE_DIR = "image_cache"
IMAGE_LIBRARY_DIR = os.path.join(IMAGE_CACHE_DIR, "library")
UPLOADS_DIR = "uploaded_csvs"
for d in [IMAGE_CACHE_DIR, IMAGE_LIBRARY_DIR, UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)

SETTINGS_FILE = "calendar_settings.json"


# ═══════════════════════════════════════════════════════════
# PERSISTENCE  – settings, monthly data, CSVs, image library
# ═══════════════════════════════════════════════════════════

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_settings(data):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving settings: {e}")


def save_monthly_data(year, month, data):
    filename = f"calendar_data_{year}_{month:02d}.json"
    try:
        serialisable = {d.isoformat(): v for d, v in data.items()}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)
    except Exception as e:
        st.error(f"Error saving monthly data: {e}")


def load_monthly_data(year, month):
    filename = f"calendar_data_{year}_{month:02d}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                raw = json.load(f)
                return {dt.date.fromisoformat(k): v for k, v in raw.items()}
        except Exception:
            return {}
    return {}


def save_uploaded_csv(uploaded_file, csv_type, year, month):
    if uploaded_file is None:
        return None
    filename = f"{csv_type}_{year}_{month:02d}.csv"
    filepath = os.path.join(UPLOADS_DIR, filename)
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error(f"❌ {csv_type.title()} CSV is empty.")
            return None
        if csv_type == "rota":
            required = ["date", "staff"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"❌ Staff Rota CSV missing columns: {', '.join(missing)}")
                return None
        elif csv_type == "activities":
            if "name" not in df.columns:
                st.error("❌ Activities CSV missing 'name' column.")
                return None
        df.to_csv(filepath, index=False)
        return filepath
    except pd.errors.EmptyDataError:
        st.error(f"❌ {csv_type.title()} CSV is completely empty.")
        return None
    except Exception as e:
        st.error(f"❌ Error reading {csv_type.title()} CSV: {e}")
        return None


def load_saved_csv(csv_type, year, month):
    filename = f"{csv_type}_{year}_{month:02d}.csv"
    filepath = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception:
            return None
    return None


def save_calendar_state(year, month, state_data):
    filename = f"calendar_state_{year}_{month:02d}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)
        return True
    except Exception:
        return False


def load_calendar_state(year, month):
    filename = f"calendar_state_{year}_{month:02d}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ── Image Library ────────────────────────────────────────
IMAGE_LIBRARY_FILE = os.path.join(IMAGE_CACHE_DIR, "image_library.json")


def load_image_library():
    if os.path.exists(IMAGE_LIBRARY_FILE):
        try:
            with open(IMAGE_LIBRARY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_image_library(library):
    try:
        with open(IMAGE_LIBRARY_FILE, "w", encoding="utf-8") as f:
            json.dump(library, f, indent=2)
    except Exception:
        pass


def save_library_image(activity_name, image_bytes, keyword=""):
    """Save an image to the library for a given activity name."""
    safe_name = re.sub(r"[^a-z0-9_]", "_", activity_name.lower().strip())
    filename = f"{safe_name}.jpg"
    filepath = os.path.join(IMAGE_LIBRARY_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        library = load_image_library()
        library[activity_name.lower().strip()] = {
            "filename": filename,
            "keyword": keyword,
        }
        save_image_library(library)
        return True
    except Exception:
        return False


def get_library_image(activity_name):
    """Get saved image bytes for an activity, or None."""
    library = load_image_library()
    key = activity_name.lower().strip()
    entry = library.get(key)
    if not entry:
        return None
    filepath = os.path.join(IMAGE_LIBRARY_DIR, entry["filename"])
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                return f.read()
        except Exception:
            return None
    return None


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

PRIORITY_KEYWORDS = [
    "quiz", "book", "bingo", "music", "painting", "baking", "gardening",
    "yoga", "exercise", "knitting", "singing", "walking", "tea", "coffee",
    "film", "movie", "dogs", "crafts", "dominoes", "cards", "scrabble", "jigsaw",
]

ACTIVITY_KEYWORDS = {
    "gardening": "gardening flowers nature",
    "dogs for health": "dogs therapy animals",
    "film night": "cinema movie film reel",
    "book club": "reading books library",
    "bookworms": "reading books cozy",
    "quiz": "trivia questions game",
    "pub quiz": "pub quiz game",
    "christmas crafts": "christmas decorations crafts",
    "remembrance": "poppy remembrance memorial",
    "poppy": "poppy flowers red",
    "baking": "baking cookies kitchen",
    "painting": "painting art creative",
    "music": "music instruments singing",
    "exercise": "seniors exercise fitness",
    "yoga": "seniors yoga stretching",
    "reminiscence": "memory nostalgia vintage",
    "bingo": "bingo game numbers",
    "balloon volleyball": "balloon games seniors",
    "target throw": "target game activity",
    "one-on-one": "conversation chat seniors",
    "coffee morning": "coffee tea social",
    "singing": "singing group music",
    "knitting": "knitting craft wool",
    "dominoes": "dominoes game seniors",
    "cards": "playing cards game",
    "scrabble": "scrabble word game",
    "jigsaw": "jigsaw puzzle",
    "walking": "walking nature outdoors",
    "afternoon tea": "tea sandwiches afternoon",
    "morning exercise": "seniors exercise fitness group",
    "news headlines": "newspaper reading morning",
}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def clean_text(s):
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    replacements = {
        "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\xa0": " ",
        "\r": " ", "\n": " ", "\u2028": " ", "\u2029": " ", "\ufeff": " ",
        "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    # Keep common accented characters (staff names etc.)
    s = re.sub(r"[^\x20-\x7E\u00C0-\u00FF]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


@st.cache_data
def load_all_holidays():
    try:
        with open("holidays_2025_2026.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("holidays", [])
    except Exception:
        return []


ALL_HOLIDAYS = load_all_holidays()


def month_date_range(year, month):
    first = dt.date(year, month, 1)
    last = dt.date(year, month, calendar.monthrange(year, month)[1])
    return first, last


def parse_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"CSV parse error: {e}")
        return None


def fetch_selected_holidays(year, month, selected_names=None):
    holidays_list = []
    selected_normalized = set()
    if selected_names:
        for s in selected_names:
            selected_normalized.add(clean_text(s).lower())
    for h in ALL_HOLIDAYS:
        try:
            d = dt.datetime.strptime(h["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if d.year == year and d.month == month:
            name = clean_text(h.get("name", ""))
            if (not selected_names) or (name.lower() in selected_normalized):
                holidays_list.append({"date": d, "title": name, "notes": "Holiday"})
    return holidays_list


def get_weeks_in_month(year, month):
    cal = calendar.monthcalendar(year, month)
    weeks = []
    for week in cal:
        valid_days = [day for day in week if day != 0]
        if not valid_days:
            continue
        start_date = dt.date(year, month, valid_days[0])
        end_date = dt.date(year, month, valid_days[-1])
        weeks.append((start_date, end_date))
    return weeks


# ═══════════════════════════════════════════════════════════
# CALENDAR BUILDING
# ═══════════════════════════════════════════════════════════

def seat_activity_into_calendar(year, month, activities_df, rota_df, rules,
                                include_holidays=True, daily_rules=None):
    first, last = month_date_range(year, month)
    daymap = {first + dt.timedelta(days=i): [] for i in range((last - first).days + 1)}

    # Holidays
    if include_holidays:
        seen_holidays = set()
        combined_holidays = fetch_selected_holidays(
            year, month, st.session_state.get("selected_holidays"))
        for ev in combined_holidays:
            d = ev["date"]
            title_norm = clean_text(ev["title"]).strip().lower()
            if (d, title_norm) in seen_holidays:
                continue
            seen_holidays.add((d, title_norm))
            if d in daymap:
                existing_titles = [e["title"] for e in daymap[d] if e["notes"] == "Holiday"]
                if existing_titles:
                    combined = " / ".join(sorted(set(existing_titles + [ev["title"]])))
                    daymap[d] = [e for e in daymap[d] if e["notes"] != "Holiday"]
                    daymap[d].append({"time": None, "title": combined, "notes": "Holiday"})
                else:
                    daymap[d].append({"time": None, "title": ev["title"], "notes": "Holiday"})

    # Rota
    if rota_df is not None:
        for _, r in rota_df.iterrows():
            try:
                d = pd.to_datetime(r.get("date")).date()
            except Exception:
                continue
            if d in daymap:
                staff = clean_text(str(r.get("staff", "")))
                staff = re.sub(r"\s*\d+$", "", staff)
                start = str(r.get("shift_start", "")).strip()
                end = str(r.get("shift_end", "")).strip()
                shift_time = f"({start} - {end})" if start and end else ""
                display = f"{staff} {shift_time}".strip()
                if display:
                    daymap[d].append({"time": None, "title": display, "notes": "staff shift"})

    # Fixed weekly rules
    fixed_rules = []
    for rule in rules:
        for d in daymap:
            if d.weekday() == rule["weekday"]:
                fixed_rules.append({
                    "date": d, "time": rule.get("time"),
                    "title": clean_text(rule["title"]), "notes": "fixed",
                })

    # Fixed daily rules
    if daily_rules:
        for d in daymap:
            for rule in daily_rules:
                daymap[d].append({
                    "date": d, "time": rule.get("time"),
                    "title": clean_text(rule["title"]), "notes": "fixed daily",
                })

    # Activities from CSV
    activities = []
    if activities_df is not None:
        for _, r in activities_df.iterrows():
            name = clean_text(r.get("name") or r.get("activity_name") or "")
            pref_days = str(r.get("preferred_days", "")).split(";")
            pref_days = [p.strip()[:3].lower() for p in pref_days if p.strip()]
            pref_time = str(r.get("preferred_time", "")).strip()
            freq = int(r.get("frequency", 0)) if str(r.get("frequency", "")).isdigit() else 0
            interval = int(r.get("interval", 1)) if str(r.get("interval", "")).isdigit() else 1
            week_type = str(r.get("week_type", "")).strip().lower()
            placed = 0

            # Specific date support
            activity_date_raw = r.get("date") if "date" in r.index else None
            specific_date = None
            if pd.notna(activity_date_raw) and activity_date_raw:
                try:
                    specific_date = pd.to_datetime(activity_date_raw).date()
                except Exception:
                    specific_date = None
            if specific_date:
                if specific_date in daymap:
                    activities.append({"date": specific_date, "time": pref_time, "title": name, "notes": "activity"})
                continue

            # Standard scheduling
            for d in sorted(daymap.keys()):
                if freq and placed >= freq:
                    break
                dow3 = calendar.day_name[d.weekday()][:3].lower()
                if dow3 not in pref_days:
                    continue
                if interval == 2:
                    week_num = d.isocalendar()[1]
                    if week_type == "odd" and week_num % 2 == 0:
                        continue
                    elif week_type == "even" and week_num % 2 == 1:
                        continue
                    elif not week_type and placed > 0:
                        continue
                activities.append({"date": d, "time": pref_time, "title": name, "notes": "activity"})
                placed += 1

    # Normalize times
    time_pattern = re.compile(r"^(\d{1,2})(?::?(\d{2}))?$")

    def normalize_time(t):
        if not t or not isinstance(t, str):
            return None
        t2 = t.strip().lower().replace(".", ":").replace(" ", "")
        match = time_pattern.match(t2)
        if match:
            hour, minute = match.groups()
            return f"{hour.zfill(2)}:{minute if minute else '00'}"
        return None

    all_events = fixed_rules + activities
    for ev in all_events:
        ev["time"] = normalize_time(ev.get("time"))

    for ev in all_events:
        d = ev["date"]
        if d not in daymap:
            continue
        title_norm = ev["title"].lower().strip()
        time_norm = ev.get("time")
        duplicates = [e for e in daymap[d] if e["title"].lower().strip() == title_norm]
        if duplicates:
            has_exact = any(e.get("time") == time_norm for e in duplicates)
            has_proper = any(e.get("time") and len(e.get("time")) == 5 for e in duplicates)
            if has_exact or (has_proper and not time_norm):
                continue
        daymap[d].append({"time": time_norm, "title": ev["title"], "notes": ev["notes"]})

    def sort_key(e):
        t = e.get("time")
        if not t:
            return dt.time(23, 59)
        try:
            h, m = map(int, t.split(":"))
            return dt.time(h, m)
        except Exception:
            return dt.time(23, 59)

    for d in daymap:
        daymap[d].sort(key=lambda e: (
            0 if e["notes"] == "Holiday" else
            1 if e["notes"] == "staff shift" else 2,
            sort_key(e),
        ))

    return daymap


# ═══════════════════════════════════════════════════════════
# PDF GENERATION
# ═══════════════════════════════════════════════════════════

def draw_calendar_pdf(title, disclaimer, year, month, cell_texts, background_bytes=None):
    """Create an A3 landscape monthly calendar PDF."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A3))
    width, height = landscape(A3)

    if background_bytes:
        try:
            img = ImageReader(BytesIO(background_bytes))
            c.drawImage(img, 0, 0, width=width, height=height, preserveAspectRatio=False, mask="auto")
        except Exception:
            pass

    title_text = clean_text(title)
    disclaimer_text = clean_text(disclaimer)

    # Title pill
    c.setFont("Helvetica-Bold", 20)
    tw = c.stringWidth(title_text, "Helvetica-Bold", 20)
    pill_w = tw + 15 * mm
    pill_h = 4 * mm + 4 * mm
    pill_y = height - 10 * mm
    pill_x = (width - pill_w) / 2
    c.setFillColor(Color(0, 0, 0))
    c.roundRect(pill_x, pill_y, pill_w, pill_h, pill_h / 2, fill=1, stroke=0)
    c.setFillColor(white)
    c.drawCentredString(width / 2, pill_y + pill_h / 2 - 20 / 3.2, title_text)

    # Disclaimer pill
    c.setFont("Helvetica-Bold", 11)
    dw = c.stringWidth(disclaimer_text, "Helvetica-Bold", 11)
    disc_w = dw + 10 * mm
    disc_h = 6 * mm + 1 * mm
    disc_x = (width - disc_w) / 2
    disc_y = pill_y - disc_h - 0.5 * mm
    c.setFillColor(Color(0, 0, 0))
    c.roundRect(disc_x, disc_y, disc_w, disc_h, disc_h / 2, fill=1, stroke=0)
    c.setFillColor(white)
    c.drawCentredString(width / 2, disc_y + disc_h / 2 - 11 / 3, disclaimer_text)

    # Grid
    left, right, top = 4 * mm, 4 * mm, 37 * mm
    grid_w = width - left - right
    cols = 7
    month_cal = calendar.monthcalendar(year, month)
    rows = len(month_cal)
    col_w = grid_w / cols

    # Weekday header
    bar_height = 8 * mm
    bar_y = height - top + 11 * mm
    c.setFillColor(Color(0, 0, 0))
    c.rect(left, bar_y, grid_w, bar_height, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 15)
    for i, wd in enumerate(WEEKDAYS):
        c.drawCentredString(left + i * col_w + col_w / 2, bar_y + 2.5 * mm, wd)

    bar_gap = 1.5 * mm
    top_of_grid = bar_y - bar_gap
    bottom = {6: 8, 5: 6}.get(rows, 5) * mm
    grid_h = top_of_grid - bottom
    row_h = grid_h / rows

    cream = Color(1, 1, 1, alpha=0.93)
    staff_blue = Color(0, 0.298, 0.6)

    for r_idx, week in enumerate(month_cal):
        for c_idx, day in enumerate(week):
            if day == 0:
                continue
            d = dt.date(year, month, day)
            x = left + c_idx * col_w
            y = bottom + (rows - 1 - r_idx) * row_h

            c.setFillColor(cream)
            c.setStrokeColor(black)
            c.roundRect(x, y, col_w, row_h, 5, fill=1, stroke=1)

            # Day number (top-right)
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(black)
            day_str = str(day)
            day_w = c.stringWidth(day_str, "Helvetica-Bold", 12)
            c.drawString(x + col_w - day_w - 1.2 * mm, y + row_h - 4.5 * mm, day_str)

            # Cell content
            lines = cell_texts.get(d, "").split("\n")
            text_y = y + row_h - 3.5 * mm
            line_spacing = 4 * mm

            for line in lines:
                line = clean_text(line).strip()
                if not line:
                    continue

                # Holiday: uppercase, underlined
                if line.isupper():
                    max_tw = col_w - (day_w + 6 * mm)
                    words = line.split()
                    cur = ""
                    wrapped = []
                    for w in words:
                        test = (cur + " " + w).strip()
                        if c.stringWidth(test, "Helvetica-Bold", 8.7) > max_tw and cur:
                            wrapped.append(cur)
                            cur = w
                        else:
                            cur = test
                    if cur:
                        wrapped.append(cur)
                    for wh in wrapped:
                        wh = wh.strip()
                        if not wh:
                            continue
                        c.setFont("Helvetica-Bold", 8.7)
                        c.setFillColor(black)
                        c.drawString(x + 2 * mm, text_y, wh)
                        tw2 = c.stringWidth(wh, "Helvetica-Bold", 8.7)
                        c.line(x + 2 * mm, text_y - 0.5 * mm, x + 2 * mm + tw2, text_y - 0.5 * mm)
                        text_y -= line_spacing
                    continue

                max_tw = col_w - (day_w + 0.5 * mm)
                c.setFont("Helvetica-Bold", 10.5)

                # Staff lines
                if line.lower().startswith("staff:"):
                    c.setFont("Helvetica-Oblique", 10.5)
                    c.setFillColor(staff_blue)
                    c.drawString(x + 2 * mm, text_y, line)
                    text_y -= line_spacing - 1
                    continue

                # Time + activity
                time_match = re.match(r"^(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?)\s?(.*)", line)
                if time_match:
                    time_part, rest = time_match.groups()
                    rest = rest.strip()
                    c.setFont("Helvetica-Bold", 10.5)
                    c.setFillColor(black)
                    c.drawString(x + 2 * mm, text_y, time_part)
                    time_w = c.stringWidth(time_part + " ", "Helvetica-Bold", 10.5)
                    avail = max_tw - time_w

                    words = rest.split()
                    cur = ""
                    wrapped = []
                    for w in words:
                        test = (cur + " " + w).strip()
                        if c.stringWidth(test, "Helvetica-Bold", 10.5) > avail and cur:
                            wrapped.append(cur)
                            cur = w
                        else:
                            cur = test
                    if cur:
                        wrapped.append(cur)

                    first_line = True
                    for wl in wrapped:
                        wl = wl.strip()
                        if not wl:
                            continue
                        if first_line:
                            c.drawString(x + 2 * mm + time_w, text_y, wl)
                            first_line = False
                        else:
                            text_y -= line_spacing
                            c.drawString(x + 2 * mm, text_y, wl)
                    text_y -= line_spacing
                    if text_y < y + 4 * mm:
                        break
                    continue

                # Normal wrapping
                c.setFillColor(black)
                words = line.split()
                cur = ""
                wrapped = []
                for w in words:
                    test = (cur + " " + w).strip()
                    if c.stringWidth(test, "Helvetica-Bold", 10.5) > max_tw and cur:
                        wrapped.append(cur)
                        cur = w
                    else:
                        cur = test
                if cur:
                    wrapped.append(cur)
                for sl in wrapped:
                    sl = sl.strip()
                    if not sl:
                        continue
                    c.drawString(x + 2 * mm, text_y, sl)
                    text_y -= line_spacing
                    if text_y < y + 4 * mm:
                        break

    c.save()
    buffer.seek(0)
    return buffer


def draw_weekly_page(c, width, height, day_obj, text, image_bytes_list=None,
                     image_layouts=None, text_sizes=None):
    """Draw a single day page on A4 landscape with custom positioned images."""
    if text_sizes is None:
        text_sizes = {"day_heading": 40, "disclaimer": 12, "staff": 15, "activities": 22, "holidays": 15}

    text_area_right = width * 0.62

    # Day heading
    c.setFont("Helvetica-Bold", text_sizes["day_heading"])
    day_str = f"{calendar.day_name[day_obj.weekday()]} {day_obj.day} {calendar.month_name[day_obj.month]}"
    c.drawString(10 * mm, height - 20 * mm, day_str)

    # Disclaimer
    c.setFont("Helvetica-Oblique", text_sizes["disclaimer"])
    disclaimer = (
        "Activities may change due to unforeseen circumstances. "
        "Families are welcome to join. "
        "Weather permitting, activities may move outdoors."
    )
    max_w = text_area_right - 20 * mm
    wrapped = _wrap_text_reportlab(c, disclaimer, "Helvetica-Oblique", text_sizes["disclaimer"], max_w)

    text_y = height - 30 * mm
    for line in wrapped:
        c.drawString(10 * mm, text_y, line)
        text_y -= 6 * mm

    y = text_y - 8 * mm

    # Images
    if image_bytes_list and image_layouts:
        try:
            for img_bytes, layout in zip(image_bytes_list, image_layouts):
                img = ImageReader(BytesIO(img_bytes))
                c.setFillColor(Color(0.95, 0.95, 0.95))
                c.roundRect(layout["x"] - 3 * mm, layout["y"] - 3 * mm,
                            layout["width"] + 6 * mm, layout["height"] + 6 * mm,
                            8, fill=1, stroke=0)
                c.drawImage(img, layout["x"], layout["y"],
                            width=layout["width"], height=layout["height"],
                            preserveAspectRatio=True, mask="auto")
        except Exception:
            pass

    # Text content
    staff_lines = []
    other_lines = []
    for line in text.split("\n"):
        line = clean_text(line)
        if not line:
            continue
        if line.lower().startswith("staff:"):
            staff_lines.append(line.strip())
        else:
            other_lines.append(line.strip())

    staff_blue = Color(0, 0.298, 0.6)
    if staff_lines:
        combined_staff = " - ".join(staff_lines)
        wrapped = _wrap_text_reportlab(c, combined_staff, "Helvetica-Oblique", text_sizes["staff"], text_area_right - 20 * mm)
        c.setFont("Helvetica-Oblique", text_sizes["staff"])
        c.setFillColor(staff_blue)
        for w in wrapped:
            c.drawString(10 * mm, y, w)
            y -= 9 * mm
        y -= 5 * mm

    # Activities
    merged = {}
    for line in other_lines:
        match = re.match(r"^(\d{1,2}:\d{2})\s*(.*)", line)
        if match:
            time, desc = match.groups()
            merged.setdefault(time, []).append(desc.strip())
        else:
            merged.setdefault(None, []).append(line.strip())

    for time_key, desc_list in merged.items():
        is_holiday = all(d.isupper() for d in desc_list)
        if is_holiday:
            combined = (" / ".join(desc_list) if time_key is None
                        else f"{time_key}: " + " / ".join(desc_list))
            font_size = text_sizes["holidays"]
            c.setFont("Helvetica-Bold", font_size)
            c.setFillColor(black)
        else:
            combined = (" → ".join(desc_list) if time_key is None
                        else f"{time_key}: " + " → ".join(desc_list))
            font_size = text_sizes["activities"]
            c.setFont("Helvetica-Bold", font_size)
            c.setFillColor(Color(0.1, 0.1, 0.1))

        wrapped = _wrap_text_reportlab(c, combined, "Helvetica-Bold", font_size, text_area_right - 20 * mm)
        spacing = 7 * mm if is_holiday else 8 * mm
        for w in wrapped:
            c.drawString(10 * mm, y, w.strip())
            y -= spacing
        y -= 6 * mm
        if y < 25 * mm:
            break


def _wrap_text_reportlab(c, text, font_name, font_size, max_width):
    """Wrap text for ReportLab canvas."""
    c.setFont(font_name, font_size)
    words = text.split()
    current = ""
    lines = []
    for w in words:
        test = (current + " " + w).strip()
        if c.stringWidth(test, font_name, font_size) > max_width and current:
            lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def generate_week_pdf(week_days, session_key, text_sizes):
    """Generate a PDF for a list of days. Returns BytesIO buffer."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(A4))
    w, h = landscape(A4)

    for d in week_days:
        # Try text_area widget state first, then fall back to calendar dict
        text = st.session_state.get(f"{session_key}_{d}", "").strip()
        if not text and session_key in st.session_state:
            cal_data = st.session_state[session_key]
            if isinstance(cal_data, dict):
                text = cal_data.get(d, "").strip()
        if not text:
            text = "(No activities planned)"

        d_key = d.isoformat()
        activities = extract_activities_from_text(text)
        unique = _dedupe_activities(activities)

        # Auto-assign images from library
        images_list = []
        for act in unique:
            lib_img = get_library_image(act)
            if lib_img:
                images_list.append(lib_img)
            if len(images_list) >= 3:
                break

        # Override with manually selected images if available
        if f"selected_images_{d_key}" in st.session_state:
            manual = st.session_state[f"selected_images_{d_key}"]
            if manual:
                images_list = manual[:3]

        layouts = get_default_image_layout(len(images_list), w, h) if images_list else None

        # Check for custom layouts
        if "image_layouts" in st.session_state and d_key in st.session_state.image_layouts:
            layouts = st.session_state.image_layouts[d_key]

        draw_weekly_page(c, w, h, d, text, images_list, layouts, text_sizes)
        c.showPage()

    c.save()
    buf.seek(0)
    return buf


def get_default_image_layout(num_images, page_width, page_height):
    defaults = [
        {"x": 560, "y": 400, "width": 240, "height": 150},
        {"x": 560, "y": 220, "width": 240, "height": 150},
        {"x": 560, "y": 50, "width": 240, "height": 150},
    ]
    return [defaults[i].copy() for i in range(min(num_images, len(defaults)))]


def create_preview_image(width, height, day_obj, text,
                         image_bytes_list=None, image_layouts=None, text_sizes=None):
    """Create a PIL preview image matching the PDF layout."""
    if text_sizes is None:
        text_sizes = {"day_heading": 40, "disclaimer": 12, "staff": 15, "activities": 22, "holidays": 15}

    img = Image.new('RGB', (int(width), int(height)), color='white')
    draw = ImageDraw.Draw(img)
    text_area_right = int(width * 0.62)
    pt = 2.83465  # mm to points

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", text_sizes["day_heading"])
        disc_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", text_sizes["disclaimer"])
        staff_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", text_sizes["staff"])
        act_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", text_sizes["activities"])
        hol_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", text_sizes["holidays"])
    except Exception:
        title_font = disc_font = staff_font = act_font = hol_font = ImageFont.load_default()

    # Day heading
    day_str = f"{calendar.day_name[day_obj.weekday()]} {day_obj.day} {calendar.month_name[day_obj.month]}"
    draw.text((int(10 * pt), int(20 * pt)), day_str, fill='black', font=title_font)

    # Disclaimer
    disclaimer = ("Activities may change due to unforeseen circumstances. "
                  "Families are welcome to join. Weather permitting, activities may move outdoors.")
    max_dw = text_area_right - int(20 * pt)
    wrapped = _wrap_text_pil(draw, disclaimer, disc_font, max_dw)
    text_y = int(30 * pt)
    for line in wrapped:
        draw.text((int(10 * pt), text_y), line, fill='gray', font=disc_font)
        text_y += int(6 * pt)

    y_pos = text_y + int(8 * pt)

    # Images
    if image_bytes_list and image_layouts:
        for img_bytes, layout in zip(image_bytes_list, image_layouts):
            try:
                pil_img = Image.open(BytesIO(img_bytes))
                pil_img = pil_img.resize((int(layout["width"]), int(layout["height"])), Image.Resampling.LANCZOS)
                bg_x = int(layout["x"] - 3 * pt)
                bg_y = int(height - layout["y"] - layout["height"] - 3 * pt)
                bg_w = int(layout["width"] + 6 * pt)
                bg_h = int(layout["height"] + 6 * pt)
                draw.rounded_rectangle([bg_x, bg_y, bg_x + bg_w, bg_y + bg_h], radius=8, fill=(242, 242, 242))
                img.paste(pil_img, (int(layout["x"]), int(height - layout["y"] - layout["height"])))
            except Exception:
                pass

    # Staff lines
    staff_lines = []
    other_lines = []
    for line in text.split("\n"):
        line = clean_text(line).strip()
        if not line:
            continue
        if line.lower().startswith("staff:"):
            staff_lines.append(line.strip())
        else:
            other_lines.append(line.strip())

    if staff_lines:
        combined = " - ".join(staff_lines)
        wrapped = _wrap_text_pil(draw, combined, staff_font, text_area_right - int(20 * pt))
        for w in wrapped:
            draw.text((int(10 * pt), y_pos), w, fill=(0, 76, 153), font=staff_font)
            y_pos += int(9 * pt)
        y_pos += int(5 * pt)

    # Activities
    merged = {}
    for line in other_lines:
        match = re.match(r"^(\d{1,2}:\d{2})\s*(.*)", line)
        if match:
            t, desc = match.groups()
            merged.setdefault(t, []).append(desc.strip())
        else:
            merged.setdefault(None, []).append(line.strip())

    for time_key, desc_list in merged.items():
        is_hol = all(d.isupper() for d in desc_list)
        if is_hol:
            combined = (" / ".join(desc_list) if time_key is None else f"{time_key}: " + " / ".join(desc_list))
            font = hol_font
            color = (0, 0, 0)
            spacing = int(7 * pt)
        else:
            combined = (" → ".join(desc_list) if time_key is None else f"{time_key}: " + " → ".join(desc_list))
            font = act_font
            color = (26, 26, 26)
            spacing = int(8 * pt)

        wrapped = _wrap_text_pil(draw, combined, font, text_area_right - int(20 * pt))
        for w in wrapped:
            draw.text((int(10 * pt), y_pos), w.strip(), fill=color, font=font)
            y_pos += spacing
        y_pos += int(6 * pt)
        if y_pos > height - int(25 * pt):
            break

    return img


def _wrap_text_pil(draw, text, font, max_width):
    """Wrap text for PIL drawing."""
    words = text.split()
    current = ""
    lines = []
    for w in words:
        test = (current + " " + w).strip()
        try:
            bbox = draw.textbbox((0, 0), test, font=font)
            lw = bbox[2] - bbox[0]
        except Exception:
            lw = len(test) * 7
        if lw > max_width and current:
            lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)
    return lines


# ═══════════════════════════════════════════════════════════
# PEXELS INTEGRATION
# ═══════════════════════════════════════════════════════════

def get_activity_keyword(activity_name):
    activity_lower = activity_name.lower().strip()
    for keyword in PRIORITY_KEYWORDS:
        if keyword in activity_lower:
            if keyword == "book":
                return "books reading"
            elif keyword == "quiz":
                return "quiz trivia"
            return keyword
    if activity_lower in ACTIVITY_KEYWORDS:
        return ACTIVITY_KEYWORDS[activity_lower]
    for key, value in ACTIVITY_KEYWORDS.items():
        if key in activity_lower:
            return value
    cleaned = activity_lower.replace("club", "").strip()
    return cleaned if cleaned else "seniors activity"


def fetch_pexels_images(keyword, count=5, page=1):
    if not PEXELS_API_KEY:
        return []
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": keyword, "orientation": "landscape", "per_page": count, "page": page}
    images = []
    try:
        resp = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for photo in data.get("photos", []):
            img_url = photo["src"]["medium"]
            img_resp = requests.get(img_url, timeout=10)
            img_resp.raise_for_status()
            images.append(img_resp.content)
    except Exception:
        pass
    return images


def fetch_images_parallel(activities, page_numbers):
    results = {}
    def _fetch(idx, activity, page_num):
        kw = get_activity_keyword(activity)
        imgs = fetch_pexels_images(kw, count=5, page=page_num)
        return idx, imgs, kw

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_fetch, i, a, p) for i, (a, p) in enumerate(zip(activities, page_numbers))]
        for future in as_completed(futures):
            try:
                idx, imgs, kw = future.result()
                results[idx] = {"images": imgs, "keyword": kw}
            except Exception:
                pass
    return results


def extract_activities_from_text(text):
    activities = []
    for line in text.split("\n"):
        line = clean_text(line).strip()
        if not line or line.isupper() or line.lower().startswith("staff:"):
            continue
        line = re.sub(r"^\d{1,2}:\d{2}:?\s*", "", line)
        parts = re.split(r"\s*→\s*", line)
        for part in parts:
            part = part.strip()
            if part and not part.lower().startswith("staff"):
                activities.append(part)
    return activities


def _dedupe_activities(activities):
    seen = set()
    unique = []
    for a in activities:
        key = a.lower().strip()
        if key not in seen:
            unique.append(a)
            seen.add(key)
    return unique


# ═══════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════

REAL_PASSWORD = st.secrets["APP_PASSWORD"]
PASSWORD_HASH = hashlib.sha256(REAL_PASSWORD.encode()).hexdigest()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔒 Care Home Calendar – Login")
    password = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


# ═══════════════════════════════════════════════════════════
# INITIALISE SESSION STATE
# ═══════════════════════════════════════════════════════════

if "settings" not in st.session_state:
    st.session_state["settings"] = load_settings()

settings = st.session_state["settings"]

# Weekly rules as structured data
if "weekly_rules" not in st.session_state:
    raw = settings.get("weekly_rules_structured", [])
    if raw:
        st.session_state.weekly_rules = raw
    else:
        # Parse legacy text format
        legacy = settings.get("weekly_rules", "Film Night:Thu:18:00\nDogs for Health:Thu:11:00\nReminiscence:Sat:18:00")
        parsed = []
        for line in legacy.splitlines():
            parts = [p.strip() for p in line.split(":")]
            if len(parts) >= 2:
                day_str = parts[1][:3]
                time_str = parts[2] if len(parts) > 2 else ""
                parsed.append({"title": parts[0], "day": day_str, "time": time_str})
        st.session_state.weekly_rules = parsed if parsed else [
            {"title": "Film Night", "day": "Thu", "time": "18:00"},
            {"title": "Dogs for Health", "day": "Thu", "time": "11:00"},
            {"title": "Reminiscence", "day": "Sat", "time": "18:00"},
        ]

if "daily_rules" not in st.session_state:
    raw = settings.get("daily_rules_structured", [])
    if raw:
        st.session_state.daily_rules = raw
    else:
        legacy = settings.get("daily_rules", "Morning Exercise:09:00\nNews Headlines:10:00")
        parsed = []
        for line in legacy.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(":", 1)]
            parsed.append({"title": parts[0], "time": parts[1] if len(parts) > 1 else ""})
        st.session_state.daily_rules = parsed if parsed else [
            {"title": "Morning Exercise", "time": "09:00"},
            {"title": "News Headlines", "time": "10:00"},
        ]

if "image_layouts" not in st.session_state:
    st.session_state.image_layouts = {}
if "selected_images" not in st.session_state:
    st.session_state.selected_images = {}
if "image_options" not in st.session_state:
    st.session_state.image_options = {}

text_sizes = settings.get("text_sizes", {
    "day_heading": 40, "disclaimer": 12, "staff": 15, "activities": 22, "holidays": 15
})


# ═══════════════════════════════════════════════════════════
# MAIN UI – TABBED LAYOUT
# ═══════════════════════════════════════════════════════════

st.title("🏡 Care Home Activities Calendar")

# ── Global controls (always visible) ─────────────────────
col_y, col_m, col_t, col_d = st.columns([1, 1, 2, 2])
with col_y:
    year = st.number_input("Year", 2024, 2035, dt.date.today().year)
with col_m:
    month = st.selectbox("Month", range(1, 13), index=dt.date.today().month - 1,
                         format_func=lambda x: calendar.month_name[x])
with col_t:
    title = st.text_input("Calendar title", f"{calendar.month_name[month]} {year}")
with col_d:
    disclaimer = st.text_input("Disclaimer", "Activities subject to change. Please confirm with staff.")

session_key = f"{year}-{month:02d}"

# Load persisted data
if session_key not in st.session_state:
    loaded = load_monthly_data(year, month)
    if loaded:
        st.session_state[session_key] = loaded
    else:
        loaded_state = load_calendar_state(year, month)
        if loaded_state and loaded_state.get("calendar_data"):
            try:
                st.session_state[session_key] = {
                    dt.date.fromisoformat(k): v
                    for k, v in loaded_state["calendar_data"].items()
                }
            except Exception:
                pass

saved_rota = load_saved_csv("rota", year, month)
saved_activities = load_saved_csv("activities", year, month)

# ── Tabs ─────────────────────────────────────────────────
tab_setup, tab_monthly, tab_weekly, tab_settings = st.tabs([
    "📋 Setup & Rules", "📅 Monthly Calendar", "🖼️ Weekly Exports", "⚙️ Settings"
])


# ═══════════════════════════════════════════════════════════
# TAB 1: SETUP & RULES
# ═══════════════════════════════════════════════════════════
with tab_setup:

    # ── CSV Uploads ──────────────────────────────────────
    st.subheader("Upload data files")
    col_rota, col_act = st.columns(2)

    with col_rota:
        st.markdown("**Staff Rota**")
        if saved_rota is not None:
            st.success(f"✅ Saved rota loaded ({len(saved_rota)} rows)")
            if st.button("Upload new rota", key="btn_new_rota"):
                st.session_state["upload_new_rota"] = True
                st.rerun()

        if saved_rota is None or st.session_state.get("upload_new_rota"):
            uploaded = st.file_uploader("CSV with columns: date, staff, shift_start, shift_end",
                                        type=["csv"], key="rota_upload")
            if uploaded is not None:
                fh = hashlib.md5(uploaded.getvalue()).hexdigest()
                if st.session_state.get("last_rota_hash") != fh:
                    df = parse_csv(uploaded)
                    if df is not None and not df.empty:
                        uploaded.seek(0)
                        if save_uploaded_csv(uploaded, "rota", year, month):
                            st.session_state["upload_new_rota"] = False
                            st.session_state["last_rota_hash"] = fh
                            st.rerun()

    rota_df = saved_rota if saved_rota is not None and not st.session_state.get("upload_new_rota") else None

    with col_act:
        st.markdown("**Activities**")
        if saved_activities is not None:
            st.success(f"✅ Saved activities loaded ({len(saved_activities)} rows)")
            if st.button("Upload new activities", key="btn_new_act"):
                st.session_state["upload_new_activities"] = True
                st.rerun()

        if saved_activities is None or st.session_state.get("upload_new_activities"):
            uploaded = st.file_uploader("CSV with columns: name, preferred_days, preferred_time, frequency",
                                        type=["csv"], key="act_upload")
            if uploaded is not None:
                fh = hashlib.md5(uploaded.getvalue()).hexdigest()
                if st.session_state.get("last_act_hash") != fh:
                    df = parse_csv(uploaded)
                    if df is not None and not df.empty:
                        uploaded.seek(0)
                        if save_uploaded_csv(uploaded, "activities", year, month):
                            st.session_state["upload_new_activities"] = False
                            st.session_state["last_act_hash"] = fh
                            st.rerun()

    activities_df = saved_activities if saved_activities is not None and not st.session_state.get("upload_new_activities") else None

    st.markdown("---")

    # ── Structured Rule Editor ───────────────────────────
    st.subheader("Weekly activity rules")
    st.caption("Activities that happen on the same day every week")

    for i, rule in enumerate(st.session_state.weekly_rules):
        cols = st.columns([3, 2, 2, 1])
        with cols[0]:
            st.session_state.weekly_rules[i]["title"] = st.text_input(
                "Activity", value=rule["title"], key=f"wr_title_{i}", label_visibility="collapsed")
        with cols[1]:
            day_idx = WEEKDAYS.index(rule["day"]) if rule["day"] in WEEKDAYS else 0
            chosen_day = st.selectbox("Day", WEEKDAYS, index=day_idx, key=f"wr_day_{i}", label_visibility="collapsed")
            st.session_state.weekly_rules[i]["day"] = chosen_day
        with cols[2]:
            st.session_state.weekly_rules[i]["time"] = st.text_input(
                "Time", value=rule["time"], key=f"wr_time_{i}",
                placeholder="HH:MM", label_visibility="collapsed")
        with cols[3]:
            if st.button("✕", key=f"wr_del_{i}", help="Remove this rule"):
                st.session_state.weekly_rules.pop(i)
                st.rerun()

    if st.button("＋ Add weekly rule", key="add_weekly"):
        st.session_state.weekly_rules.append({"title": "", "day": "Mon", "time": ""})
        st.rerun()

    st.markdown("---")

    st.subheader("Daily activity rules")
    st.caption("Activities that happen every single day")

    for i, rule in enumerate(st.session_state.daily_rules):
        cols = st.columns([4, 2, 1])
        with cols[0]:
            st.session_state.daily_rules[i]["title"] = st.text_input(
                "Activity", value=rule["title"], key=f"dr_title_{i}", label_visibility="collapsed")
        with cols[1]:
            st.session_state.daily_rules[i]["time"] = st.text_input(
                "Time", value=rule["time"], key=f"dr_time_{i}",
                placeholder="HH:MM", label_visibility="collapsed")
        with cols[2]:
            if st.button("✕", key=f"dr_del_{i}", help="Remove this rule"):
                st.session_state.daily_rules.pop(i)
                st.rerun()

    if st.button("＋ Add daily rule", key="add_daily"):
        st.session_state.daily_rules.append({"title": "", "time": ""})
        st.rerun()

    st.markdown("---")

    # ── Holiday Selection ────────────────────────────────
    include_holidays = st.checkbox("Include UK national holidays", True, key="inc_hols")

    if include_holidays:
        st.subheader("Select holidays to include")

        holidays_by_day = {}
        for h in ALL_HOLIDAYS:
            try:
                d = dt.datetime.strptime(h["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if d.year == year and d.month == month:
                holidays_by_day.setdefault(d, []).append(h["name"])

        if not holidays_by_day:
            st.info("No holidays found for this month.")
        else:
            all_names = {n for names in holidays_by_day.values() for n in names}

            if "selected_holidays" not in st.session_state or not st.session_state.get("selected_holidays"):
                st.session_state["selected_holidays"] = list(all_names)

            col_sel, col_clr = st.columns([1, 1])
            with col_sel:
                if st.button("Select all holidays", key="sel_all_hol"):
                    st.session_state["selected_holidays"] = list(all_names)
                    st.rerun()
            with col_clr:
                if st.button("Clear all holidays", key="clr_all_hol"):
                    st.session_state["selected_holidays"] = []
                    st.rerun()

            current_selection = set()
            saved_sel = set(st.session_state.get("selected_holidays", []))
            month_days = calendar.monthcalendar(year, month)

            for week in month_days:
                cols = st.columns(7)
                for c_idx, day in enumerate(week):
                    if day == 0:
                        continue
                    date_obj = dt.date(year, month, day)
                    day_hols = holidays_by_day.get(date_obj, [])
                    with cols[c_idx]:
                        st.markdown(f"**{calendar.month_abbr[month]} {day}**")
                        if not day_hols:
                            st.caption("No holidays")
                        else:
                            for name in sorted(set(day_hols)):
                                key = f"hol_{year}-{month:02d}-{day:02d}_{name}"
                                if st.checkbox(name, value=name in saved_sel, key=key):
                                    current_selection.add(name)

            st.session_state["selected_holidays"] = list(current_selection)

    st.markdown("---")

    # ── Save Rules ───────────────────────────────────────
    if st.button("💾 Save all rules", type="primary", key="save_rules"):
        settings["weekly_rules_structured"] = st.session_state.weekly_rules
        settings["daily_rules_structured"] = st.session_state.daily_rules
        # Also save legacy format for backward compatibility
        legacy_weekly = "\n".join(f"{r['title']}:{r['day']}:{r['time']}" for r in st.session_state.weekly_rules if r["title"])
        legacy_daily = "\n".join(f"{r['title']}:{r['time']}" for r in st.session_state.daily_rules if r["title"])
        settings["weekly_rules"] = legacy_weekly
        settings["daily_rules"] = legacy_daily
        save_settings(settings)
        st.success("✅ Rules saved!")

    # ── Background Image ─────────────────────────────────
    bg_file = st.file_uploader("Background image (optional, for A3 monthly PDF)",
                               type=["png", "jpg", "jpeg"], key="bg_upload")


# ═══════════════════════════════════════════════════════════
# TAB 2: MONTHLY CALENDAR
# ═══════════════════════════════════════════════════════════
with tab_monthly:

    # Convert structured rules for the calendar builder
    parsed_rules = []
    for r in st.session_state.weekly_rules:
        if not r["title"]:
            continue
        day_str = r["day"][:3].lower()
        try:
            wd = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].index(day_str)
        except ValueError:
            continue
        parsed_rules.append({"weekday": wd, "time": r["time"], "title": r["title"]})

    parsed_daily = [{"time": r["time"], "title": r["title"]} for r in st.session_state.daily_rules if r["title"]]

    col_gen, col_clone, col_reset = st.columns([2, 2, 1])

    with col_gen:
        if st.button("🔄 Generate / refresh calendar", type="primary", key="gen_preview"):
            with st.spinner("Building calendar..."):
                daymap = seat_activity_into_calendar(
                    year, month, activities_df, rota_df, parsed_rules,
                    include_holidays, parsed_daily)
                st.session_state[session_key] = {}
                for d, events in daymap.items():
                    lines = []
                    for ev in events:
                        if ev["notes"] == "Holiday":
                            lines.append(ev["title"].upper())
                        elif ev["notes"] == "staff shift":
                            lines.append(f"Staff: {ev['title']}")
                        elif ev["notes"] in ("fixed", "fixed daily", "activity"):
                            t = ev.get("time", "")
                            lines.append(f"{t} {ev['title']}".strip())
                    st.session_state[session_key][d] = "\n".join(lines)
                save_monthly_data(year, month, st.session_state[session_key])
            st.rerun()

    with col_clone:
        # Clone from previous month
        prev_month = month - 1 if month > 1 else 12
        prev_year = year if month > 1 else year - 1
        prev_key = f"{prev_year}-{prev_month:02d}"
        prev_data = load_monthly_data(prev_year, prev_month)
        if prev_data:
            if st.button(f"📋 Clone from {calendar.month_abbr[prev_month]} {prev_year}", key="clone_prev"):
                st.info("Cloning uses last month's rules to regenerate for the current month. Click 'Generate' after cloning.")
                # Load previous month's CSVs if current ones are missing
                if saved_rota is None:
                    prev_rota = load_saved_csv("rota", prev_year, prev_month)
                    if prev_rota is not None:
                        prev_rota.to_csv(os.path.join(UPLOADS_DIR, f"rota_{year}_{month:02d}.csv"), index=False)
                if saved_activities is None:
                    prev_act = load_saved_csv("activities", prev_year, prev_month)
                    if prev_act is not None:
                        prev_act.to_csv(os.path.join(UPLOADS_DIR, f"activities_{year}_{month:02d}.csv"), index=False)
                st.rerun()

    with col_reset:
        if st.button("🗑️ Reset", key="reset_month"):
            st.session_state.pop(session_key, None)
            filename = f"calendar_data_{year}_{month:02d}.json"
            if os.path.exists(filename):
                os.remove(filename)
            st.rerun()

    # ── Editable Grid ────────────────────────────────────
    if session_key in st.session_state:
        st.subheader(f"Edit {calendar.month_name[month]} {year}")
        month_days = calendar.monthcalendar(year, month)

        # Weekday headers
        hdr_cols = st.columns(7)
        for i, wd in enumerate(WEEKDAY_FULL):
            hdr_cols[i].markdown(f"**{wd}**")

        for week in month_days:
            cols = st.columns(7)
            for c_idx, day in enumerate(week):
                if day == 0:
                    cols[c_idx].markdown("")
                    continue
                d = dt.date(year, month, day)
                with cols[c_idx]:
                    current_text = st.session_state[session_key].get(d, "")
                    new_text = st.text_area(
                        f"{day}", current_text,
                        key=f"{session_key}_{d}", height=160,
                        label_visibility="visible")
                    if new_text != current_text:
                        st.session_state[session_key][d] = new_text
                        save_monthly_data(year, month, st.session_state[session_key])

        st.markdown("---")

        # ── A3 PDF Export ────────────────────────────────
        if st.button("📄 Generate monthly PDF (A3 landscape)", type="primary", key="gen_a3"):
            bg_bytes = bg_file.read() if bg_file else None
            edited = {}
            for k, v in st.session_state.items():
                if k.startswith(session_key + "_"):
                    try:
                        date_str = k.split("_", 1)[1]  # e.g. "2026-03-01"
                        edited[dt.date.fromisoformat(date_str)] = v
                    except Exception:
                        continue
            if not edited:
                edited = st.session_state.get(session_key, {})

            pdf_buf = draw_calendar_pdf(title, disclaimer, year, month, edited, background_bytes=bg_bytes)
            st.success("✅ A3 PDF generated!")
            st.download_button("📥 Download monthly calendar (A3)", data=pdf_buf,
                               file_name=f"calendar_{year}_{month:02d}_A3.pdf",
                               mime="application/pdf")

        # ── Save state ───────────────────────────────────
        col_save, col_load = st.columns(2)
        with col_save:
            if st.button("💾 Save calendar state", key="save_state"):
                state_data = {
                    "calendar_data": {d.isoformat(): v for d, v in st.session_state[session_key].items()},
                    "selected_holidays": st.session_state.get("selected_holidays", []),
                    "last_updated": dt.datetime.now().isoformat(),
                }
                if save_calendar_state(year, month, state_data):
                    st.success("✅ Calendar state saved!")
        with col_load:
            if st.button("📂 Load saved calendar", key="load_state"):
                loaded_state = load_calendar_state(year, month)
                if loaded_state and loaded_state.get("calendar_data"):
                    st.session_state[session_key] = {
                        dt.date.fromisoformat(k): v
                        for k, v in loaded_state["calendar_data"].items()
                    }
                    st.success("✅ Calendar loaded!")
                    st.rerun()
                else:
                    st.info("No saved calendar found for this month.")
    else:
        st.info("👆 Click **Generate / refresh calendar** to build the calendar from your rules and data.")


# ═══════════════════════════════════════════════════════════
# TAB 3: WEEKLY EXPORTS
# ═══════════════════════════════════════════════════════════
with tab_weekly:

    if session_key not in st.session_state:
        st.info("Please generate the monthly calendar first (in the Monthly Calendar tab).")
    else:
        # ── Text Size Controls ───────────────────────────
        with st.expander("🔤 Adjust text sizes", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                text_sizes["day_heading"] = st.slider("Day heading", 20, 60, text_sizes["day_heading"], 2, key="ts_heading")
                text_sizes["disclaimer"] = st.slider("Disclaimer", 8, 18, text_sizes["disclaimer"], 1, key="ts_disc")
            with c2:
                text_sizes["staff"] = st.slider("Staff names", 10, 25, text_sizes["staff"], 1, key="ts_staff")
                text_sizes["activities"] = st.slider("Activities", 14, 36, text_sizes["activities"], 2, key="ts_act")
            with c3:
                text_sizes["holidays"] = st.slider("Holidays", 10, 28, text_sizes["holidays"], 1, key="ts_hol")

            if st.button("💾 Save text size preferences", key="save_ts"):
                settings["text_sizes"] = text_sizes
                save_settings(settings)
                st.success("✅ Saved!")

        st.markdown("---")

        # ── Week Selector ────────────────────────────────
        weeks = get_weeks_in_month(year, month)
        if not weeks:
            st.info("No weeks found.")
        else:
            week_labels = [
                f"Week {i + 1}: {w[0].strftime('%b %d')} – {w[1].strftime('%b %d')}"
                for i, w in enumerate(weeks)]
            selected_week_idx = st.selectbox("Select week", range(len(weeks)),
                                             format_func=lambda i: week_labels[i],
                                             key="week_sel")

            start_date, end_date = weeks[selected_week_idx]
            week_days = [start_date + dt.timedelta(days=i) for i in range((end_date - start_date).days + 1)]

            # ── Image Management ─────────────────────────
            st.subheader("Images")

            # Show image library status
            library = load_image_library()
            if library:
                st.success(f"📚 Image library has {len(library)} saved activity images. These will be auto-assigned.")
            else:
                st.info("📚 No saved images yet. Select images below and they'll be remembered for future months.")

            # Day navigation
            if "preview_day_idx" not in st.session_state:
                st.session_state.preview_day_idx = 0

            # Clamp index
            st.session_state.preview_day_idx = min(st.session_state.preview_day_idx, len(week_days) - 1)

            col_prev, col_day_label, col_next = st.columns([1, 4, 1])
            with col_prev:
                if st.button("← Previous", key="prev_day", disabled=st.session_state.preview_day_idx == 0):
                    st.session_state.preview_day_idx -= 1
                    st.rerun()
            with col_next:
                if st.button("Next →", key="next_day", disabled=st.session_state.preview_day_idx >= len(week_days) - 1):
                    st.session_state.preview_day_idx += 1
                    st.rerun()

            current_day = week_days[st.session_state.preview_day_idx]
            day_key = current_day.isoformat()

            with col_day_label:
                day_name = calendar.day_name[current_day.weekday()]
                st.markdown(f"### {day_name} {current_day.day} {calendar.month_name[current_day.month]}")

            text = st.session_state.get(f"{session_key}_{current_day}", "").strip()
            if not text and session_key in st.session_state:
                cal_data = st.session_state[session_key]
                if isinstance(cal_data, dict):
                    text = cal_data.get(current_day, "").strip()
            if not text:
                text = "(No activities planned)"

            activities = extract_activities_from_text(text)
            unique_activities = _dedupe_activities(activities)

            # Image selection per activity
            if unique_activities:
                for act_idx, activity in enumerate(unique_activities):
                    act_lower = activity.lower().strip()
                    lib_img = get_library_image(act_lower)

                    with st.expander(f"{'✅' if lib_img else '📷'} {activity}", expanded=not lib_img):
                        if lib_img:
                            st.image(lib_img, width=200, caption="Saved in library")
                            if st.button(f"Change image for '{activity}'", key=f"change_{day_key}_{act_idx}"):
                                # Fetch new options
                                kw = get_activity_keyword(activity)
                                with st.spinner(f"Finding images for '{activity}'..."):
                                    options = fetch_pexels_images(kw, count=5)
                                st.session_state.image_options[f"{day_key}_{act_idx}"] = options
                                st.rerun()
                        else:
                            # Auto-fetch if no options loaded
                            opt_key = f"{day_key}_{act_idx}"
                            if opt_key not in st.session_state.image_options:
                                kw = get_activity_keyword(activity)
                                with st.spinner(f"Finding images for '{activity}'..."):
                                    options = fetch_pexels_images(kw, count=5)
                                st.session_state.image_options[opt_key] = options

                            options = st.session_state.image_options.get(opt_key, [])
                            if options:
                                img_cols = st.columns(min(5, len(options)))
                                for img_idx, img_bytes in enumerate(options):
                                    with img_cols[img_idx]:
                                        st.image(img_bytes, caption=f"Option {img_idx + 1}", use_container_width=True)
                                        if st.button("Select", key=f"sel_{opt_key}_{img_idx}"):
                                            save_library_image(activity, img_bytes, get_activity_keyword(activity))
                                            st.success(f"✅ Saved for '{activity}'!")
                                            st.rerun()

                                if st.button("🔄 More options", key=f"refresh_{opt_key}"):
                                    page = st.session_state.get(f"page_{opt_key}", 1) + 1
                                    st.session_state[f"page_{opt_key}"] = page
                                    kw = get_activity_keyword(activity)
                                    with st.spinner("Loading more..."):
                                        new_opts = fetch_pexels_images(kw, count=5, page=page)
                                    if new_opts:
                                        st.session_state.image_options[opt_key] = new_opts
                                    else:
                                        st.session_state[f"page_{opt_key}"] = 1
                                        new_opts = fetch_pexels_images(kw, count=5, page=1)
                                        st.session_state.image_options[opt_key] = new_opts
                                    st.rerun()
                            else:
                                st.warning(f"No images found for '{activity}'. Check your Pexels API key.")

            st.markdown("---")

            # ── Preview ──────────────────────────────────
            st.subheader("Preview")

            # Collect images for preview
            preview_images = []
            for act in unique_activities:
                lib_img = get_library_image(act)
                if lib_img:
                    preview_images.append(lib_img)
                if len(preview_images) >= 3:
                    break

            page_width, page_height = landscape(A4)
            layouts = get_default_image_layout(len(preview_images), page_width, page_height) if preview_images else None

            if day_key in st.session_state.image_layouts:
                layouts = st.session_state.image_layouts[day_key]

            # Layout controls
            if preview_images and layouts:
                with st.expander("📐 Adjust image positions", expanded=False):
                    default_layouts = [
                        {"x": 560, "y": 400, "width": 240, "height": 150},
                        {"x": 560, "y": 220, "width": 240, "height": 150},
                        {"x": 560, "y": 50, "width": 240, "height": 150},
                    ]
                    for idx in range(len(preview_images)):
                        if idx >= len(layouts):
                            layouts.append(default_layouts[min(idx, 2)].copy())
                        c1, c2 = st.columns(2)
                        with c1:
                            layouts[idx]["x"] = st.slider(f"Image {idx+1} X", 0, int(page_width), int(layouts[idx]["x"]), 5, key=f"lx_{day_key}_{idx}")
                            layouts[idx]["y"] = st.slider(f"Image {idx+1} Y", 0, int(page_height), int(layouts[idx]["y"]), 5, key=f"ly_{day_key}_{idx}")
                        with c2:
                            layouts[idx]["width"] = st.slider(f"Image {idx+1} width", 50, int(page_width * 0.5), int(layouts[idx]["width"]), 5, key=f"lw_{day_key}_{idx}")
                            layouts[idx]["height"] = st.slider(f"Image {idx+1} height", 50, int(page_height * 0.8), int(layouts[idx]["height"]), 5, key=f"lh_{day_key}_{idx}")
                    st.session_state.image_layouts[day_key] = layouts

                    if st.button("Reset to defaults", key=f"reset_layout_{day_key}"):
                        if day_key in st.session_state.image_layouts:
                            del st.session_state.image_layouts[day_key]
                        st.rerun()

            preview_img = create_preview_image(
                page_width, page_height, current_day, text,
                preview_images, layouts, text_sizes)
            st.image(preview_img, use_container_width=True,
                     caption=f"Day {st.session_state.preview_day_idx + 1} of {len(week_days)}")

            st.markdown("---")

            # ── PDF Generation ───────────────────────────
            col_gen_sel, col_gen_all = st.columns(2)

            with col_gen_sel:
                if st.button(f"📄 Generate {week_labels[selected_week_idx]}", type="primary", key="gen_week"):
                    with st.spinner("Generating PDF..."):
                        buf = generate_week_pdf(week_days, session_key, text_sizes)
                    st.success("✅ PDF generated!")
                    st.download_button("📥 Download this week", data=buf,
                                       file_name=f"week_{selected_week_idx + 1}_{year}_{month:02d}.pdf",
                                       mime="application/pdf", key="dl_week")

            with col_gen_all:
                if st.button("📄 Generate all weeks", key="gen_all_weeks"):
                    with st.spinner("Generating all weekly PDFs..."):
                        all_bufs = []
                        for wk_idx, (ws, we) in enumerate(weeks):
                            wk_days = [ws + dt.timedelta(days=i) for i in range((we - ws).days + 1)]
                            buf = generate_week_pdf(wk_days, session_key, text_sizes)
                            all_bufs.append(buf.getvalue())

                        merger = PyPDF2.PdfMerger()
                        for pdf in all_bufs:
                            merger.append(BytesIO(pdf))
                        merged = BytesIO()
                        merger.write(merged)
                        merger.close()
                        merged.seek(0)

                    st.success("✅ All weekly PDFs generated!")
                    st.download_button("📥 Download all weeks", data=merged,
                                       file_name=f"weekly_calendar_{year}_{month:02d}.pdf",
                                       mime="application/pdf", key="dl_all_weeks")


# ═══════════════════════════════════════════════════════════
# TAB 4: SETTINGS
# ═══════════════════════════════════════════════════════════
with tab_settings:

    st.subheader("Image library")
    library = load_image_library()
    if library:
        st.write(f"You have **{len(library)}** saved activity images.")
        st.caption("These images are automatically used when generating weekly PDFs. Select a different image in the Weekly Exports tab to update.")

        lib_cols = st.columns(4)
        for i, (name, entry) in enumerate(sorted(library.items())):
            with lib_cols[i % 4]:
                img_bytes = get_library_image(name)
                if img_bytes:
                    st.image(img_bytes, caption=name.title(), use_container_width=True)
                else:
                    st.write(f"**{name.title()}**")
                    st.caption("(image file missing)")

        if st.button("🗑️ Clear entire image library", key="clear_lib"):
            save_image_library({})
            # Remove files
            for f in os.listdir(IMAGE_LIBRARY_DIR):
                try:
                    os.remove(os.path.join(IMAGE_LIBRARY_DIR, f))
                except Exception:
                    pass
            st.success("Image library cleared.")
            st.rerun()
    else:
        st.info("No images saved yet. Images will be saved here automatically when you select them in the Weekly Exports tab.")

    st.markdown("---")

    st.subheader("Data management")

    col_del1, col_del2 = st.columns(2)
    with col_del1:
        if st.button("🗑️ Delete this month's data", key="del_month"):
            files_to_del = [
                f"calendar_data_{year}_{month:02d}.json",
                f"calendar_state_{year}_{month:02d}.json",
                os.path.join(UPLOADS_DIR, f"rota_{year}_{month:02d}.csv"),
                os.path.join(UPLOADS_DIR, f"activities_{year}_{month:02d}.csv"),
            ]
            deleted = 0
            for f in files_to_del:
                if os.path.exists(f):
                    os.remove(f)
                    deleted += 1
            st.session_state.pop(session_key, None)
            st.success(f"Deleted {deleted} file(s).")
            st.rerun()

    with col_del2:
        if st.button("🗑️ Delete all saved settings", key="del_settings"):
            if os.path.exists(SETTINGS_FILE):
                os.remove(SETTINGS_FILE)
            st.session_state["settings"] = {}
            st.success("Settings cleared.")
            st.rerun()

    st.markdown("---")

    st.subheader("About")
    st.caption("Care Home Activities Calendar v2.0")
    st.caption("Generates printable A3 monthly and A4 weekly activity calendars for care homes.")
    st.caption(f"Holidays data covers 2024–2026. Images powered by Pexels API.")
