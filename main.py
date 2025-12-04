"""
main.py

Care Home Monthly Calendar – Streamlit app (refactored with Pexels integration and image selection)

Features:
- Secure single-user login (uses Streamlit secrets["APP_PASSWORD"])
- CSV uploads for staff rota and activities
- Local holidays JSON read (holidays_2025_2026.json)
- Editable monthly preview
- A3 monthly PDF export (styled)
- Weekly splitting (4-5 weeks) + UI to choose a week
- Generate A4 weekly PDF for a selected week, or all weeks
- Pexels API integration with 5 image options per activity
- Interactive image selection and layout editor
- Persisted monthly edits and saved default rules
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
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, black, white
from reportlab.lib.units import mm
import requests
from PIL import Image, ImageDraw, ImageFont
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(page_title="Care Home Monthly Calendar", layout="wide")

# -------------------------
# Pexels API Configuration
# -------------------------
PEXELS_API_KEY = st.secrets.get("PEXELS_API_KEY", "")
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

# Cache directory for images
IMAGE_CACHE_DIR = "image_cache"
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# -------------------------
# Settings file handling (moved up so settings are available before UI)
# -------------------------
SETTINGS_FILE = "calendar_settings.json"


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


# -------------------------
# Monthly data persistence
# -------------------------
def save_monthly_data(year, month, data):
    """Save edited month data (dict[date -> text]) to disk."""
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


# -------------------------
# Activity Keyword Mapping
# -------------------------
# Keywords that should take priority when found in activity names
PRIORITY_KEYWORDS = [
    "quiz", "book", "bingo", "music", "painting", "baking", "gardening",
    "yoga", "exercise", "knitting", "singing", "walking", "tea", "coffee",
    "film", "movie", "dogs", "crafts", "dominoes", "cards", "scrabble", "jigsaw"
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
}

# -------------------------
# Helper functions
# -------------------------


def clean_text(s):
    """
    Normalise and "clean" text input for safe PDF rendering.
    """
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
    s = re.sub(r"[^\x20-\x7E]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


@st.cache_data
def load_all_holidays():
    """
    Load local holidays JSON (if present).
    """
    try:
        with open("holidays_2025_2026.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("holidays", [])
    except Exception as e:
        st.warning(f"Could not load holidays file: {e}")
        return []


ALL_HOLIDAYS = load_all_holidays()


def month_date_range(year: int, month: int):
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
        except:
            continue
        if d.year == year and d.month == month:
            name = clean_text(h.get("name", ""))
            normalized_name = name.lower()
            if (not selected_names) or (normalized_name in selected_normalized):
                holidays_list.append({
                    "date": d,
                    "title": name,
                    "notes": "Holiday"
                })
    return holidays_list


def seat_activity_into_calendar(year, month, activities_df, rota_df, rules,
                                include_holidays=True, daily_rules=None):
    """
    Build daymap: date -> list of event dicts (time, title, notes)
    Supports date-specific activities via an optional `date` column in activities CSV.
    """
    first, last = month_date_range(year, month)
    daymap = {first + dt.timedelta(days=i): [] for i in range((last - first).days + 1)}

    # Holidays
    if include_holidays:
        seen_holidays = set()
        combined_holidays = fetch_selected_holidays(year, month,
                                                    st.session_state.get("selected_holidays"))
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
            except:
                continue
            if d in daymap:
                staff = clean_text(str(r.get("staff", "")))
                staff = re.sub(r"\s*\d+$", "", staff)
                start = str(r.get("shift_start", "")).strip()
                end = str(r.get("shift_end", "")).strip()
                shift_time = f"({start} – {end})" if start and end else ""
                display = f"{staff} {shift_time}".strip()
                if display:
                    daymap[d].append({"time": None, "title": display, "notes": "staff shift"})

    # Fixed weekly rules
    fixed_rules = []
    for rule in rules:
        for d in daymap:
            if d.weekday() == rule["weekday"]:
                fixed_rules.append({"date": d, "time": rule.get("time"),
                                    "title": clean_text(rule["title"]),
                                    "notes": "fixed"})

    # Fixed daily rules
    if daily_rules:
        for d in daymap:
            for rule in daily_rules:
                daymap[d].append({
                    "date": d,
                    "time": rule.get("time"),
                    "title": clean_text(rule["title"]),
                    "notes": "fixed daily"
                })

    # Activities
    activities = []
    if activities_df is not None:
        for _, r in activities_df.iterrows():
            name = clean_text(r.get("name") or r.get("activity_name") or "")
            pref_days = str(r.get("preferred_days", "")).split(";")
            pref_days = [p.strip()[:3].lower() for p in pref_days if p.strip()]
            pref_time = str(r.get("preferred_time", "")).strip()
            freq = int(r.get("frequency", 0)) if str(r.get("frequency", "")).isdigit() else 0
            placed = 0

            # ---- NEW: support specific date column in activities CSV ----
            activity_date_raw = None
            # r can be a Series with index; check both 'date' key and column existence
            if isinstance(r, pd.Series):
                if "date" in r.index:
                    activity_date_raw = r.get("date")
                else:
                    activity_date_raw = None
            else:
                activity_date_raw = r.get("date") if "date" in r else None

            specific_date = None
            if pd.notna(activity_date_raw) and activity_date_raw:
                try:
                    specific_date = pd.to_datetime(activity_date_raw).date()
                except Exception:
                    specific_date = None

            if specific_date:
                # Place directly on that date if it is in the current month
                if specific_date in daymap:
                    activities.append({"date": specific_date, "time": pref_time, "title": name, "notes": "activity"})
                # Skip the standard preferred-day logic for date-specific entries
                continue
            # ---- END NEW BLOCK ----

            for d in sorted(daymap.keys()):
                if freq and placed >= freq:
                    break
                dow3 = calendar.day_name[d.weekday()][:3].lower()
                if dow3 in pref_days:
                    activities.append({"date": d, "time": pref_time, "title": name, "notes": "activity"})
                    placed += 1

    # Normalize times and dedupe
    time_pattern = re.compile(r"^(\d{1,2})(?::?(\d{2}))?$")

    def normalize_time(t):
        if not t or not isinstance(t, str):
            return None
        t2 = t.strip().lower().replace(".", ":").replace(" ", "")
        match = time_pattern.match(t2)
        if match:
            hour, minute = match.groups()
            hour = hour.zfill(2)
            minute = minute if minute else "00"
            return f"{hour}:{minute}"
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
        except:
            return dt.time(23, 59)

    for d in daymap:
        daymap[d].sort(key=lambda e: (
            0 if e["notes"] == "Holiday" else
            1 if e["notes"] == "staff shift" else
            2, sort_key(e)
        ))

    return daymap


def draw_calendar_pdf(title, disclaimer, year, month, cell_texts, background_bytes=None):
    """
    Create an A3 landscape PDF with the calendar grid and text rendered.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A3))
    width, height = landscape(A3)

    # Optional background
    if background_bytes:
        try:
            img = ImageReader(BytesIO(background_bytes))
            c.drawImage(img, 0, 0, width=width, height=height, preserveAspectRatio=False, mask="auto")
        except Exception as e:
            st.warning(f"Background load failed: {e}")

    title_text = clean_text(title)
    disclaimer_text = clean_text(disclaimer)

    # Title pill
    title_font = "Helvetica-Bold"
    title_size = 20
    c.setFont(title_font, title_size)
    title_width = c.stringWidth(title_text, title_font, title_size)

    side_padding = 15 * mm
    vertical_padding = 4 * mm
    pill_w = title_width + side_padding
    pill_h = 4 * mm + vertical_padding
    pill_y = height - 10 * mm
    pill_x = (width - pill_w) / 2

    c.setFillColor(Color(0, 0, 0))
    c.roundRect(pill_x, pill_y, pill_w, pill_h, pill_h / 2, fill=1, stroke=0)
    c.setFillColor(white)
    text_y = pill_y + (pill_h / 2) - (title_size / 3.2)
    c.drawCentredString(width / 2, text_y, title_text)

    # Disclaimer pill
    disclaimer_font = "Helvetica-Bold"
    disclaimer_size = 11
    c.setFont(disclaimer_font, disclaimer_size)
    disclaimer_width = c.stringWidth(disclaimer_text, disclaimer_font, disclaimer_size)

    disc_padding_x = 10 * mm
    disc_padding_y = 1 * mm
    disc_w = disclaimer_width + disc_padding_x
    disc_h = 6 * mm + disc_padding_y
    disc_x = (width - disc_w) / 2
    disc_y = pill_y - disc_h - 0.5 * mm

    c.setFillColor(Color(0, 0, 0))
    c.roundRect(disc_x, disc_y, disc_w, disc_h, disc_h / 2, fill=1, stroke=0)
    c.setFillColor(white)
    disc_text_y = disc_y + (disc_h / 2) - (disclaimer_size / 3)
    c.drawCentredString(width / 2, disc_text_y, disclaimer_text)

    # Grid variables
    left, right, top, bottom = 4 * mm, 4 * mm, 37 * mm, 5 * mm
    grid_w = width - left - right
    cols, rows = 7, 5
    col_w = grid_w / cols

    # Weekday header bar
    weekday_bg = Color(0, 0, 0)
    bar_height = 8 * mm
    bar_y = height - top + 11 * mm
    c.setFillColor(weekday_bg)
    c.rect(left, bar_y, grid_w, bar_height, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 15)
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, wd in enumerate(weekdays):
        x = left + i * col_w + col_w / 2
        c.drawCentredString(x, bar_y + 2.5 * mm, wd)

    bar_gap = 1.5 * mm
    top_of_grid = bar_y - bar_gap
    grid_h = top_of_grid - bottom
    row_h = grid_h / rows

    cream = Color(1, 1, 1, alpha=0.93)
    staff_blue = Color(0, 0.298, 0.6)
    month_days = calendar.monthcalendar(year, month)

    # Draw cells
    for r_idx, week in enumerate(month_days):
        for c_idx, day in enumerate(week):
            if day == 0:
                continue
            d = dt.date(year, month, day)
            x = left + c_idx * col_w
            y = bottom + (rows - 1 - r_idx) * row_h

            c.setFillColor(cream)
            c.setStrokeColor(black)
            c.roundRect(x, y, col_w, row_h, 5, fill=1, stroke=1)

            # Date number
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(black)
            day_str = str(day)
            day_width = c.stringWidth(day_str, "Helvetica-Bold", 12)
            c.drawString(x + col_w - day_width - 1.2 * mm, y + row_h - 4.5 * mm, day_str)

            # Cell content
            lines = cell_texts.get(d, "").split("\n")
            text_y = y + row_h - 3.5 * mm
            line_spacing = 4 * mm

            for line in lines:
                line = clean_text(line).strip()
                if not line:
                    continue

                # Holiday: uppercase
                if line.isupper():
                    max_text_width = col_w - (day_width + 6 * mm)
                    words = line.split()
                    current_line = ""
                    wrapped_holiday = []

                    for word in words:
                        test_line = (current_line + " " + word).strip()
                        line_width = c.stringWidth(test_line, "Helvetica-Bold", 8.7)
                        if line_width > max_text_width and current_line:
                            wrapped_holiday.append(current_line)
                            current_line = word
                        else:
                            current_line = test_line
                    if current_line:
                        wrapped_holiday.append(current_line)

                    for wh in wrapped_holiday:
                        wh = wh.strip()
                        if not wh:
                            continue
                        c.setFont("Helvetica-Bold", 8.7)
                        c.setFillColor(black)
                        c.drawString(x + 2 * mm, text_y, wh)
                        text_width = c.stringWidth(wh, "Helvetica-Bold", 8.7)
                        underline_y = text_y - 0.5 * mm
                        c.line(x + 2 * mm, underline_y, x + 2 * mm + text_width, underline_y)
                        text_y -= line_spacing
                    continue

                c.setFont("Helvetica-Bold", 10.5)
                max_text_width = col_w - (day_width + 0.5 * mm)

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
                    time_width = c.stringWidth(time_part + " ", "Helvetica-Bold", 10.5)
                    available_width = max_text_width - time_width

                    words = rest.split()
                    current_line = ""
                    wrapped_lines = []
                    for word in words:
                        test_line = (current_line + " " + word).strip()
                        if c.stringWidth(test_line, "Helvetica-Bold", 10.5) > available_width and current_line:
                            wrapped_lines.append(current_line)
                            current_line = word
                        else:
                            current_line = test_line
                    if current_line:
                        wrapped_lines.append(current_line)

                    first_line = True
                    for wline in wrapped_lines:
                        wline = wline.strip()
                        if not wline:
                            continue
                        if first_line:
                            c.drawString(x + 2 * mm + time_width, text_y, wline)
                            first_line = False
                        else:
                            text_y -= line_spacing
                            c.drawString(x + 2 * mm, text_y, wline)
                    text_y -= line_spacing
                    if text_y < y + 4 * mm:
                        break
                    continue

                # Normal wrapping
                words = line.split()
                current_line = ""
                wrapped_lines = []
                for word in words:
                    test_line = (current_line + " " + word).strip()
                    if c.stringWidth(test_line, "Helvetica-Bold", 10.5) > max_text_width and current_line:
                        wrapped_lines.append(current_line)
                        current_line = word
                    else:
                        current_line = test_line
                if current_line:
                    wrapped_lines.append(current_line)

                for subline in wrapped_lines:
                    subline = subline.strip()
                    if not subline:
                        continue
                    c.setFont("Helvetica-Bold", 10.5)
                    c.setFillColor(black)
                    c.drawString(x + 2 * mm, text_y, subline)
                    text_y -= line_spacing
                    if text_y < y + 4 * mm:
                        break

    c.save()
    buffer.seek(0)
    return buffer


# -------------------------
# Pexels Integration Functions
# -------------------------


def get_activity_keyword(activity_name):
    """
    Extract search keyword for an activity.
    Returns the mapped keyword or a cleaned version of the activity name.
    """
    activity_lower = activity_name.lower().strip()

    # Check for priority keywords first (e.g., "quiz", "book")
    for keyword in PRIORITY_KEYWORDS:
        if keyword in activity_lower:
            # Return more specific search terms for better results
            if keyword == "book":
                return "books reading"
            elif keyword == "quiz":
                return "quiz trivia"
            else:
                return keyword

    # Check for direct matches
    if activity_lower in ACTIVITY_KEYWORDS:
        return ACTIVITY_KEYWORDS[activity_lower]

    # Check for partial matches
    for key, value in ACTIVITY_KEYWORDS.items():
        if key in activity_lower:
            return value

    # Default: clean the activity name
    cleaned = activity_lower.replace("club", "").strip()
    return cleaned if cleaned else "seniors activity"


def fetch_pexels_images(keyword, orientation="landscape", size="medium", count=5, page=1):
    """
    Fetch multiple images from Pexels API based on keyword.
    Returns list of image bytes or empty list if fetch fails.
    """
    if not PEXELS_API_KEY:
        return []

    # Fetch from Pexels
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": keyword,
        "orientation": orientation,
        "per_page": count,
        "page": page
    }

    images = []
    try:
        response = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("photos") and len(data["photos"]) > 0:
            for photo in data["photos"]:
                # Choose appropriate size
                if size == "large":
                    image_url = photo["src"].get("large2x", photo["src"]["large"])
                elif size == "small":
                    image_url = photo["src"].get("small", photo["src"]["medium"])
                else:
                    image_url = photo["src"]["medium"]

                # Download image
                img_response = requests.get(image_url, timeout=10)
                img_response.raise_for_status()
                img_bytes = img_response.content

                images.append(img_bytes)

    except Exception:
        pass

    return images


def fetch_pexels_image(keyword, orientation="landscape", size="medium", page=1):
    """
    Fetch a single image from Pexels API based on keyword.
    Returns image bytes or None if fetch fails.
    """
    images = fetch_pexels_images(keyword, orientation, size, count=1, page=page)
    return images[0] if images else None


def fetch_images_parallel(activities, page_numbers):
    """
    Fetch images for multiple activities in parallel.
    Returns dict mapping activity index to list of image bytes.
    """
    results = {}

    def fetch_for_activity(idx, activity, page_num):
        keyword = get_activity_keyword(activity)
        images = fetch_pexels_images(keyword, orientation="landscape", size="medium", count=5, page=page_num)
        return idx, images, keyword

    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, (activity, page_num) in enumerate(zip(activities, page_numbers)):
            future = executor.submit(fetch_for_activity, idx, activity, page_num)
            futures.append(future)

        for future in as_completed(futures):
            try:
                idx, images, keyword = future.result()
                results[idx] = {"images": images, "keyword": keyword}
            except Exception:
                results[idx] = {"images": [], "keyword": ""}

    return results


def extract_activities_from_text(text):
    """
    Extract individual activities from formatted text.
    Returns list of activity names (without times).
    """
    activities = []

    for line in text.split("\n"):
        line = clean_text(line).strip()
        if not line:
            continue

        # Skip holidays (uppercase)
        if line.isupper():
            continue

        # Skip staff lines
        if line.lower().startswith("staff:"):
            continue

        # Remove time prefix (e.g., "14:30: ")
        line = re.sub(r"^\d{1,2}:\d{2}:?\s*", "", line)

        # Split by arrow separator
        parts = re.split(r"\s*→\s*", line)

        for part in parts:
            part = part.strip()
            if part and not part.lower().startswith("staff"):
                activities.append(part)

    return activities


def get_default_image_layout(num_images, page_width, page_height):
    """
    Get default positions and sizes for images - vertically stacked on the right.
    Returns list of dicts with keys: x, y, width, height (in points)
    """
    layouts = []

    # Fixed positions and sizes matching your specification
    # Image 1: X=560, Y=400, Width=240, Height=150
    # Image 2: X=560, Y=220, Width=240, Height=150
    # Image 3: X=560, Y=50, Width=240, Height=150

    default_layouts = [
        {"x": 560, "y": 400, "width": 240, "height": 150},
        {"x": 560, "y": 220, "width": 240, "height": 150},
        {"x": 560, "y": 50, "width": 240, "height": 150}
    ]

    # Return only the number of layouts needed
    for idx in range(min(num_images, len(default_layouts))):
        layouts.append(default_layouts[idx].copy())  # Use .copy() to avoid reference issues

    return layouts


def draw_weekly_page_with_custom_layout(c, width, height, day_obj, text, image_bytes_list=None, image_layouts=None):
    """
    Draw a single day page on A4 landscape with custom positioned images.
    image_bytes_list: list of up to 3 image bytes
    image_layouts: list of dicts with x, y, width, height for each image
    """
    # Define layout areas
    text_area_right = width * 0.62  # Text takes left 62%

    # Draw day heading (LEFT ALIGNED)
    c.setFont("Helvetica-Bold", 40)
    day_str = f"{calendar.day_name[day_obj.weekday()]} {day_obj.day} {calendar.month_name[day_obj.month]}"
    c.drawString(10 * mm, height - 20 * mm, day_str)

    # Draw disclaimer (LEFT ALIGNED)
    c.setFont("Helvetica-Oblique", 12)
    disclaimer_text = (
        "Activities may change due to unforeseen circumstances. "
        "Families are welcome to join. "
        "Weather permitting, activities may move outdoors."
    )

    max_text_width = text_area_right - 20 * mm
    words = disclaimer_text.split()
    current_line = ""
    wrapped_lines = []
    for word in words:
        test_line = (current_line + " " + word).strip()
        if c.stringWidth(test_line, "Helvetica-Oblique",
                         12) > max_text_width and current_line:
            wrapped_lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        wrapped_lines.append(current_line)

    line_spacing = 6 * mm
    text_y = height - 30 * mm
    for line in wrapped_lines:
        c.drawString(10 * mm, text_y, line)  # LEFT ALIGNED
        text_y -= line_spacing

    y = text_y - 8 * mm

    # Draw images with custom layout
    if image_bytes_list and image_layouts:
        try:
            for idx, (image_bytes, layout) in enumerate(
                    zip(image_bytes_list, image_layouts)):
                img = ImageReader(BytesIO(image_bytes))

                # Draw rounded rectangle background
                c.setFillColor(Color(0.95, 0.95, 0.95))
                c.roundRect(layout["x"] - 3 * mm, layout["y"] - 3 * mm,
                            layout["width"] + 6 * mm,
                            layout["height"] + 6 * mm,
                            8, fill=1, stroke=0)

                # Draw image
                c.drawImage(img, layout["x"], layout["y"],
                            width=layout["width"], height=layout["height"],
                            preserveAspectRatio=True, mask="auto")

        except Exception as e:
            # Silently fail - images are optional
            pass

    # Draw activities text (left side only)
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
        words = combined_staff.split()
        current_line = ""
        wrapped_staff = []
        max_width = text_area_right - 20 * mm
        for word in words:
            test_line = (current_line + " " + word).strip()
            if c.stringWidth(test_line, "Helvetica-Oblique",
                             15) > max_width and current_line:
                wrapped_staff.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            wrapped_staff.append(current_line)

        c.setFont("Helvetica-Oblique", 15)
        c.setFillColor(staff_blue)
        for wrapped in wrapped_staff:
            c.drawString(10 * mm, y, wrapped)
            y -= 9 * mm
        y -= 5 * mm

    # Draw activities
    merged_activities = {}
    for line in other_lines:
        match = re.match(r"^(\d{1,2}:\d{2})\s*(.*)", line)
        if match:
            time, desc = match.groups()
            merged_activities.setdefault(time, []).append(desc.strip())
        else:
            merged_activities.setdefault(None, []).append(line.strip())

    for time, desc_list in merged_activities.items():
        if all(d.isupper() for d in desc_list):
            combined_text = (" / ".join(
                desc_list) if time is None else f"{time}: " + " / ".join(
                desc_list))
            font_size = 15
            c.setFont("Helvetica-Bold", font_size)
            c.setFillColor(black)
        else:
            combined_text = (" → ".join(
                desc_list) if time is None else f"{time}: " + " → ".join(
                desc_list))
            font_size = 22
            c.setFont("Helvetica-Bold", font_size)
            c.setFillColor(Color(0.1, 0.1, 0.1))

        x_start = 10 * mm
        max_width_text = text_area_right - 20 * mm

        words = combined_text.split()
        current_line = ""
        wrapped_lines = []
        for word in words:
            test_line = (current_line + " " + word).strip()
            if c.stringWidth(test_line, "Helvetica-Bold",
                             font_size) > max_width_text and current_line:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            wrapped_lines.append(current_line)

        for wrapped in wrapped_lines:
            c.drawString(x_start, y, wrapped.strip())
            y -= 8 * mm if not all(d.isupper() for d in desc_list) else 7 * mm

        y -= 6 * mm
        if y < 25 * mm:
            break


def create_preview_image_with_layout(width, height, day_obj, text,
                                     image_bytes_list=None,
                                     image_layouts=None):
    """
    Create a preview image using PIL that matches the PDF layout exactly.
    Returns PIL Image object.
    """
    # Create white background
    img = Image.new('RGB', (int(width), int(height)), color='white')
    draw = ImageDraw.Draw(img)

    # Define layout areas (matching PDF)
    text_area_right = int(width * 0.62)

    # Try to load fonts (matching PDF sizes)
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        disclaimer_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 12)
        staff_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 15)
        activity_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        holiday_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except:
        # Fallback to default fonts with approximate sizes
        title_font = ImageFont.load_default()
        disclaimer_font = ImageFont.load_default()
        staff_font = ImageFont.load_default()
        activity_font = ImageFont.load_default()
        holiday_font = ImageFont.load_default()

    # Draw day heading (LEFT ALIGNED - matching PDF position)
    day_str = f"{calendar.day_name[day_obj.weekday()]} {day_obj.day} {calendar.month_name[day_obj.month]}"
    day_x = int(10 * 2.83465)  # 10mm from left
    day_y = int(20 * 2.83465)  # 20mm from top
    draw.text((day_x, day_y), day_str, fill='black', font=title_font)

    # Draw disclaimer (LEFT ALIGNED - matching PDF position and wrapping)
    disclaimer_text = (
        "Activities may change due to unforeseen circumstances. "
        "Families are welcome to join. "
        "Weather permitting, activities may move outdoors."
    )

    # Wrap disclaimer text (matching PDF logic)
    max_disclaimer_width = text_area_right - int(20 * 2.83465)
    words = disclaimer_text.split()
    current_line = ""
    wrapped_lines = []

    for word in words:
        test_line = (current_line + " " + word).strip()
        try:
            bbox = draw.textbbox((0, 0), test_line, font=disclaimer_font)
            line_width = bbox[2] - bbox[0]
        except:
            line_width = len(test_line) * 7

        if line_width > max_disclaimer_width and current_line:
            wrapped_lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        wrapped_lines.append(current_line)

    # Draw wrapped disclaimer (LEFT ALIGNED)
    line_spacing = int(6 * 2.83465)
    text_y = int(30 * 2.83465)  # 30mm from top
    line_x = int(10 * 2.83465)  # 10mm from left

    for line in wrapped_lines:
        draw.text((line_x, text_y), line, fill='gray',
                  font=disclaimer_font)
        text_y += line_spacing

    y_pos = text_y + int(8 * 2.83465)  # 8mm gap

    # Draw images with custom layout (matching PDF)
    if image_bytes_list and image_layouts:
        for idx, (image_bytes, layout) in enumerate(
                zip(image_bytes_list, image_layouts)):
            try:
                pil_img = Image.open(BytesIO(image_bytes))

                # Resize to fit layout
                pil_img = pil_img.resize(
                    (int(layout["width"]), int(layout["height"])),
                    Image.Resampling.LANCZOS)

                # Draw gray background (matching PDF rounded rectangle)
                bg_x = int(layout["x"] - 3 * 2.83465)
                bg_y = int(height - layout["y"] - layout[
                    "height"] - 3 * 2.83465)  # Flip Y
                bg_w = int(layout["width"] + 6 * 2.83465)
                bg_h = int(layout["height"] + 6 * 2.83465)

                draw.rounded_rectangle(
                    [bg_x, bg_y, bg_x + bg_w, bg_y + bg_h],
                    radius=8,
                    fill=(242, 242, 242)  # Light gray matching PDF
                )

                # Paste image
                img_x = int(layout["x"])
                img_y = int(
                    height - layout["y"] - layout["height"])  # Flip Y for PIL
                img.paste(pil_img, (img_x, img_y))

            except Exception as e:
                pass

    # Parse and draw text content (matching PDF logic exactly)
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

    # Draw staff lines (matching PDF)
    staff_blue = (0, 76, 153)  # RGB for Color(0, 0.298, 0.6)

    if staff_lines:
        combined_staff = " - ".join(staff_lines)
        words = combined_staff.split()
        current_line = ""
        wrapped_staff = []
        max_width = text_area_right - int(20 * 2.83465)

        for word in words:
            test_line = (current_line + " " + word).strip()
            try:
                bbox = draw.textbbox((0, 0), test_line, font=staff_font)
                line_width = bbox[2] - bbox[0]
            except:
                line_width = len(test_line) * 9

            if line_width > max_width and current_line:
                wrapped_staff.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            wrapped_staff.append(current_line)

        for wrapped in wrapped_staff:
            draw.text((int(10 * 2.83465), y_pos), wrapped, fill=staff_blue,
                      font=staff_font)
            y_pos += int(9 * 2.83465)
        y_pos += int(5 * 2.83465)

    # Draw activities (matching PDF merging and formatting logic)
    merged_activities = {}
    for line in other_lines:
        match = re.match(r"^(\d{1,2}:\d{2})\s*(.*)", line)
        if match:
            time, desc = match.groups()
            merged_activities.setdefault(time, []).append(desc.strip())
        else:
            merged_activities.setdefault(None, []).append(line.strip())

    for time, desc_list in merged_activities.items():
        # Check if all uppercase (holiday)
        if all(d.isupper() for d in desc_list):
            combined_text = (" / ".join(
                desc_list) if time is None else f"{time}: " + " / ".join(
                desc_list))
            current_font = holiday_font
            font_color = (0, 0, 0)  # Black
            line_spacing_val = int(7 * 2.83465)
        else:
            combined_text = (" → ".join(
                desc_list) if time is None else f"{time}: " + " → ".join(
                desc_list))
            current_font = activity_font
            font_color = (26, 26,
                          26)  # Dark gray matching Color(0.1, 0.1, 0.1)
            line_spacing_val = int(8 * 2.83465)

        x_start = int(10 * 2.83465)
        max_width_text = text_area_right - int(20 * 2.83465)

        # Word wrap
        words = combined_text.split()
        current_line = ""
        wrapped_lines = []

        for word in words:
            test_line = (current_line + " " + word).strip()
            try:
                bbox = draw.textbbox((0, 0), test_line, font=current_font)
                line_width = bbox[2] - bbox[0]
            except:
                line_width = len(test_line) * 13

            if line_width > max_width_text and current_line:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            wrapped_lines.append(current_line)

        for wrapped in wrapped_lines:
            draw.text((x_start, y_pos), wrapped.strip(), fill=font_color,
                      font=current_font)
            y_pos += line_spacing_val

        y_pos += int(6 * 2.83465)  # Gap between activities

        # Stop if we run out of space
        if y_pos > height - int(25 * 2.83465):
            break

    return img


def ordinal(n):
    """Return ordinal suffix for a number (1st, 2nd, 3rd, etc.)"""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# -----------------------------------------------
# Helper: split month into week ranges (start_date, end_date)
# -----------------------------------------------
def get_weeks_in_month(year, month):
    """
    Returns a list of tuples [(start_date, end_date), ...] for each week row
    in the calendar.monthcalendar output.
    """
    cal = calendar.monthcalendar(year, month)
    weeks = []
    for week in cal:
        valid_days = [day for day in week if day != 0]
        if not valid_days:
            continue
        start_day, end_day = valid_days[0], valid_days[-1]
        start_date = dt.date(year, month, start_day)
        end_date = dt.date(year, month, end_day)
        weeks.append((start_date, end_date))
    return weeks


# -------------------------
# Secure single-user login
# -------------------------
REAL_PASSWORD = st.secrets["APP_PASSWORD"]
PASSWORD_HASH = hashlib.sha256(REAL_PASSWORD.encode()).hexdigest()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔒 Secure Access")
    password = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH:
            st.session_state.logged_in = True
            st.success("Access granted ✅")
            st.rerun()
        else:
            st.error("Incorrect password. Try again.")
    st.stop()

# -------------------------
# Streamlit UI - Main
# -------------------------

# Load and persist settings into session state early so UI picks them up every run
if "settings" not in st.session_state:
    st.session_state["settings"] = load_settings()

# Important Notice Alert
st.warning("""
### ⚠️ Important Notice
**Fiverr Account Suspension:** My Fiverr account has been permanently suspended due to a "suspicious clone account" issue. 
This is beyond my control, but I remain committed to supporting you.

**For continued support and updates, please contact me directly at:**  
📧 **hamza.ahmed.ws@gmail.com**

Thank you for your understanding and patience.
""")

st.title("🏡 Care Home Monthly Activities – Editable Preview & A3 PDF")

col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Year", 2024, 2035, dt.date.today().year)
    month = st.selectbox("Month", range(1, 13),
                         index=dt.date.today().month - 1,
                         format_func=lambda x: calendar.month_name[x])
with col2:
    title = st.text_input("Calendar Title",
                          f"{calendar.month_name[month]} {year}")
    disclaimer = st.text_input("Disclaimer",
                               "Activities subject to change. Please confirm with staff.")

st.markdown("### 📋 CSV Upload Instructions")
with st.expander("🧑‍💼 Staff Rota CSV Format (Example)"):
    st.write("""
            **Required Headers:**
            - `date` → Date in format `YYYY-MM-DD`
            - `staff` → Staff member's full name  
            - `shift_start` → Start time (e.g. `09:00`)
            - `shift_end` → End time (e.g. `16:30`)
            - `role` → (Optional) Staff role or position
            """)

with st.expander("🎯 Activities CSV Format (Example)"):
    st.write("""
            **Required Headers:**
            - `name` → Activity name  
            - `preferred_days` → Day(s) of week, separated by `;` (e.g. `Mon; Wed; Fri`)  
            - `preferred_time` → Start time (e.g. `14:30`)  
            - `frequency` → Number of times per month  
            - `staff_required` → Number of staff required for the activity  
            - `notes` → (Optional) Any notes or description  
            - `date` → (Optional) YYYY-MM-DD for an activity that occurs on a specific date
            """)

rota_df = parse_csv(st.file_uploader("📂 Upload Staff Rota CSV", type=["csv"]))
activities_df = parse_csv(
    st.file_uploader("📂 Upload Activities CSV", type=["csv"]))
bg_file = st.file_uploader("Background Image (optional)",
                           type=["png", "jpg", "jpeg"])

saved_weekly = st.session_state["settings"].get("weekly_rules",
                                                "Film Night:Thu:18:00\nDogs for Health:Thu:11:00\nReminiscence:Sat:18:00")
saved_daily = st.session_state["settings"].get("daily_rules",
                                               "Morning Exercise:09:00\nNews Headlines:10:00")

fixed_rules_text = st.text_area(
    "Fixed Weekly Rules (e.g. Film Night:Thu:18:00)",
    value=saved_weekly, key="weekly_rules_input")

daily_rules_text = st.text_area(
    "Fixed Daily Rules (e.g. Morning Exercise:09:00)",
    value=saved_daily, key="daily_rules_input")

if st.button("💾 Save Default Rules"):
    st.session_state["settings"]["weekly_rules"] = st.session_state[
        "weekly_rules_input"]
    st.session_state["settings"]["daily_rules"] = st.session_state[
        "daily_rules_input"]
    save_settings(st.session_state["settings"])
    st.success("✅ Default rules saved successfully!")

# Parse fixed weekly rules
rules = []
for line in fixed_rules_text.splitlines():
    parts = [p.strip() for p in line.split(":")]
    if len(parts) >= 2:
        day = parts[1][:3].lower()
        time = parts[2] if len(parts) > 2 else ""
        title_txt = parts[0]
        try:
            weekday = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].index(
                day)
        except ValueError:
            continue
        rules.append({"weekday": weekday, "time": time, "title": title_txt})

daily_rules = []
for line in daily_rules_text.splitlines():
    line = line.strip()
    if not line:
        continue
    parts = [p.strip() for p in line.split(":", 1)]
    if len(parts) == 2:
        title_txt, time = parts
    else:
        title_txt, time = parts[0], ""
    if title_txt:
        daily_rules.append({"time": time, "title": title_txt})

include_holidays = st.checkbox("Include UK National Holidays", True)

# Reset selected holidays when month/year changes
if "last_month" not in st.session_state or st.session_state.get(
        "last_month") != month or st.session_state.get("last_year") != year:
    st.session_state["selected_holidays"] = []
    st.session_state["last_month"] = month
    st.session_state["last_year"] = year

# Holiday selection UI
if include_holidays:
    st.markdown("### 🗓️ Select Holidays to Include")
    holidays_by_day = {}
    for h in ALL_HOLIDAYS:
        try:
            d = dt.datetime.strptime(h["date"], "%Y-%m-%d").date()
        except:
            continue
        if d.year == year and d.month == month:
            holidays_by_day.setdefault(d, []).append(h["name"])

    if not holidays_by_day:
        st.info("No holidays found for this month.")
    else:
        saved_selection = set(st.session_state.get("selected_holidays", []))
        current_selection = set()
        if not saved_selection:
            all_holiday_names = {hname for hlist in holidays_by_day.values()
                                 for hname in hlist}
            saved_selection = all_holiday_names

        st.markdown("""
                <style>
                .day-block {
                    border: 1px solid #aaa;
                    border-radius: 6px;
                    padding: 6px 8px;
                    margin: 4px 0;
                }
                .day-header {
                    font-weight: bold;
                    margin-bottom: 4px;
                }
                </style>
                """, unsafe_allow_html=True)

        month_days = calendar.monthcalendar(year, month)
        for week in month_days:
            cols = st.columns(7)
            for c_idx, day in enumerate(week):
                if day == 0:
                    continue
                date_obj = dt.date(year, month, day)
                day_holidays = holidays_by_day.get(date_obj, [])
                with cols[c_idx]:
                    st.markdown(
                        f"<div class='day-block'><div class='day-header'>{calendar.month_abbr[month]} {day}</div>",
                        unsafe_allow_html=True)
                    if not day_holidays:
                        st.markdown("<em>No holidays</em>",
                                    unsafe_allow_html=True)
                    else:
                        for name in sorted(set(day_holidays)):
                            key = f"hol_{year}-{month:02d}-{day:02d}_{name}"
                            checked = name in saved_selection
                            if st.checkbox(name, value=checked, key=key):
                                current_selection.add(name)
                    st.markdown("</div>", unsafe_allow_html=True)

        st.session_state["selected_holidays"] = list(current_selection)

# -------------------------
# Editable Preview
# -------------------------
st.markdown("---")
st.write("## Preview & Edit Monthly Calendar")

session_key = f"{year}-{month:02d}"

# Load persisted monthly edits automatically into session state if not present
if session_key not in st.session_state:
    loaded = load_monthly_data(year, month)
    if loaded:
        st.session_state[session_key] = loaded

# Reset preview when changing month/year
if "last_preview_year" not in st.session_state or st.session_state.get(
        "last_preview_year") != year or st.session_state.get(
        "last_preview_month") != month:
    st.session_state["last_preview_year"] = year
    st.session_state["last_preview_month"] = month

if st.button("Preview Calendar"):
    with st.spinner("Generating preview..."):
        daymap = seat_activity_into_calendar(year, month, activities_df,
                                             rota_df, rules, include_holidays,
                                             daily_rules)
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

        # Persist monthly edits to disk so they survive refresh/logouts
        try:
            save_monthly_data(year, month, st.session_state[session_key])
        except Exception:
            pass

    st.rerun()

if session_key in st.session_state:
    st.subheader(
        f"📝 Edit Calendar for {calendar.month_name[month]} {year} Before Generating PDF")
    month_days = calendar.monthcalendar(year, month)
    for week in month_days:
        cols = st.columns(7)
        for c_idx, day in enumerate(week):
            if day == 0:
                with cols[c_idx]:
                    st.markdown(" ")
                continue
            d = dt.date(year, month, day)
            with cols[c_idx]:
                current_text = st.session_state[session_key].get(d, "")
                new_text = st.text_area(f"{day}", current_text, key=f"{session_key}_{d}", height=180)
                # Save edits back to session_state and disk immediately
                if new_text != current_text:
                    st.session_state[session_key][d] = new_text
                    try:
                        save_monthly_data(year, month, st.session_state[session_key])
                    except Exception:
                        pass

    if st.button("🔄 Reset This Month's Edits"):
        st.session_state.pop(session_key, None)
        # remove persisted file as well
        filename = f"calendar_data_{year}_{month:02d}.json"
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception:
            pass
        st.rerun()

    # -------------------------
    # Monthly PDF (A3) export
    # -------------------------
    if st.button("Generate Monthly PDF (A3 Landscape)"):
        bg_bytes = bg_file.read() if bg_file else None
        # Gather edited texts
        edited_texts = {
            dt.date.fromisoformat(k.split("_")[-1]): v
            for k, v in st.session_state.items()
            if k.startswith(session_key + "_")
        }
        # Fallback: use persisted edits if the above method didn't find any
        if not edited_texts:
            edited_texts = st.session_state.get(session_key, {})

        pdf_buf = draw_calendar_pdf(title, disclaimer, year, month,
                                    edited_texts, background_bytes=bg_bytes)
        st.success("✅ A3 PDF calendar generated successfully!")
        st.download_button(
            "📥 Download Calendar (A3 Landscape PDF)",
            data=pdf_buf,
            file_name=f"calendar_{year}_{month:02d}_A3.pdf",
            mime="application/pdf",
        )

    # -------------------------
    # Weekly selection UI
    # -------------------------
    st.markdown("---")
    st.write("## Weekly Exports (A4 Landscape)")
    weeks = get_weeks_in_month(year, month)
    if not weeks:
        st.info("No week ranges found for this month.")
    else:
        week_labels = [
            f"Week {i + 1}: {w[0].strftime('%b %d')} – {w[1].strftime('%b %d')}"
            for i, w in enumerate(weeks)]
        selected_week_idx = st.selectbox("📆 Select Week to Generate",
                                         range(len(weeks)),
                                         format_func=lambda i: week_labels[i])
        selected_week_range = weeks[selected_week_idx]

        # -------------------------
        # Weekly Preview with Image Editor
        # -------------------------
        st.markdown("---")
        st.write("## 🎨 Image Selection & Layout Editor")

        # Initialize image layouts in session state
        if "image_layouts" not in st.session_state:
            st.session_state.image_layouts = {}

        # Initialize selected images storage
        if "selected_images" not in st.session_state:
            st.session_state.selected_images = {}

        # Initialize image options cache
        if "image_options" not in st.session_state:
            st.session_state.image_options = {}

        # Initialize preview indices
        if "preview_week_idx" not in st.session_state:
            st.session_state.preview_week_idx = selected_week_idx
        if "preview_day_idx" not in st.session_state:
            st.session_state.preview_day_idx = 0

        # Ensure week is valid
        if weeks and 0 <= selected_week_idx < len(weeks):
            st.session_state.preview_week_idx = selected_week_idx
            start_date, end_date = weeks[st.session_state.preview_week_idx]
            week_days = [start_date + dt.timedelta(days=i) for i in
                         range((end_date - start_date).days + 1)]
            total_days = len(week_days)

            # Navigation buttons
            col_prev, col_next, col_reset = st.columns([1, 1, 1])
            with col_prev:
                if st.button("⬅️ Previous Day"):
                    st.session_state.preview_day_idx = max(0,
                                                           st.session_state.preview_day_idx - 1)
                    st.rerun()
            with col_next:
                if st.button("Next Day ➡️"):
                    st.session_state.preview_day_idx = min(total_days - 1,
                                                           st.session_state.preview_day_idx + 1)
                    st.rerun()
            with col_reset:
                if st.button("🔄 Reset Layout"):
                    current_day = week_days[st.session_state.preview_day_idx]
                    day_key = current_day.isoformat()
                    if day_key in st.session_state.image_layouts:
                        del st.session_state.image_layouts[day_key]
                    st.rerun()

            # Current day
            current_day = week_days[st.session_state.preview_day_idx]
            day_key = current_day.isoformat()

            text = st.session_state.get(f"{session_key}_{current_day}",
                                        "").strip()
            if not text:
                text = "(No activities planned)"

                # Extract activities for image fetching
            activities = extract_activities_from_text(text)
            unique_activities = []
            seen = set()
            for activity in activities:
                activity_lower = activity.lower().strip()
                if activity_lower not in seen:
                    unique_activities.append(activity)
                    seen.add(activity_lower)

            # Store all activities but only allow selecting 3
            max_selectable = 3

            # -------------------------
            # Image Selection Interface with Parallel Loading
            # -------------------------
            st.markdown("### 🖼️ Step 1: Choose Images for Each Activity")
            st.info(
                f"ℹ️ Found {len(unique_activities)} activities. You can select images for all of them, but only 3 will appear in the final document.")

            if not unique_activities:
                st.info("No activities found for this day.")
                images_list = []
            else:
                images_list = []

                # Initialize page numbers for refresh functionality
                if "image_page_numbers" not in st.session_state:
                    st.session_state.image_page_numbers = {}

                # Initialize page numbers for all activities
                for act_idx, activity in enumerate(unique_activities):
                    activity_key = f"{day_key}_{act_idx}"
                    if activity_key not in st.session_state.image_page_numbers:
                        st.session_state.image_page_numbers[activity_key] = 1

                # Check if we need to fetch images for any activities
                activities_to_fetch = []
                page_numbers_to_fetch = []

                for act_idx, activity in enumerate(unique_activities):
                    activity_key = f"{day_key}_{act_idx}"
                    if activity_key not in st.session_state.image_options:
                        activities_to_fetch.append(activity)
                        page_numbers_to_fetch.append(
                            st.session_state.image_page_numbers[activity_key])

                # Fetch all missing images in parallel
                if activities_to_fetch:
                    with st.spinner(
                            f"Loading images for {len(activities_to_fetch)} activities..."):
                        results = fetch_images_parallel(activities_to_fetch,
                                                        page_numbers_to_fetch)

                        # Store results in session state
                        fetch_idx = 0
                        for act_idx, activity in enumerate(unique_activities):
                            activity_key = f"{day_key}_{act_idx}"
                            if activity_key not in st.session_state.image_options:
                                if fetch_idx in results:
                                    st.session_state.image_options[
                                        activity_key] = results[fetch_idx][
                                        "images"]
                                fetch_idx += 1

                # Display all activities with their images
                for act_idx, activity in enumerate(unique_activities):
                    st.markdown(f"**Activity {act_idx + 1}: {activity}**")

                    activity_key = f"{day_key}_{act_idx}"
                    options = st.session_state.image_options.get(activity_key,
                                                                 [])

                    if not options:
                        st.warning(f"No images found for '{activity}'")
                        continue

                    # Display image options in columns
                    cols = st.columns(min(5, len(options)))

                    for img_idx, img_bytes in enumerate(options):
                        with cols[img_idx]:
                            st.image(img_bytes,
                                     caption=f"Option {img_idx + 1}",
                                     use_container_width=True)
                            if st.button(f"Select",
                                         key=f"select_{activity_key}_{img_idx}"):
                                st.session_state.selected_images[
                                    activity_key] = img_bytes
                                st.rerun()

                    # Refresh button and selection status
                    col_refresh, col_status = st.columns([1, 3])
                    with col_refresh:
                        if st.button("🔄 Refresh Images",
                                     key=f"refresh_{activity_key}"):
                            # Increment page number to get different results
                            st.session_state.image_page_numbers[
                                activity_key] += 1
                            page_num = st.session_state.image_page_numbers[
                                activity_key]
                            keyword = get_activity_keyword(activity)

                            with st.spinner(
                                    f"Fetching new images for '{activity}' (page {page_num})..."):
                                new_options = fetch_pexels_images(keyword,
                                                                  orientation="landscape",
                                                                  size="medium",
                                                                  count=5,
                                                                  page=page_num)

                            # Update options even if empty (to show "no more results")
                            st.session_state.image_options[
                                activity_key] = new_options

                            if not new_options:
                                # Reset to page 1 if we've run out of results
                                st.session_state.image_page_numbers[
                                    activity_key] = 1
                                with st.spinner(
                                        f"No more results. Fetching from page 1 using keyword: '{keyword}'..."):
                                    new_options = fetch_pexels_images(keyword,
                                                                      orientation="landscape",
                                                                      size="medium",
                                                                      count=5,
                                                                      page=1)
                                st.session_state.image_options[
                                    activity_key] = new_options

                            st.rerun()

                    with col_status:
                        if activity_key in st.session_state.selected_images:
                            st.success(f"✓ Image selected for {activity}")
                        else:
                            st.info(f"Please select an image for {activity}")

                    st.markdown("---")

                # Collect selected images for this day (limit to 3 for display)
                for act_idx, activity in enumerate(unique_activities):
                    activity_key = f"{day_key}_{act_idx}"
                    if activity_key in st.session_state.selected_images:
                        images_list.append(
                            st.session_state.selected_images[activity_key])
                        if len(images_list) >= max_selectable:
                            break

            # Get page dimensions
            page_width, page_height = landscape(A4)

            # Initialize or get layouts for this day
            if day_key not in st.session_state.image_layouts and images_list:
                st.session_state.image_layouts[
                    day_key] = get_default_image_layout(
                    len(images_list), page_width, page_height
                )

            current_layouts = st.session_state.image_layouts.get(day_key, [])

            # Image editor controls
            if images_list and current_layouts:
                st.markdown("### 📐 Step 2: Adjust Image Positions and Sizes")

                # Define default layouts
                default_layouts = [
                    {"x": 560, "y": 400, "width": 240, "height": 150},
                    {"x": 560, "y": 220, "width": 240, "height": 150},
                    {"x": 560, "y": 50, "width": 240, "height": 150}
                ]

                for idx in range(len(images_list)):
                    # Ensure layout entry exists for each image
                    if idx >= len(current_layouts):
                        if idx < len(default_layouts):
                            current_layouts.append(default_layouts[idx].copy())
                        else:
                            # Fallback for more than 3 images
                            current_layouts.append({
                                "x": 560,
                                "y": 50,
                                "width": 240,
                                "height": 150
                            })

                    with st.expander(f"Image {idx + 1} Position Controls",
                                     expanded=(idx == 0)):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Position (X, Y)**")
                            x_pos = st.slider(
                                f"X Position (Image {idx + 1})",
                                min_value=0,
                                max_value=int(page_width),
                                value=int(current_layouts[idx]["x"]),
                                step=5,
                                key=f"x_{day_key}_{idx}"
                            )
                            y_pos = st.slider(
                                f"Y Position (Image {idx + 1})",
                                min_value=0,
                                max_value=int(page_height),
                                value=int(current_layouts[idx]["y"]),
                                step=5,
                                key=f"y_{day_key}_{idx}"
                            )

                        with col2:
                            st.markdown("**Size (Width, Height)**")
                            width = st.slider(
                                f"Width (Image {idx + 1})",
                                min_value=50,
                                max_value=int(page_width * 0.5),
                                value=int(current_layouts[idx]["width"]),
                                step=5,
                                key=f"w_{day_key}_{idx}"
                            )
                            height = st.slider(
                                f"Height (Image {idx + 1})",
                                min_value=50,
                                max_value=int(page_height * 0.8),
                                value=int(current_layouts[idx]["height"]),
                                step=5,
                                key=f"h_{day_key}_{idx}"
                            )

                        # Update layout
                        current_layouts[idx] = {
                            "x": x_pos,
                            "y": y_pos,
                            "width": width,
                            "height": height
                        }

                # Save updated layouts
                st.session_state.image_layouts[day_key] = current_layouts

                # Generate preview
                st.markdown("### 📄 Step 3: Live Preview")
                preview_img = create_preview_image_with_layout(
                    page_width, page_height, current_day, text,
                    images_list, current_layouts
                )
                st.image(preview_img, use_container_width=True,
                         caption=f"Day {st.session_state.preview_day_idx + 1} of {total_days}")

            elif not images_list:
                st.info(
                    "🔍 Select images in Step 1 above to see the preview and layout controls.")

        else:
            st.info("Please select a valid week to preview.")

            # --- Generate selected week ---
        if st.button("📅 Generate Selected Week (A4 Landscape)"):
            with st.spinner(
                    f"Generating PDF for {week_labels[selected_week_idx]}..."):
                start_date, end_date = selected_week_range
                week_days = [start_date + dt.timedelta(days=i) for i in
                             range((end_date - start_date).days + 1)]

                buf = BytesIO()
                c = canvas.Canvas(buf, pagesize=landscape(A4))
                width, height = landscape(A4)

                for d in week_days:
                    text = st.session_state.get(f"{session_key}_{d}",
                                                "").strip()
                    if not text:
                        text = "(No activities planned)"

                    # Use selected images if available
                    d_key = d.isoformat()
                    activities = extract_activities_from_text(text)
                    unique_activities = []
                    seen = set()
                    for activity in activities:
                        activity_lower = activity.lower().strip()
                        if activity_lower not in seen:
                            unique_activities.append(activity)
                            seen.add(activity_lower)

                    # Collect only selected images (up to 3)
                    images_list = []
                    for act_idx, activity in enumerate(unique_activities):
                        activity_key = f"{d_key}_{act_idx}"
                        if activity_key in st.session_state.selected_images:
                            images_list.append(
                                st.session_state.selected_images[activity_key])
                            if len(images_list) >= 3:
                                break

                    # Get custom layout if exists
                    if d_key in st.session_state.image_layouts and images_list:
                        layouts = st.session_state.image_layouts[d_key]
                    else:
                        layouts = get_default_image_layout(len(images_list),
                                                           width,
                                                           height) if images_list else None

                    draw_weekly_page_with_custom_layout(c, width, height, d,
                                                        text, images_list,
                                                        layouts)
                    c.showPage()

                c.save()
                buf.seek(0)
                st.success(
                    f"✅ PDF for {week_labels[selected_week_idx]} generated successfully!")
                st.download_button(
                    "📥 Download Selected Week (A4 Landscape)",
                    data=buf,
                    file_name=f"week_{selected_week_idx + 1}_{year}_{month:02d}.pdf",
                    mime="application/pdf",
                )

            # --- Generate all weeks ---
        if st.button("📅 Generate All Weeks (A4 Landscape)"):
            with st.spinner("Generating PDFs for all weeks..."):
                all_week_buffers = []
                for wk_idx, (start_date, end_date) in enumerate(weeks):
                    buf = BytesIO()
                    c = canvas.Canvas(buf, pagesize=landscape(A4))
                    width, height = landscape(A4)

                    week_days = [start_date + dt.timedelta(days=i) for i in
                                 range((end_date - start_date).days + 1)]
                    for d in week_days:
                        text = st.session_state.get(f"{session_key}_{d}",
                                                    "").strip()
                        if not text:
                            text = "(No activities planned)"

                        # Use selected images if available
                        d_key = d.isoformat()
                        activities = extract_activities_from_text(text)
                        unique_activities = []
                        seen = set()
                        for activity in activities:
                            activity_lower = activity.lower().strip()
                            if activity_lower not in seen:
                                unique_activities.append(activity)
                                seen.add(activity_lower)

                        # Collect only selected images (up to 3)
                        images_list = []
                        for act_idx, activity in enumerate(unique_activities):
                            activity_key = f"{d_key}_{act_idx}"
                            if activity_key in st.session_state.selected_images:
                                images_list.append(
                                    st.session_state.selected_images[
                                        activity_key])
                                if len(images_list) >= 3:
                                    break

                        # Get custom layout if exists
                        if d_key in st.session_state.image_layouts and images_list:
                            layouts = st.session_state.image_layouts[d_key]
                        else:
                            layouts = get_default_image_layout(
                                len(images_list), width,
                                height) if images_list else None

                        draw_weekly_page_with_custom_layout(c, width, height,
                                                            d, text,
                                                            images_list,
                                                            layouts)
                        c.showPage()

                    c.save()
                    buf.seek(0)
                    all_week_buffers.append(buf.getvalue())

                # Merge all per-week buffers into one PDF
                merger = PyPDF2.PdfMerger()
                for pdf in all_week_buffers:
                    merger.append(BytesIO(pdf))
                merged_output = BytesIO()
                merger.write(merged_output)
                merger.close()
                merged_output.seek(0)

                st.success("✅ Weekly A4 PDFs generated successfully!")
                st.download_button(
                    "📥 Download Weekly Calendar (A4 Landscape)",
                    data=merged_output,
                    file_name=f"weekly_calendar_{year}_{month:02d}.pdf",
                    mime="application/pdf",
                )

# End of file
