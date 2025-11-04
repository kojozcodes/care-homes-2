import streamlit as st
import hashlib
import pandas as pd
import datetime as dt
import calendar
from io import BytesIO
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color, black, white
import textwrap
import re
import requests
import json
import os

# -------------------------
# Streamlit page setup
# -------------------------
st.set_page_config(page_title="Care Home Monthly Calendar", layout="wide")

# -------------------------
# Secure single-user login
# -------------------------
# Read password from Streamlit Secrets (not in code!)
REAL_PASSWORD = st.secrets["APP_PASSWORD"]
PASSWORD_HASH = hashlib.sha256(REAL_PASSWORD.encode()).hexdigest()

# Initialize login session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Secure Access")
    password = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH:
            st.session_state.logged_in = True
            st.success("Access granted ✅")
            st.rerun()
        else:
            st.error("Incorrect password. Try again.")
    st.stop()  # Stops the rest of the app from loading until login

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
# Utility functions
# -------------------------
def parse_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"CSV parse error: {e}")
        return None


def month_date_range(year: int, month: int):
    first = dt.date(year, month, 1)
    last = dt.date(year, month, calendar.monthrange(year, month)[1])
    return first, last


def clean_text(s):
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    # Replace known special characters and invisible breaks
    replacements = {
        "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\xa0": " ",
        "\r": " ", "\n": " ", "\u2028": " ", "\u2029": " ", "\ufeff": " ",
        "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "",  # zero-width chars
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    # Remove any other non-printable or non-ASCII characters
    s = re.sub(r"[^\x20-\x7E]", " ", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()



@st.cache_data
def load_all_holidays():
    try:
        with open("holidays_2025_2026.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("holidays", [])
    except Exception as e:
        st.warning(f"Could not load holidays file: {e}")
        return []


ALL_HOLIDAYS = load_all_holidays()


# -------------------------
# Holiday fetcher
# -------------------------
def fetch_selected_holidays(year, month, selected_names=None):
    """Return holidays for this month/year, filtered by selected_names if provided."""
    holidays_list = []
    for h in ALL_HOLIDAYS:
        try:
            d = dt.datetime.strptime(h["date"], "%Y-%m-%d").date()
        except:
            continue
        if d.year == year and d.month == month:
            name = clean_text(h["name"])
            if (not selected_names) or (name in selected_names):
                holidays_list.append({"date": d, "title": name, "notes": "Holiday"})
    return holidays_list



# -------------------------
# Core: Build calendar day mapping
# -------------------------
def seat_activity_into_calendar(year, month, activities_df, rota_df, rules,
                                include_holidays=True, daily_rules=None):
    first, last = month_date_range(year, month)
    daymap = {first + dt.timedelta(days=i): [] for i in
              range((last - first).days + 1)}

    # 1️⃣ Holidays (auto-fetch)
    if include_holidays:
        seen_holidays = set()  # to track (date, normalized_title)

        # Combine all sources
        combined_holidays = fetch_selected_holidays(year, month, st.session_state.get("selected_holidays"))


        for ev in combined_holidays:
            d = ev["date"]
            title_norm = clean_text(ev["title"]).strip().lower()

            # Skip duplicates by date + normalized title
            if (d, title_norm) in seen_holidays:
                continue
            seen_holidays.add((d, title_norm))

            if d in daymap:
                # Combine multiple holidays on same day
                existing_titles = [e["title"] for e in daymap[d] if
                                   e["notes"] == "Holiday"]
                if existing_titles:
                    combined = " / ".join(
                        sorted(set(existing_titles + [ev["title"]])))
                    # Replace existing entry with combined title
                    daymap[d] = [e for e in daymap[d] if
                                 e["notes"] != "Holiday"]
                    daymap[d].append(
                        {"time": None, "title": combined, "notes": "Holiday"})
                else:
                    daymap[d].append({"time": None, "title": ev["title"],
                                      "notes": "Holiday"})

    # 2️⃣ Staff Shifts
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
                    daymap[d].append({"time": None, "title": display,
                                      "notes": "staff shift"})

    # 3️⃣ Fixed Weekly Rules
    fixed_rules = []
    for rule in rules:
        for d in daymap:
            if d.weekday() == rule["weekday"]:
                fixed_rules.append({"date": d, "time": rule.get("time"),
                                    "title": clean_text(rule["title"]),
                                    "notes": "fixed"})

    # 3️⃣b Fixed Daily Rules
    if daily_rules:
        for d in daymap:
            for rule in daily_rules:
                daymap[d].append({
                    "date": d,
                    "time": rule.get("time"),
                    "title": clean_text(rule["title"]),
                    "notes": "fixed daily"
                })

    # 4️⃣ Regular Activities
    activities = []
    if activities_df is not None:
        for _, r in activities_df.iterrows():
            name = clean_text(r.get("name") or r.get("activity_name") or "")
            pref_days = str(r.get("preferred_days", "")).split(";")
            pref_days = [p.strip()[:3].lower() for p in pref_days if p.strip()]
            pref_time = str(r.get("preferred_time", "")).strip()
            freq = int(r.get("frequency", 0)) if str(
                r.get("frequency", "")).isdigit() else 0
            placed = 0
            for d in sorted(daymap.keys()):
                if freq and placed >= freq:
                    break
                dow3 = calendar.day_name[d.weekday()][:3].lower()
                if dow3 in pref_days:
                    activities.append(
                        {"date": d, "time": pref_time, "title": name,
                         "notes": "activity"})
                    placed += 1

    # 5️⃣ Merge + Normalize Times + Deduplicate + Sort
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
        duplicates = [e for e in daymap[d] if
                      e["title"].lower().strip() == title_norm]
        if duplicates:
            has_exact = any(e.get("time") == time_norm for e in duplicates)
            has_proper = any(
                e.get("time") and len(e.get("time")) == 5 for e in duplicates)
            if has_exact or (has_proper and not time_norm):
                continue
        daymap[d].append(
            {"time": time_norm, "title": ev["title"], "notes": ev["notes"]})

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


def draw_calendar_pdf(title, disclaimer, year, month, cell_texts,
                      background_bytes=None):
    """Generate styled non-editable A3 calendar PDF with improved readability and formatting"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A3))
    width, height = landscape(A3)

    # --------------------------
    # Background
    # --------------------------
    if background_bytes:
        try:
            img = ImageReader(BytesIO(background_bytes))
            c.drawImage(img, 0, 0, width=width, height=height,
                        preserveAspectRatio=False, mask="auto")
        except Exception as e:
            st.warning(f"Background load failed: {e}")

    # --------------------------
    # Header (Month & Year + Disclaimer, each with pill background)
    # --------------------------
    title_text = clean_text(title)
    disclaimer_text = clean_text(disclaimer)

    # === Month Pill ===
    title_font = "Helvetica-Bold"
    title_size = 20
    c.setFont(title_font, title_size)
    title_width = c.stringWidth(title_text, title_font, title_size)

    # Pill dimensions and position
    side_padding = 15 * mm
    vertical_padding = 4 * mm
    pill_w = title_width + side_padding
    pill_h = 4 * mm + vertical_padding
    pill_y = height - 10 * mm  # move up slightly
    pill_x = (width - pill_w) / 2

    # Draw month pill (black background, rounded)
    month_pill_color = Color(0, 0, 0)
    c.setFillColor(month_pill_color)
    c.roundRect(pill_x, pill_y, pill_w, pill_h, pill_h / 2, fill=1, stroke=0)

    # Draw month text (white)
    c.setFillColor(white)
    text_y = pill_y + (pill_h / 2) - (title_size / 3.2)
    c.drawCentredString(width / 2, text_y, title_text)

    # === Disclaimer Pill ===
    disclaimer_font = "Helvetica-Bold"
    disclaimer_size = 11
    c.setFont(disclaimer_font, disclaimer_size)
    disclaimer_width = c.stringWidth(disclaimer_text, disclaimer_font,
                                     disclaimer_size)

    disc_padding_x = 10 * mm
    disc_padding_y = 1 * mm
    disc_w = disclaimer_width + disc_padding_x
    disc_h = 6 * mm + disc_padding_y
    disc_x = (width - disc_w) / 2
    disc_y = pill_y - disc_h - 0.5 * mm  # small gap below the month pill

    # Draw disclaimer pill (light grey)
    disc_pill_color = Color(0, 0, 0)
    c.setFillColor(disc_pill_color)
    c.roundRect(disc_x, disc_y, disc_w, disc_h, disc_h / 2, fill=1, stroke=0)

    # Draw disclaimer text (black, centered)
    c.setFillColor(white)
    disc_text_y = disc_y + (disc_h / 2) - (disclaimer_size / 3)
    c.drawCentredString(width / 2, disc_text_y, disclaimer_text)

    # --------------------------
    # Layout
    # --------------------------
    left, right, top, bottom = 4 * mm, 4 * mm, 37 * mm, 5 * mm
    grid_w = width - left - right
    cols, rows = 7, 5
    col_w = grid_w / cols

    # Weekday header bar (Mon–Sun)
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

    # 🔽 Control gap between bar and top of first calendar row
    bar_gap = 1.5 * mm  # reduce or increase to adjust spacing (try 3–6mm)
    top_of_grid = bar_y - bar_gap

    # Keep grid height consistent below
    grid_h = top_of_grid - bottom
    row_h = grid_h / rows

    # --------------------------
    # Calendar cells
    # --------------------------
    cream = Color(1, 1, 1, alpha=0.93)
    staff_blue = Color(0, 0.298, 0.6)
    month_days = calendar.monthcalendar(year, month)

    for r_idx, week in enumerate(month_days):
        for c_idx, day in enumerate(week):
            if day == 0:
                continue

            d = dt.date(year, month, day)
            x = left + c_idx * col_w
            y = bottom + (rows - 1 - r_idx) * row_h

            # Background + border
            c.setFillColor(cream)
            c.setStrokeColor(black)
            c.roundRect(x, y, col_w, row_h, 5, fill=1, stroke=1)

            # --- Date (top-right, bold)
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(black)

            # Measure text width so it's properly aligned to the right margin of the cell
            day_str = str(day)
            day_width = c.stringWidth(day_str, "Helvetica-Bold", 12)

            # Position a few millimetres from the right edge and near the top
            c.drawString(x + col_w - day_width - 1.2 * mm, y + row_h - 4.5 * mm,
                         day_str)

            # --- Prepare text
            lines = cell_texts.get(d, "").split("\n")
            text_y = y + row_h - 3.5 * mm
            line_spacing = 4 * mm  # more readable spacing

            for line in lines:
                line = clean_text(line).strip()
                if not line:
                    continue

                # 🧩 Handle each line separately
                if line.isupper():
                    # 🔹 Dynamic wrapping for holidays (skip generic wrapping)
                    max_text_width = col_w - (day_width + 6 * mm)
                    words = line.split()
                    current_line = ""
                    wrapped_holiday = []

                    for word in words:
                        test_line = (current_line + " " + word).strip()
                        line_width = c.stringWidth(test_line, "Helvetica-Bold",
                                                   8.7)
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

                        # underline
                        text_width = c.stringWidth(wh, "Helvetica-Bold", 8.7)
                        underline_y = text_y - 0.5 * mm
                        c.line(x + 2 * mm, underline_y,
                               x + 2 * mm + text_width, underline_y)
                        text_y -= line_spacing
                    continue  # move to next line (skip normal wrapping below)

                # 🔹 For non-holiday lines, smart wrapping (no broken times or parentheses)
                c.setFont("Helvetica-Bold", 10.5)
                max_text_width = col_w - (day_width + 0.5 * mm)

                # Handle staff lines separately (never wrap mid-line)
                if line.lower().startswith("staff:"):
                    c.setFont("Helvetica-Oblique", 10.5)
                    c.setFillColor(staff_blue)
                    c.drawString(x + 2 * mm, text_y, line)
                    text_y -= line_spacing - 1
                    continue

                # Handle activity lines — wrap neatly without cutting times
                time_match = re.match(
                    r"^(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?)\s?(.*)", line)
                if time_match:
                    time_part, rest = time_match.groups()
                    rest = rest.strip()

                    # Draw the time part first
                    c.setFont("Helvetica-Bold", 10.5)
                    c.setFillColor(black)
                    c.drawString(x + 2 * mm, text_y, time_part)

                    # Start wrapping the rest dynamically
                    time_width = c.stringWidth(time_part + " ",
                                               "Helvetica-Bold", 10.5)
                    available_width = max_text_width - time_width

                    words = rest.split()
                    current_line = ""
                    wrapped_lines = []

                    for word in words:
                        test_line = (current_line + " " + word).strip()
                        if c.stringWidth(test_line, "Helvetica-Bold",
                                         10.5) > available_width and current_line:
                            wrapped_lines.append(current_line)
                            current_line = word
                        else:
                            current_line = test_line
                    if current_line:
                        wrapped_lines.append(current_line)

                    # Draw wrapped lines
                    first_line = True
                    for wline in wrapped_lines:
                        wline = wline.strip()
                        if not wline:
                            continue
                        if first_line:
                            c.drawString(x + 2 * mm + time_width, text_y,
                                         wline)
                            first_line = False
                        else:
                            text_y -= line_spacing
                            c.drawString(x + 2 * mm, text_y, wline)

                    text_y -= line_spacing
                    if text_y < y + 4 * mm:
                        break
                    continue

                # All other normal text (wrap safely)
                words = line.split()
                current_line = ""
                wrapped_lines = []
                for word in words:
                    test_line = (current_line + " " + word).strip()
                    if c.stringWidth(test_line, "Helvetica-Bold",
                                     10.5) > max_text_width and current_line:
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

                    # 🔹 Staff (italic, blue)
                    if subline.lower().startswith("staff:"):
                        c.setFont("Helvetica-Oblique", 10.5)
                        c.setFillColor(staff_blue)
                        c.drawString(x + 2 * mm, text_y, subline)
                        text_y -= line_spacing - 1
                        continue

                    # 🔹 Activities (bold time, normal text)
                    time_match = re.match(
                        r"^(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?)\s?(.*)", subline)
                    if time_match:
                        time_part, rest = time_match.groups()
                        c.setFont("Helvetica-Bold", 10.5)
                        c.setFillColor(black)
                        c.drawString(x + 2 * mm, text_y, time_part)
                        time_width = c.stringWidth(time_part + " ",
                                                   "Helvetica-Bold", 9.5)
                        c.setFont("Helvetica-Bold", 10.5)
                        c.drawString(x + 2 * mm + time_width, text_y, rest)
                    else:
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
# Streamlit UI
# -------------------------
st.title("🏡 Care Home Monthly Activities — Editable Preview & A3 PDF")

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
    - `staff` → Staff member’s full name  
    - `shift_start` → Start time (e.g. `09:00`)
    - `shift_end` → End time (e.g. `16:30`)
    - `role` → (Optional) Staff role or position

    **Example:**
    | date       | staff  | shift_start | shift_end | role      |
    |-------------|--------|--------------|------------|-----------|
    | 2025-11-01  | Lucy   | 09:00        | 16:30      | activities     |
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

    **Example:**
    | name             | preferred_days | preferred_time | frequency | staff_required | notes                    |
    |------------------|----------------|----------------|------------|----------------|---------------------------|
    | Coffee & Chat      | Mon;Wed;Fri;Sun            | 11:00          | 12          | 1              | Social session with refreshments  |
    """)

rota_df = parse_csv(st.file_uploader("📂 Upload Staff Rota CSV", type=["csv"]))
activities_df = parse_csv(
    st.file_uploader("📂 Upload Activities CSV", type=["csv"]))

bg_file = st.file_uploader("Background Image (optional)",
                           type=["png", "jpg", "jpeg"])

# Load persisted settings
if "settings" not in st.session_state:
    st.session_state["settings"] = load_settings()

saved_weekly = st.session_state["settings"].get("weekly_rules",
    "Film Night:Thu:18:00\nDogs for Health:Thu:11:00\nReminiscence:Sat:18:00"
)
saved_daily = st.session_state["settings"].get("daily_rules",
    "Morning Exercise:09:00\nNews Headlines:10:00"
)


fixed_rules_text = st.text_area(
    "Fixed Weekly Rules (e.g. Film Night:Thu:18:00)",
    value=saved_weekly,
    key="weekly_rules_input"
)

daily_rules_text = st.text_area(
    "Fixed Daily Rules (e.g. Morning Exercise:09:00)",
    value=saved_daily,
    key="daily_rules_input"
)

if st.button("💾 Save Default Rules"):
    st.session_state["settings"]["weekly_rules"] = st.session_state["weekly_rules_input"]
    st.session_state["settings"]["daily_rules"] = st.session_state["daily_rules_input"]
    save_settings(st.session_state["settings"])
    st.success("✅ Default rules saved successfully!")


rules = []
for line in fixed_rules_text.splitlines():
    parts = [p.strip() for p in line.split(":")]
    if len(parts) >= 2:
        day = parts[1][:3].lower()
        time = parts[2] if len(parts) > 2 else ""
        title_txt = parts[0]
        weekday = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].index(day)
        rules.append({"weekday": weekday, "time": time, "title": title_txt})

# 🔹 ADD THIS BELOW weekly rule parsing
# 🔹 ADD THIS BELOW weekly rule parsing
daily_rules = []
for line in daily_rules_text.splitlines():
    line = line.strip()
    if not line:
        continue

    # Split on only the first colon (to keep "09:00" intact)
    parts = [p.strip() for p in line.split(":", 1)]
    if len(parts) == 2:
        title_txt, time = parts
    else:
        title_txt, time = parts[0], ""

    if title_txt:
        daily_rules.append({"time": time, "title": title_txt})

include_holidays = st.checkbox("Include UK National Holidays", True)

# -------------------------
# Holiday selection system (simple border around each day group)
# -------------------------
if include_holidays:
    st.markdown("### 🗓️ Select Holidays to Include")

    # Group holidays by date
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
        selected_holidays = []

        # Simple CSS border style
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

        # Display in table-like layout by week
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
                        unsafe_allow_html=True
                    )
                    if not day_holidays:
                        st.markdown("<em>No holidays</em>", unsafe_allow_html=True)
                    else:
                        for name in sorted(set(day_holidays)):
                            key = f"hol_{year}-{month:02d}-{day:02d}_{name}"
                            if st.checkbox(name, value=True, key=key):
                                selected_holidays.append(name)
                    st.markdown("</div>", unsafe_allow_html=True)

        st.session_state["selected_holidays"] = selected_holidays


# -------------------------
# Preview and Editable Calendar Section
# -------------------------

# Create a unique session key for each (year, month) combo
session_key = f"{year}-{month:02d}"

if st.button("Preview Calendar"):
    with st.spinner("Generating preview..."):
        daymap = seat_activity_into_calendar(
            year, month, activities_df, rota_df, rules, include_holidays,
            daily_rules
        )
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

# Editable preview (only for currently selected month)
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
                st.text_area(
                    f"{day}",
                    st.session_state[session_key].get(d, ""),
                    key=f"{session_key}_{d}",
                    height=180,
                )

    # Optional reset button for this month’s edits
    if st.button("🔄 Reset This Month's Edits"):
        st.session_state.pop(session_key, None)
        st.rerun()

    # Generate PDF button
    if st.button("Generate PDF"):
        bg_bytes = bg_file.read() if bg_file else None

        # Gather edited text areas for this month only
        edited_texts = {
            dt.date.fromisoformat(k.split("_")[-1]): v
            for k, v in st.session_state.items()
            if k.startswith(session_key + "_")
        }

        pdf_buf = draw_calendar_pdf(
            title, disclaimer, year, month, edited_texts,
            background_bytes=bg_bytes
        )

        st.success("✅ A3 PDF calendar generated successfully!")
        st.download_button(
            "📥 Download Calendar (A3 Landscape PDF)",
            data=pdf_buf,
            file_name=f"calendar_{year}_{month:02d}_A3.pdf",
            mime="application/pdf",
        )
