"""
main.py

Care Home Monthly Calendar — Streamlit app (annotated for beginners)

What this script does:
- Provides a secured (single-user) Streamlit interface to create an editable
  monthly activities calendar for a care home.
- Accepts CSV uploads for staff rota and regular activities.
- Auto-inserts selected public holidays from a local JSON file.
- Lets the user preview and edit the month in the browser, then generate a
  styled A3 landscape PDF using ReportLab.

How to run:
1. Put this file in the same folder as `holidays_2025_2026.json` (optional).
2. Set a Streamlit secret named `APP_PASSWORD` (Streamlit secrets mechanism).
3. Install dependencies: streamlit, pandas, reportlab (and others used).
4. Run with: `streamlit run main.py`

Notes for beginners:
- This file is intentionally documented with detailed comments to help you learn.
- Comments start with `#`. They do not affect how the program runs.
- If you change logic, keep the indentation exactly as Python is indentation sensitive.
- The code below is the same as your working app — only comments were added.
"""

import streamlit as st
import hashlib
import pandas as pd
import datetime as dt
import calendar
from io import BytesIO
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.utils import ImageReader
import re
import json
import os
import PyPDF2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, black, white
from reportlab.lib.units import mm

# -------------------------
# Streamlit page setup
# -------------------------
# `st.set_page_config` configures the Streamlit app's top-level layout and title.
# - page_title: the browser tab title.
# - layout="wide": gives more horizontal space for two-column layout.
st.set_page_config(page_title="Care Home Monthly Calendar", layout="wide")

# -------------------------
# Secure single-user login
# -------------------------
# We store the real password in Streamlit Secrets (not in this file).
# This keeps sensitive information out of source control.
REAL_PASSWORD = st.secrets["APP_PASSWORD"]

# Hash the real password with SHA256. We will compare the hash of the
# typed password to this value. Storing and comparing hashes is safer
# than comparing raw passwords.
PASSWORD_HASH = hashlib.sha256(REAL_PASSWORD.encode()).hexdigest()

# Use Streamlit's session_state to remember whether the user is logged in.
# session_state persists across reruns in the same browser session.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# If the user is not logged in, show a simple password prompt.
# st.stop() prevents the rest of the app from loading until login succeeds.
if not st.session_state.logged_in:
    st.title("🔐 Secure Access")
    # A password input field hides the characters as the user types.
    password = st.text_input("Enter password", type="password")

    # When the user clicks "Login", we hash their input and compare.
    if st.button("Login"):
        if hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH:
            # Correct password: set logged_in flag and refresh the app.
            st.session_state.logged_in = True
            st.success("Access granted ✅")
            st.rerun()  # re-run the script so the rest of the app displays
        else:
            # Wrong password: show an error and remain on the login screen.
            st.error("Incorrect password. Try again.")
    # Stop executing the rest of the script while not logged in.
    st.stop()  # Stops the rest of the app from loading until login

# -------------------------
# Settings file handling
# -------------------------
# We'll persist some user preferences (like default rules) to a local JSON file.
SETTINGS_FILE = "calendar_settings.json"


def load_settings():
    """
    Read the JSON settings file if it exists and return its contents as a dict.
    If the file doesn't exist or can't be read, return an empty dict.

    This keeps user preferences between runs (for example, default weekly rules).
    """
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # If parsing fails, treat as no settings saved.
            return {}
    return {}


def save_settings(data):
    """
    Write the provided dict `data` to the settings JSON file.
    On error, show a Streamlit message for debugging.
    """
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving settings: {e}")

# -------------------------
# Utility functions
# -------------------------
def parse_csv(uploaded_file):
    """
    Read an uploaded file (from Streamlit's file_uploader) into a pandas DataFrame.
    Returns None if no file provided or if parsing fails.
    """
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"CSV parse error: {e}")
        return None


def month_date_range(year: int, month: int):
    """
    Return the first and last date objects for a given year and month.
    This helps build the dictionary of calendar days later.
    """
    first = dt.date(year, month, 1)
    last = dt.date(year, month, calendar.monthrange(year, month)[1])
    return first, last


def clean_text(s):
    """
    Normalise and "clean" text input:
    - Convert non-strings to strings safely.
    - Replace common unicode punctuation and invisible characters.
    - Remove non-printable characters (restrict to basic ASCII range).
    - Collapse multiple spaces into one and trim.
    This helps avoid strange characters when rendering into the PDF.
    """
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    # Replace known special characters and invisible breaks with safer alternatives.
    replacements = {
        "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\xa0": " ",
        "\r": " ", "\n": " ", "\u2028": " ", "\u2029": " ", "\ufeff": " ",
        "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "",
        # zero-width chars
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    # Remove any other non-printable or non-ASCII characters by replacing them with a space.
    s = re.sub(r"[^\x20-\x7E]", " ", s)
    # Collapse multiple whitespace characters into a single space.
    s = re.sub(r"\s+", " ", s)
    return s.strip()


@st.cache_data
def load_all_holidays():
    """
    Load a local JSON file containing holidays for 2025–2026.

    - Uses @st.cache_data so this file is read only once per session unless it changes.
    - Returns a list of holiday dicts from the JSON under the "holidays" key.
    - If loading fails, return an empty list and warn the user.
    """
    try:
        with open("holidays_2025_2026.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("holidays", [])
    except Exception as e:
        st.warning(f"Could not load holidays file: {e}")
        return []


# Load cached holidays into a module-level variable for reuse.
ALL_HOLIDAYS = load_all_holidays()

# -------------------------
# Holiday fetcher
# -------------------------
def fetch_selected_holidays(year, month, selected_names=None):
    """
    Return holidays for the specified year and month.

    - `selected_names` can be a list of holiday names the user wants included.
    - The function normalises names (with clean_text and .lower()) before filtering.
    - It returns a list of dicts with keys: date (datetime.date), title (string), notes ("Holiday").
    """
    holidays_list = []

    # Normalise selected_names into a set for quick comparisons.
    selected_normalized = set()
    if selected_names:
        for s in selected_names:
            selected_normalized.add(clean_text(s).lower())

    for h in ALL_HOLIDAYS:
        try:
            d = dt.datetime.strptime(h["date"], "%Y-%m-%d").date()
        except:
            # Skip entries with invalid dates.
            continue

        if d.year == year and d.month == month:
            name = clean_text(h["name"])
            normalized_name = name.lower()

            # Include all if no filter supplied, otherwise only include if name matches.
            if (not selected_names) or (normalized_name in selected_normalized):
                holidays_list.append({
                    "date": d,
                    "title": name,
                    "notes": "Holiday"
                })

    return holidays_list

# -------------------------
# Core: Build calendar day mapping
# -------------------------
def seat_activity_into_calendar(year, month, activities_df, rota_df, rules,
                                include_holidays=True, daily_rules=None):
    """
    Produce a dictionary (daymap) mapping each date in the month to a list of event dicts.

    Events have keys:
    - "time": optional string like "09:00" or None
    - "title": event title
    - "notes": classification like "Holiday", "staff shift", "fixed", "fixed daily", "activity"

    Sources used (in priority/order):
    1. Holidays (optional, fetched)
    2. Staff rota (CSV)
    3. Fixed weekly rules (user supplied)
    4. Fixed daily rules (user supplied)
    5. Regular activities (CSV preferences)
    """
    first, last = month_date_range(year, month)

    # Build dictionary for every day in the month, initialize with empty lists.
    daymap = {first + dt.timedelta(days=i): [] for i in
              range((last - first).days + 1)}

    # 1️⃣ Holidays (auto-fetch)
    if include_holidays:
        seen_holidays = set()  # Track (date, normalized_title) to avoid duplicates.

        # Fetch holidays for this month - respects st.session_state selected_holidays.
        combined_holidays = fetch_selected_holidays(year, month,
                                                    st.session_state.get(
                                                        "selected_holidays"))

        for ev in combined_holidays:
            d = ev["date"]
            title_norm = clean_text(ev["title"]).strip().lower()

            # Skip exact duplicates by date + normalized title.
            if (d, title_norm) in seen_holidays:
                continue
            seen_holidays.add((d, title_norm))

            if d in daymap:
                # If the day already has a Holiday entry, combine titles with " / ".
                existing_titles = [e["title"] for e in daymap[d] if
                                   e["notes"] == "Holiday"]
                if existing_titles:
                    combined = " / ".join(
                        sorted(set(existing_titles + [ev["title"]])))
                    # Remove old holiday entries, replace with combined title.
                    daymap[d] = [e for e in daymap[d] if
                                 e["notes"] != "Holiday"]
                    daymap[d].append(
                        {"time": None, "title": combined, "notes": "Holiday"})
                else:
                    # Otherwise just append the holiday entry for that day.
                    daymap[d].append({"time": None, "title": ev["title"],
                                      "notes": "Holiday"})

    # 2️⃣ Staff Shifts (read from rota_df if provided)
    if rota_df is not None:
        # Iterate through each row of the rota DataFrame.
        for _, r in rota_df.iterrows():
            try:
                # Convert the 'date' column to a Python date object.
                d = pd.to_datetime(r.get("date")).date()
            except:
                # If parsing fails for this row, skip it.
                continue
            if d in daymap:
                # Clean staff name and strip trailing numbers that may be attached.
                staff = clean_text(str(r.get("staff", "")))
                staff = re.sub(r"\s*\d+$", "", staff)
                start = str(r.get("shift_start", "")).strip()
                end = str(r.get("shift_end", "")).strip()
                shift_time = f"({start} – {end})" if start and end else ""
                display = f"{staff} {shift_time}".strip()
                if display:
                    # Add the staff line to the day's list.
                    daymap[d].append({"time": None, "title": display,
                                      "notes": "staff shift"})

    # 3️⃣ Fixed Weekly Rules (e.g. Film Night every Thursday)
    fixed_rules = []
    for rule in rules:
        # For each day in the month, if the weekday matches, schedule the fixed item.
        for d in daymap:
            if d.weekday() == rule["weekday"]:
                fixed_rules.append({"date": d, "time": rule.get("time"),
                                    "title": clean_text(rule["title"]),
                                    "notes": "fixed"})

    # 3️⃣b Fixed Daily Rules (same every day)
    if daily_rules:
        for d in daymap:
            for rule in daily_rules:
                daymap[d].append({
                    "date": d,
                    "time": rule.get("time"),
                    "title": clean_text(rule["title"]),
                    "notes": "fixed daily"
                })

    # 4️⃣ Regular Activities (from activities_df) — place according to preferred days and frequency
    activities = []
    if activities_df is not None:
        for _, r in activities_df.iterrows():
            # Try different column names for activity title for robustness.
            name = clean_text(r.get("name") or r.get("activity_name") or "")
            # preferred_days is expected as "Mon; Wed; Fri" etc.
            pref_days = str(r.get("preferred_days", "")).split(";")
            # Normalize to 3-letter lowercase day names like "mon", "wed".
            pref_days = [p.strip()[:3].lower() for p in pref_days if p.strip()]
            pref_time = str(r.get("preferred_time", "")).strip()
            # frequency is how many times per month this activity should appear.
            freq = int(r.get("frequency", 0)) if str(
                r.get("frequency", "")).isdigit() else 0
            placed = 0
            # Walk through all days in date order and schedule up to freq occurrences.
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
    # time_pattern accepts "9", "09", "9:30", "09:30" etc.
    time_pattern = re.compile(r"^(\d{1,2})(?::?(\d{2}))?$")

    def normalize_time(t):
        """
        Convert various simple time formats into "HH:MM" with zero padding.
        Returns None if the time can't be normalised.
        """
        if not t or not isinstance(t, str):
            return None
        # Remove dots, spaces and convert to lower-case to make matching easier.
        t2 = t.strip().lower().replace(".", ":").replace(" ", "")
        match = time_pattern.match(t2)
        if match:
            hour, minute = match.groups()
            hour = hour.zfill(2)
            minute = minute if minute else "00"
            return f"{hour}:{minute}"
        return None

    # Combine fixed rules and activities into one list to normalise and dedupe.
    all_events = fixed_rules + activities
    for ev in all_events:
        ev["time"] = normalize_time(ev.get("time"))

    # Insert events into daymap with deduplication logic.
    for ev in all_events:
        d = ev["date"]
        if d not in daymap:
            continue
        title_norm = ev["title"].lower().strip()
        time_norm = ev.get("time")
        # Find existing entries on that day with the same title.
        duplicates = [e for e in daymap[d] if
                      e["title"].lower().strip() == title_norm]
        if duplicates:
            # If an existing duplicate has the exact same time, skip adding.
            has_exact = any(e.get("time") == time_norm for e in duplicates)
            # If an existing duplicate has a "proper" time (HH:MM) and this
            # candidate has no time, prefer the existing proper time and skip.
            has_proper = any(
                e.get("time") and len(e.get("time")) == 5 for e in duplicates)
            if has_exact or (has_proper and not time_norm):
                continue
        daymap[d].append(
            {"time": time_norm, "title": ev["title"], "notes": ev["notes"]})

    def sort_key(e):
        """
        Sorting helper: events are ordered by:
        - Holidays first (priority), then staff shifts, then activities/fixed.
        - Within each group, events with times come before those without, ordered by time.
        """
        t = e.get("time")
        if not t:
            # Put events without times at the end of their group (use 23:59).
            return dt.time(23, 59)
        try:
            h, m = map(int, t.split(":"))
            return dt.time(h, m)
        except:
            return dt.time(23, 59)

    # Sort events on each day using the custom priority and time sorting.
    for d in daymap:
        daymap[d].sort(key=lambda e: (
            0 if e["notes"] == "Holiday" else
            1 if e["notes"] == "staff shift" else
            2, sort_key(e)
        ))
    return daymap

# -------------------------
# PDF drawing function
# -------------------------
def draw_calendar_pdf(title, disclaimer, year, month, cell_texts,
                      background_bytes=None):
    """
    Create an A3 landscape PDF with the calendar grid and text rendered.

    Parameters:
    - title: top month/year title
    - disclaimer: small text under title
    - year, month: which month to draw
    - cell_texts: dict mapping datetime.date -> string with lines separated by \n
      (these are the user's edited strings for each day)
    - background_bytes: optional image bytes to draw under everything

    Returns:
    - BytesIO buffer pointing to the generated PDF (ready to be saved/downloaded)
    """
    buffer = BytesIO()
    # Create a ReportLab canvas with A3 landscape dimensions.
    c = canvas.Canvas(buffer, pagesize=landscape(A3))
    width, height = landscape(A3)

    # --------------------------
    # Background image (optional)
    # --------------------------
    if background_bytes:
        try:
            # ImageReader can accept a BytesIO to load the image.
            img = ImageReader(BytesIO(background_bytes))
            # Draw the background stretched to cover the entire page.
            c.drawImage(img, 0, 0, width=width, height=height,
                        preserveAspectRatio=False, mask="auto")
        except Exception as e:
            # If background fails to load, warn but continue. This keeps the PDF generation robust.
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
    # Measure the title width to size the pill background.
    title_width = c.stringWidth(title_text, title_font, title_size)

    # Pill dimensions and positioning — tuned in millimetres for A3.
    side_padding = 15 * mm
    vertical_padding = 4 * mm
    pill_w = title_width + side_padding
    pill_h = 4 * mm + vertical_padding
    pill_y = height - 10 * mm  # y-position a little down from the top of the page
    pill_x = (width - pill_w) / 2  # horizontally centered

    # Draw the rounded black pill and the white title text on top.
    month_pill_color = Color(0, 0, 0)
    c.setFillColor(month_pill_color)
    c.roundRect(pill_x, pill_y, pill_w, pill_h, pill_h / 2, fill=1, stroke=0)

    # Draw month text (white), vertically centered inside the pill.
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

    # Draw the disclaimer pill and white text (same styling).
    disc_pill_color = Color(0, 0, 0)
    c.setFillColor(disc_pill_color)
    c.roundRect(disc_x, disc_y, disc_w, disc_h, disc_h / 2, fill=1, stroke=0)

    c.setFillColor(white)
    disc_text_y = disc_y + (disc_h / 2) - (disclaimer_size / 3)
    c.drawCentredString(width / 2, disc_text_y, disclaimer_text)

    # --------------------------
    # Layout variables for the calendar grid
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

    # Small gap between the weekday bar and the top of the calendar grid.
    # Adjust `bar_gap` to control vertical spacing if needed.
    bar_gap = 1.5 * mm  # reduce or increase to adjust spacing (try 3–6mm)
    top_of_grid = bar_y - bar_gap

    # Compute grid height and row height for the 5 rows of weeks.
    grid_h = top_of_grid - bottom
    row_h = grid_h / rows

    # --------------------------
    # Calendar cells drawing
    # --------------------------
    # Define a near-white background used for the cells so text is readable.
    cream = Color(1, 1, 1, alpha=0.93)
    # Colour used for staff lines where we display staff names in italic.
    staff_blue = Color(0, 0.298, 0.6)
    # Get the matrix of weeks for this month (list of weeks with day numbers)
    month_days = calendar.monthcalendar(year, month)

    # Loop through the weeks and days produced by calendar.monthcalendar
    for r_idx, week in enumerate(month_days):
        for c_idx, day in enumerate(week):
            if day == 0:
                # Zero means the slot is outside the current month (leading/trailing blank).
                continue

            d = dt.date(year, month, day)
            x = left + c_idx * col_w
            # Calculate the bottom-left corner y of the cell based on row index.
            y = bottom + (rows - 1 - r_idx) * row_h

            # Draw cell background (rounded rectangle) with a thin border.
            c.setFillColor(cream)
            c.setStrokeColor(black)
            c.roundRect(x, y, col_w, row_h, 5, fill=1, stroke=1)

            # --- Date (top-right, bold)
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(black)

            # Convert day number to string; measure width to place it snugly in the corner.
            day_str = str(day)
            day_width = c.stringWidth(day_str, "Helvetica-Bold", 12)

            # Place the date number a few millimetres from the top-right corner of the cell.
            c.drawString(x + col_w - day_width - 1.2 * mm,
                         y + row_h - 4.5 * mm,
                         day_str)

            # --- Prepare text inside the cell
            # Get the pre-composed text for this date (edited by the user), or empty string.
            lines = cell_texts.get(d, "").split("\n")
            # Start a bit below the top of the cell for the content.
            text_y = y + row_h - 3.5 * mm
            # Spacing between lines - measured in mm for readability.
            line_spacing = 4 * mm  # more readable spacing

            # Each line may represent a holiday (full uppercase), staff line, time+activity, or normal text.
            for line in lines:
                line = clean_text(line).strip()
                if not line:
                    continue

                # 🧩 Handle each line separately

                # If the line is uppercase, treat it as a holiday and use bold small caps style.
                if line.isupper():
                    # For holidays we do a special wrapping algorithm and underline each wrapped line.
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
                        # Draw holiday line in a small bold font and underline it.
                        c.setFont("Helvetica-Bold", 8.7)
                        c.setFillColor(black)
                        c.drawString(x + 2 * mm, text_y, wh)

                        # Draw a small underline beneath the holiday text.
                        text_width = c.stringWidth(wh, "Helvetica-Bold", 8.7)
                        underline_y = text_y - 0.5 * mm
                        c.line(x + 2 * mm, underline_y,
                               x + 2 * mm + text_width, underline_y)
                        text_y -= line_spacing
                    # Move on to the next logical line (skip the "normal" wrapping below).
                    continue

                # 🔹 For non-holiday lines, use smart wrapping that preserves times and parentheses.
                c.setFont("Helvetica-Bold", 10.5)
                max_text_width = col_w - (day_width + 0.5 * mm)

                # Handle staff lines specially. The convention used in the app is "Staff: NAME".
                # We render these lines in italic and in `staff_blue` color.
                if line.lower().startswith("staff:"):
                    c.setFont("Helvetica-Oblique", 10.5)
                    c.setFillColor(staff_blue)
                    c.drawString(x + 2 * mm, text_y, line)
                    # Slightly reduce the vertical space used for staff lines to fit more.
                    text_y -= line_spacing - 1
                    continue

                # Handle activity lines that begin with a time, e.g. "11:00 Coffee & Chat".
                # This regex captures a time (with optional am/pm) and the rest of the text.
                time_match = re.match(
                    r"^(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?)\s?(.*)", line)
                if time_match:
                    time_part, rest = time_match.groups()
                    rest = rest.strip()

                    # Draw the time part first at the left margin.
                    c.setFont("Helvetica-Bold", 10.5)
                    c.setFillColor(black)
                    c.drawString(x + 2 * mm, text_y, time_part)

                    # Now wrap the remaining text to fit the remainder of the cell row.
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

                    # Draw the wrapped lines — the first is positioned after the time,
                    # subsequent lines are indented to the left margin.
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

                    # Move cursor down to the next line after finishing the wrapped block.
                    text_y -= line_spacing
                    if text_y < y + 4 * mm:
                        # If we run out of space within the cell, break out early.
                        break
                    continue

                # All other normal text lines: wrap safely across multiple lines.
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

                # Draw the wrapped lines one by one.
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

                    # 🔹 Staff (italic, blue) — a second check because some lines could
                    #            be split such that "Staff:" appears inside wrapped text.
                    if subline.lower().startswith("staff:"):
                        c.setFont("Helvetica-Oblique", 10.5)
                        c.setFillColor(staff_blue)
                        c.drawString(x + 2 * mm, text_y, subline)
                        text_y -= line_spacing - 1
                        continue

                    # 🔹 Activities (bold time, normal text) — if this subline begins
                    #            with a time we show the time and the rest clearly.
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

    # Save the canvas into the buffer and make it available to the caller.
    c.save()
    buffer.seek(0)
    return buffer

# -------------------------
# Streamlit UI
# -------------------------
# Page title shown at the top of the Streamlit app
st.title("🏡 Care Home Monthly Activities — Editable Preview & A3 PDF")

# Two-column layout for the main inputs:
# left column: year and month selection
# right column: calendar title and disclaimer text
col1, col2 = st.columns(2)
with col1:
    # Year selection: number_input that limits to a sensible range (2024-2035).
    year = st.number_input("Year", 2024, 2035, dt.date.today().year)
    # Month selection: show month names instead of numbers using format_func.
    month = st.selectbox("Month", range(1, 13),
                         index=dt.date.today().month - 1,
                         format_func=lambda x: calendar.month_name[x])
with col2:
    # Title and disclaimer inputs allow users to customise the generated PDF text.
    title = st.text_input("Calendar Title",
                          f"{calendar.month_name[month]} {year}")
    disclaimer = st.text_input("Disclaimer",
                               "Activities subject to change. Please confirm with staff.")

# Markdown section header for CSV upload instructions.
st.markdown("### 📋 CSV Upload Instructions")

# Provide example formats for the staff rota CSV so users know which columns are needed.
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

# Provide example formats for the activities CSV so users know expected column names.
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

# File upload widgets — the parse_csv function will convert CSVs to pandas DataFrames.
rota_df = parse_csv(st.file_uploader("📂 Upload Staff Rota CSV", type=["csv"]))
activities_df = parse_csv(
    st.file_uploader("📂 Upload Activities CSV", type=["csv"]))

# Optional background image upload for the PDF header/background.
bg_file = st.file_uploader("Background Image (optional)",
                           type=["png", "jpg", "jpeg"])

# -------------------------
# Persisted settings for default weekly/daily rules
# -------------------------
# Load saved settings into session_state if not already present.
if "settings" not in st.session_state:
    st.session_state["settings"] = load_settings()

# Provide default values (if the JSON settings file doesn't have them).
saved_weekly = st.session_state["settings"].get("weekly_rules",
                                                "Film Night:Thu:18:00\nDogs for Health:Thu:11:00\nReminiscence:Sat:18:00"
                                                )
saved_daily = st.session_state["settings"].get("daily_rules",
                                               "Morning Exercise:09:00\nNews Headlines:10:00"
                                               )

# Text areas allow the user to edit and save default weekly and daily rules.
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

# Save button writes the current text area contents into the settings JSON file.
if st.button("💾 Save Default Rules"):
    st.session_state["settings"]["weekly_rules"] = st.session_state[
        "weekly_rules_input"]
    st.session_state["settings"]["daily_rules"] = st.session_state[
        "daily_rules_input"]
    save_settings(st.session_state["settings"])
    st.success("✅ Default rules saved successfully!")

# Parse the fixed weekly rules text into a list of dicts:
# Expected format per line: Title:Day:Time  (e.g. Film Night:Thu:18:00)
rules = []
for line in fixed_rules_text.splitlines():
    parts = [p.strip() for p in line.split(":")]
    if len(parts) >= 2:
        # Keep only first 3 chars of day (Thu -> thu), to match weekday index.
        day = parts[1][:3].lower()
        time = parts[2] if len(parts) > 2 else ""
        title_txt = parts[0]
        weekday = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].index(day)
        rules.append({"weekday": weekday, "time": time, "title": title_txt})

# Parse daily rules. We split on the first colon so "Morning Exercise:09:00" keeps time intact.
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

# Checkbox to include holidays. If checked, the UI below allows selecting which.
include_holidays = st.checkbox("Include UK National Holidays", True)

# -------------------------
# Holiday selection system (persistent selections)
# -------------------------
if include_holidays:
    st.markdown("### 🗓️ Select Holidays to Include")

    # Build a dict mapping date -> list of holiday names for the chosen month.
    holidays_by_day = {}
    for h in ALL_HOLIDAYS:
        try:
            d = dt.datetime.strptime(h["date"], "%Y-%m-%d").date()
        except:
            continue
        if d.year == year and d.month == month:
            holidays_by_day.setdefault(d, []).append(h["name"])

    if not holidays_by_day:
        # If the local holidays JSON has nothing for the month, show an info box.
        st.info("No holidays found for this month.")
    else:
        # Load saved selection if available. This keeps user choices between changes.
        saved_selection = set(st.session_state.get("selected_holidays", []))
        current_selection = set()

        # If there's no saved selection yet, default to selecting all holidays in this month.
        if not saved_selection:
            all_holiday_names = {hname for hlist in holidays_by_day.values()
                                 for hname in hlist}
            saved_selection = all_holiday_names

        # Some small CSS to give each day block a border in the Streamlit app.
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

        # Render the calendar grid here but only to show holiday checkboxes.
        month_days = calendar.monthcalendar(year, month)
        for week in month_days:
            cols = st.columns(7)
            for c_idx, day in enumerate(week):
                if day == 0:
                    continue
                date_obj = dt.date(year, month, day)
                day_holidays = holidays_by_day.get(date_obj, [])

                # Each day gets its own small block with checkboxes for any holidays.
                with cols[c_idx]:
                    st.markdown(
                        f"<div class='day-block'><div class='day-header'>{calendar.month_abbr[month]} {day}</div>",
                        unsafe_allow_html=True
                    )
                    if not day_holidays:
                        st.markdown("<em>No holidays</em>",
                                    unsafe_allow_html=True)
                    else:
                        for name in sorted(set(day_holidays)):
                            # Create a unique Streamlit key so checkboxes are preserved per-month/per-holiday.
                            key = f"hol_{year}-{month:02d}-{day:02d}_{name}"
                            checked = name in saved_selection
                            # If the checkbox is ticked, add to current_selection so it persists below.
                            if st.checkbox(name, value=checked, key=key):
                                current_selection.add(name)
                    st.markdown("</div>", unsafe_allow_html=True)

        # Save the updated selection back into session_state once after rendering all checkboxes.
        st.session_state["selected_holidays"] = list(current_selection)

# -------------------------
# Preview and Editable Calendar Section
# -------------------------

# Create a unique session key for each year-month combo, allowing editable content to be stored.
session_key = f"{year}-{month:02d}"

# Reset holiday selection when changing month or year to avoid cross-month leakage.
if "last_month" not in st.session_state or \
   st.session_state["last_month"] != month or \
   st.session_state["last_year"] != year:
    st.session_state["selected_holidays"] = []
    st.session_state["last_month"] = month
    st.session_state["last_year"] = year

# When the user clicks "Preview Calendar", build the internal daymap and store editable text in state.
if st.button("Preview Calendar"):
    with st.spinner("Generating preview..."):
        daymap = seat_activity_into_calendar(
            year, month, activities_df, rota_df, rules, include_holidays,
            daily_rules
        )
        # Save an editable copy into session_state for the chosen month.
        st.session_state[session_key] = {}

        # Convert the daymap items into newline-separated strings for each day.
        for d, events in daymap.items():
            lines = []
            for ev in events:
                if ev["notes"] == "Holiday":
                    # Holidays are displayed uppercase for visual priority.
                    lines.append(ev["title"].upper())
                elif ev["notes"] == "staff shift":
                    # Staff lines are prefixed for clarity.
                    lines.append(f"Staff: {ev['title']}")
                elif ev["notes"] in ("fixed", "fixed daily", "activity"):
                    t = ev.get("time", "")
                    lines.append(f"{t} {ev['title']}".strip())
            # Store the editable string for this date in session_state.
            st.session_state[session_key][d] = "\n".join(lines)

# Editable preview UI — only appears if the preview has been generated and saved in session_state.
if session_key in st.session_state:
    st.subheader(
        f"📝 Edit Calendar for {calendar.month_name[month]} {year} Before Generating PDF")
    month_days = calendar.monthcalendar(year, month)

    # Display the calendar as 7 columns (Mon-Sun), each cell contains a text_area for editing.
    for week in month_days:
        cols = st.columns(7)
        for c_idx, day in enumerate(week):
            if day == 0:
                with cols[c_idx]:
                    # For blank cells, render an empty placeholder to preserve layout.
                    st.markdown(" ")
                continue
            d = dt.date(year, month, day)
            with cols[c_idx]:
                # Each text_area uses a unique key combining session_key and the date.
                # The default value is the previously stored value (or empty string).
                st.text_area(
                    f"{day}",
                    st.session_state[session_key].get(d, ""),
                    key=f"{session_key}_{d}",
                    height=180,
                )

    # Optional reset button for this month's edits: removes the stored preview and reruns.
    if st.button("🔄 Reset This Month's Edits"):
        st.session_state.pop(session_key, None)
        st.rerun()

    # Generate PDF: read back the edited fields from session_state and produce the PDF to download.
    if st.button("Generate PDF"):
        bg_bytes = bg_file.read() if bg_file else None

        # Gather edited text areas for this month only.
        # Keys follow the pattern f"{session_key}_{date}" so we filter by that prefix.
        edited_texts = {
            dt.date.fromisoformat(k.split("_")[-1]): v
            for k, v in st.session_state.items()
            if k.startswith(session_key + "_")
        }

        # Create the PDF buffer using the draw_calendar_pdf function.
        pdf_buf = draw_calendar_pdf(
            title, disclaimer, year, month, edited_texts,
            background_bytes=bg_bytes
        )

        st.success("✅ A3 PDF calendar generated successfully!")
        # Provide a download button for the generated PDF buffer.
        st.download_button(
            "📥 Download Calendar (A3 Landscape PDF)",
            data=pdf_buf,
            file_name=f"calendar_{year}_{month:02d}_A3.pdf",
            mime="application/pdf",
        )

    if st.button("📅 Generate Weekly PDFs (A4 Landscape)"):

        with st.spinner("Generating weekly PDFs..."):
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.pdfgen import canvas
            from reportlab.lib.colors import black
            from reportlab.lib.units import mm
            import PyPDF2


            # Helper to add ordinal suffix (1st, 2nd, 3rd, 4th...)
            def ordinal(n):
                if 10 <= n % 100 <= 20:
                    suffix = "th"
                else:
                    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
                return f"{n}{suffix}"


            # Prepare list of days for the month
            first_day, last_day = month_date_range(year, month)
            days = [first_day + dt.timedelta(days=i)
                    for i in range((last_day - first_day).days + 1)]

            # Split into chunks of 7 (weeks)
            weeks = [days[i:i + 7] for i in range(0, len(days), 7)]

            pdf_buffers = []  # collect each week’s PDF bytes

            for week_days in weeks:
                buf = BytesIO()
                c = canvas.Canvas(buf, pagesize=landscape(A4))
                width, height = landscape(A4)

                # For each day in the week, create a page
                for d in week_days:
                    # 🔹 Draw large centered day heading (date)
                    c.setFont("Helvetica-Bold", 26)
                    c.setFillColor(black)
                    day_str = f"{calendar.day_name[d.weekday()]} {ordinal(d.day)} {calendar.month_name[d.month]}"
                    day_width = c.stringWidth(day_str, "Helvetica-Bold", 26)
                    day_x = (width - day_width) / 2
                    day_y = height - 14 * mm
                    c.drawString(day_x, day_y, day_str)

                    # 🔹 Draw standard disclaimer under the date — centered & wrapped
                    c.setFont("Helvetica-Oblique", 12)
                    disclaimer_text = (
                        "Activities may change due to unforeseen circumstances. "
                        "Families are welcome to join. "
                        "Weather permitting, activities may move outdoors."
                    )

                    # Wrap disclaimer text to fit within centered margins
                    max_text_width = width - 80 * mm  # generous side margins (40 mm each)
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

                    # Draw each line centered below the date with extra vertical spacing
                    line_spacing = 6 * mm
                    text_y = day_y - 10 * mm  # significant margin between date and disclaimer
                    for line in wrapped_lines:
                        line_width = c.stringWidth(line, "Helvetica-BoldOblique",
                                                   12)
                        c.drawString((width - line_width) / 2, text_y, line)
                        text_y -= line_spacing

                    # 🔹 Draw events for the day — bold and spaced further below disclaimer
                    margin_below_disclaimer = 5 * mm  # increased space for clarity
                    y = text_y - margin_below_disclaimer

                    # Retrieve day’s content from session_state
                    text = st.session_state.get(f"{session_key}_{d}",
                                                "").strip()
                    if not text:
                        text = "(No activities planned)"

                    # Combine staff lines into a single formatted line
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

                    # Draw staff line first (if any)
                    if staff_lines:
                        combined_staff = " - ".join(staff_lines)
                        x_start = 25 * mm
                        max_width = width - (2 * x_start)

                        # Wrap combined staff line if it's too long for one line
                        words = combined_staff.split()
                        current_line = ""
                        wrapped_lines = []
                        for word in words:
                            test_line = (current_line + " " + word).strip()
                            if c.stringWidth(test_line, "Helvetica-Oblique",
                                             15) > max_width and current_line:
                                wrapped_lines.append(current_line)
                                current_line = word
                            else:
                                current_line = test_line
                        if current_line:
                            wrapped_lines.append(current_line)

                        # Staff color (same as monthly)
                        staff_blue = Color(0, 0.298, 0.6)
                        c.setFont("Helvetica-Oblique", 15)
                        c.setFillColor(staff_blue)

                        for wrapped in wrapped_lines:
                            wrapped = wrapped.strip()
                            c.drawString(x_start, y, wrapped)
                            y -= 9 * mm  # spacing between wrapped lines

                        y -= 5 * mm  # extra gap after staff section

                    # Draw remaining non-staff events normally
                    for line in other_lines:
                        c.setFont("Helvetica-Bold", 15)
                        c.setFillColor(black)
                        x_start = 25 * mm
                        max_width = width - (2 * x_start)
                        words = line.split()
                        current_line = ""
                        wrapped_lines = []
                        for word in words:
                            test_line = (current_line + " " + word).strip()
                            if c.stringWidth(test_line, "Helvetica-Bold",
                                             15) > max_width and current_line:
                                wrapped_lines.append(current_line)
                                current_line = word
                            else:
                                current_line = test_line
                        if current_line:
                            wrapped_lines.append(current_line)
                        for wrapped in wrapped_lines:
                            c.drawString(x_start, y, wrapped.strip())
                            y -= 9 * mm
                        y -= 5 * mm

                        # Start new page if running out of space
                        if y < 25 * mm:
                            c.showPage()
                            y = height - 40 * mm

                    c.showPage()  # move to next day

                c.save()
                buf.seek(0)
                pdf_buffers.append(buf.getvalue())

            # 🔹 Combine all weeks into one PDF
            merger = PyPDF2.PdfMerger()
            for pdf in pdf_buffers:
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
