"""
Note Taking Assistant AI - demo server

Features:
  1. summary    -> when the Summarize button is pressed
  2. chart      -> automatically on pause, if the newest writing is a data set
                   (table, list, % breakdown) -> line / bar / pie chart
  3. answer     -> on pause, if the newest writing is an unsolved math question
  4. correction -> on pause, if the newest writing contains a wrong fact or wrong math
  + underline   -> the AI also returns WHERE the key/wrong word is, so the page can underline it

Run (Windows PowerShell):
    python -m pip install flask anthropic
    $env:ANTHROPIC_API_KEY="your-api-key-here"
    python server.py
Open http://localhost:5000
(Never paste your real key into this file.)
"""
import json
import re

import anthropic
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from the environment

MODEL = "claude-sonnet-5-5"             # use "claude-haiku-5-5" while testing to save credit

# ---------- Prompts ----------
PAUSE_PROMPT = """This image shows a student's handwritten notes, written live.
Look at the MOST RECENTLY written line(s) (usually the lowest text on the page) and decide which ONE case applies:

1. "data"       - it is a data set that would be clearer as a chart: a table, a list of
                  label/value pairs, or a percentage breakdown / distribution
                  (e.g. "HW 20%, attendance 10%, quiz 10%, midterm 30%, final 30%",
                  "Jan 20, Feb 35, Mar 30"). Needs at least 3 values.
2. "answer"     - it is a math question or problem that is NOT solved yet
                  (e.g. "2x + 3 = 7, x = ?", "integral of x^2", "15% of 80").
                  Give the answer with very short steps.
3. "correction" - it contains a mistake: a wrong math step/result, or a wrong fact
                  (e.g. "x = 5" when x = 2, "mitochondria is not the powerhouse of the cell").
                  Say what is wrong and give the correct version.
4. "ok"         - it is correct and needs nothing.
5. "none"       - unreadable, a doodle, or too little to judge.

Also locate the KEY part in the image:
- for "correction": the exact wrong word(s) or number (e.g. "not", "x = 5")
- for "answer": the question itself (e.g. "x = ?")
Give its position as a bounding box in FRACTIONS of the image size (0 to 1):
[left, top, right, bottom], where [0,0] is the top-left corner of the image.
Make the box tight around just that handwriting.

Reply with ONLY a JSON object, no other text:
{"type": "data" | "answer" | "correction" | "ok" | "none",
 "text": "<max 2 short sentences>",
 "keyword": "<the key word(s) as written, or empty>",
 "box": [left, top, right, bottom] or null}
For "data", "ok" and "none" use box null. For "ok" use text "Looks correct." For "none" use text "".
"""

CHART_PROMPT = """This image is a page of handwritten notes.
Find data in it that can be plotted (if there are several, use the MOST RECENTLY written one,
usually the lowest on the page), such as:
- a percentage breakdown / distribution / parts of a whole
  (e.g. grading: HW 20%, quiz 10%, midterm 30%, final 30%; budget split)
- a table or list of numbers (e.g. time vs distance, month vs sales)
- a function or equation (e.g. y = x^2, y = 2x + 1): compute 7-11 evenly spaced sample points yourself

Choose the chart type:
- "pie"  for parts of a whole (percentages / shares that add up to a total)
- "line" for trends over time, or functions
- "bar"  for comparing separate categories

If percentages in a breakdown do not add up to 100, still plot them, and mention it in "note".

Reply with ONLY a JSON object, no other text:
{"found": true,
 "chart_type": "pie" | "line" | "bar",
 "title": "<short title>",
 "x_label": "<x axis label, empty for pie>",
 "y_label": "<y axis label, empty for pie>",
 "labels": [<category / x values as strings>],
 "datasets": [{"label": "<series name>", "data": [<numbers>]}],
 "unit": "<unit to show after each value, e.g. '%', 'kg', '$', or '' if none>",
 "note": "<1 short sentence describing the data>"}
Each dataset's data must have the same length as labels. A pie chart has exactly one dataset.
If there is nothing plottable, reply {"found": false, "note": "<short reason>"}.
"""

SUMMARY_PROMPT = """This image is a page of handwritten notes.
Summarize it as 3-5 short bullet points (use "• ").
If you notice a mistake, write the corrected version and mark it with ⚠.
Plain text only."""


# ---------- Helpers ----------
def call_claude(img_b64: str, prompt: str, max_tokens: int) -> str:
    reply = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return reply.content[0].text.strip()


def extract_json(text: str) -> dict:
    """Pull the JSON object out of the model's reply, even if it adds extra text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def parse_pause(text: str) -> dict:
    data = extract_json(text)
    if data.get("type") not in {"data", "answer", "correction", "ok", "none"}:
        return {"type": "ok", "text": text, "keyword": "", "box": None}
    return data


def clean_box(box):
    """Return [l, t, r, b] with values clamped to 0..1, or None if invalid."""
    try:
        l, t, r, b = [min(max(float(v), 0.0), 1.0) for v in box]
    except (TypeError, ValueError):
        return None
    if r <= l or b <= t:
        return None
    return [l, t, r, b]


def make_chart(img_b64: str):
    """Ask Claude for chart data. Returns a JSON response for the page."""
    chart = extract_json(call_claude(img_b64, CHART_PROMPT, max_tokens=800))
    if not chart.get("found") or not chart.get("labels") or not chart.get("datasets"):
        note = chart.get("note") or "No plottable data found in the notes."
        return jsonify(type="ok", text=note, keyword="", box=None)
    if chart.get("chart_type") not in {"pie", "line", "bar"}:
        chart["chart_type"] = "bar"
    return jsonify(type="chart", text=chart.get("note", ""), chart=chart,
                   keyword="", box=None)


# ---------- Routes ----------
@app.get("/")
def index():
    return send_file("index.html")


@app.post("/analyze")
def analyze():
    data = request.get_json()
    mode = data.get("mode", "feedback")
    img_b64 = data["image"].split(",", 1)[1]          # strip "data:image/png;base64,"

    try:
        if mode == "summary":
            text = call_claude(img_b64, SUMMARY_PROMPT, max_tokens=400)
            return jsonify(type="summary", text=text, keyword="", box=None)

        if mode == "chart":
            return make_chart(img_b64)

        # ---- pause: decide what kind of feedback ----
        result = parse_pause(call_claude(img_b64, PAUSE_PROMPT, max_tokens=250))

        if result["type"] == "data":              # detected a data set → auto chart
            return make_chart(img_b64)

        return jsonify(
            type=result["type"],
            text=result.get("text", ""),
            keyword=result.get("keyword", "") or "",
            box=clean_box(result.get("box")),
        )

    except Exception as e:
        return jsonify(type="error", text=str(e), keyword="", box=None), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)