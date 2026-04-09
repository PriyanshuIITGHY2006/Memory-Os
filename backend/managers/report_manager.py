"""
MemoryOS — "Your Year in Memory" Annual Report Generator
=========================================================
Produces a beautiful, typeset A4 PDF using reportlab (pure Python, no system deps).

Layout:
  Page 1  — Cover          (dark navy, large year, user name)
  Page 2  — Year in Numbers (stat cards grid)
  Page 3  — The People in Your World
  Page 4  — Key Moments    (event timeline)
  Page 5  — What You've Learned (knowledge nodes)
  Page 6  — Know Yourself  (profile + preferences)
  Page 7  — Closing page   (dark, branding)
"""

import io
import textwrap
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas as rl_canvas

# ── Page geometry ──────────────────────────────────────────────────────────────
W, H    = A4          # 595.27 × 841.89 pt
MX      = 52          # horizontal margin
MY      = 52          # vertical margin
CW      = W - 2 * MX  # content width ≈ 491

# ── Brand palette ──────────────────────────────────────────────────────────────
NAVY    = colors.HexColor('#080E1E')
NAVY2   = colors.HexColor('#101828')
BLUE    = colors.HexColor('#2563EB')
BLUE_L  = colors.HexColor('#3B82F6')
BLUE_XL = colors.HexColor('#93C5FD')
TEAL    = colors.HexColor('#0891B2')
PURPLE  = colors.HexColor('#7C3AED')
GREEN   = colors.HexColor('#16A34A')
AMBER   = colors.HexColor('#D97706')
RED_C   = colors.HexColor('#DC2626')
WHITE   = colors.white
OFFWHT  = colors.HexColor('#F8FAFC')
SLATE   = colors.HexColor('#CBD5E1')
GRAY    = colors.HexColor('#94A3B8')
GRAY_D  = colors.HexColor('#64748B')
LINE    = colors.HexColor('#E2E8F0')
INK     = colors.HexColor('#0F172A')
INK_2   = colors.HexColor('#1E293B')

# ── Type scale ─────────────────────────────────────────────────────────────────
FONT_REGULAR = "Helvetica"
FONT_BOLD    = "Helvetica-Bold"
FONT_OBLIQUE = "Helvetica-Oblique"


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _rect(c, x, y, w, h, fill_color, stroke_color=None, radius=0):
    c.saveState()
    c.setFillColor(fill_color)
    if stroke_color:
        c.setStrokeColor(stroke_color)
        c.setLineWidth(0.5)
    else:
        c.setStrokeColor(colors.transparent)
    if radius:
        c.roundRect(x, y, w, h, radius, fill=1, stroke=1 if stroke_color else 0)
    else:
        c.rect(x, y, w, h, fill=1, stroke=1 if stroke_color else 0)
    c.restoreState()


def _text(c, x, y, txt, font=FONT_REGULAR, size=10, color=INK, align="left"):
    c.saveState()
    c.setFont(font, size)
    c.setFillColor(color)
    if align == "center":
        c.drawCentredString(x, y, txt)
    elif align == "right":
        c.drawRightString(x, y, txt)
    else:
        c.drawString(x, y, txt)
    c.restoreState()


def _line(c, x1, y1, x2, y2, color=LINE, width=0.5):
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    c.restoreState()


def _page_num(c, page: int):
    _text(c, W / 2, 22, f"— {page} —", FONT_REGULAR, 8, GRAY, "center")


def _section_header(c, y: float, title: str, subtitle: str = "") -> float:
    """Draw a consistent section heading. Returns y after the header."""
    _line(c, MX, y, MX + 36, y, BLUE, 2)
    y -= 14
    _text(c, MX, y, title.upper(), FONT_BOLD, 22, INK)
    if subtitle:
        y -= 18
        _text(c, MX, y, subtitle, FONT_REGULAR, 11, GRAY_D)
    return y - 28


def _wrap(text: str, max_chars: int = 72) -> list[str]:
    return textwrap.wrap(text, max_chars) or [""]


# ── Pages ──────────────────────────────────────────────────────────────────────

def _page_cover(c: rl_canvas.Canvas, user_name: str, stats: dict, year: int):
    # Full dark background
    _rect(c, 0, 0, W, H, NAVY)

    # Subtle decorative grid lines
    c.saveState()
    c.setStrokeColor(colors.HexColor('#1A2744'))
    c.setLineWidth(0.4)
    for i in range(0, int(W), 40):
        c.line(i, 0, i, H)
    for j in range(0, int(H), 40):
        c.line(0, j, W, j)
    c.restoreState()

    # Blue accent bar — top
    _rect(c, 0, H - 6, W, 6, BLUE)

    # "MemoryOS" wordmark top-left
    _text(c, MX, H - 38, "MemoryOS", FONT_BOLD, 13, BLUE_L)

    # Oversized ghost year behind — decorative
    c.saveState()
    c.setFont(FONT_BOLD, 220)
    c.setFillColor(colors.HexColor('#111E3A'))
    c.drawCentredString(W / 2, H / 2 - 80, str(year))
    c.restoreState()

    # "YOUR YEAR IN MEMORY"
    _text(c, W / 2, H - 130, "Y O U R   Y E A R   I N   M E M O R Y", FONT_BOLD, 9, BLUE_L, "center")

    # Large year — foreground
    _text(c, W / 2, H / 2 + 20, str(year), FONT_BOLD, 96, WHITE, "center")

    # Horizontal rule
    _rect(c, MX, H / 2 - 18, CW, 1.5, BLUE)

    # User name
    _text(c, W / 2, H / 2 - 44, user_name, FONT_BOLD, 26, WHITE, "center")

    # Stats summary line at bottom
    turns = stats.get("total_turns", 0)
    ents  = stats.get("entity_count", 0)
    evts  = stats.get("event_count", 0)
    summary = f"{turns} conversations  ·  {ents} entities tracked  ·  {evts} events logged"
    _text(c, W / 2, MY + 40, summary, FONT_REGULAR, 9, GRAY, "center")

    # "Generated by MemoryOS" bottom
    _text(c, W / 2, MY + 20, f"Generated {datetime.now().strftime('%B %d, %Y')}", FONT_OBLIQUE, 8, GRAY_D, "center")

    # Blue accent bar — bottom
    _rect(c, 0, 0, W, 5, BLUE)

    c.showPage()


def _page_numbers(c: rl_canvas.Canvas, stats: dict):
    _rect(c, 0, 0, W, H, OFFWHT)
    _rect(c, 0, H - 5, W, 5, BLUE)

    y = H - MY
    y = _section_header(c, y, "A Year in Numbers", "Everything your memory has collected")

    cards = [
        (stats.get("total_turns", 0),      "Conversations",       BLUE,   "Every message, every thought captured."),
        (stats.get("entity_count", 0),      "Entities Tracked",    PURPLE, "People, places, and organizations remembered."),
        (stats.get("event_count", 0),       "Events Logged",       RED_C,  "Moments that shaped your story."),
        (stats.get("knowledge_count", 0),   "Knowledge Nodes",     TEAL,   "Facts and insights added to your graph."),
        (stats.get("people", 0),            "People Known",        GREEN,  "Individuals in your network."),
        (stats.get("pref_count", 0),        "Preferences Stored",  AMBER,  "Your habits, goals, and constraints."),
    ]

    card_w = (CW - 16) / 2
    card_h = 110
    gap    = 16
    start_y = y

    for i, (val, label, color, desc) in enumerate(cards):
        col  = i % 2
        row  = i // 2
        cx   = MX + col * (card_w + gap)
        cy   = start_y - row * (card_h + gap) - card_h

        _rect(c, cx, cy, card_w, card_h, WHITE, LINE, 8)

        # Color top strip
        _rect(c, cx, cy + card_h - 5, card_w, 5, color, radius=0)

        # Number
        _text(c, cx + 20, cy + card_h - 52, str(val), FONT_BOLD, 36, color)

        # Label
        _text(c, cx + 20, cy + card_h - 70, label, FONT_BOLD, 11, INK)

        # Description
        _text(c, cx + 20, cy + 18, desc, FONT_REGULAR, 8, GRAY_D)

    _page_num(c, 2)
    c.showPage()


def _page_people(c: rl_canvas.Canvas, nodes: list):
    _rect(c, 0, 0, W, H, WHITE)
    _rect(c, 0, H - 5, W, 5, PURPLE)

    people  = [n for n in nodes if n.get("type") == "Person"]
    places  = [n for n in nodes if n.get("type") == "Place"]
    orgs    = [n for n in nodes if n.get("type") == "Organization"]

    y = H - MY
    y = _section_header(c, y, "The People in Your World",
                         f"{len(people)} people · {len(places)} places · {len(orgs)} organizations")

    def _entity_block(c, nodes_list, label, dot_color, y_start) -> float:
        if not nodes_list:
            return y_start
        _text(c, MX, y_start, label.upper(), FONT_BOLD, 7, dot_color)
        y_start -= 6
        _line(c, MX, y_start, MX + CW, y_start, dot_color, 0.4)
        y_start -= 14

        for n in nodes_list[:14]:
            if y_start < MY + 20:
                break
            name = n.get("name") or n.get("label") or "Unknown"
            rel  = (n.get("relationship") or "").strip()

            # Dot
            c.saveState()
            c.setFillColor(dot_color)
            c.circle(MX + 5, y_start + 3.5, 3.5, fill=1, stroke=0)
            c.restoreState()

            _text(c, MX + 16, y_start, name, FONT_BOLD, 11, INK)
            if rel:
                _text(c, MX + 16, y_start - 12, rel, FONT_REGULAR, 9, GRAY_D)

            # Attributes (occupation, company, etc.)
            attrs = n.get("attributes", {})
            attr_bits = [f"{k}: {v}" for k, v in attrs.items()
                         if k not in {"user_id", "id"} and v][:3]
            if attr_bits:
                attr_str = "  ·  ".join(attr_bits)
                for wrapped in _wrap(attr_str, 70)[:1]:
                    _text(c, MX + 16, y_start - (24 if rel else 12), wrapped, FONT_REGULAR, 8, GRAY)

            y_start -= (42 if (rel or attr_bits) else 22)

        return y_start - 10

    y = _entity_block(c, people, "People", PURPLE, y)
    y = _entity_block(c, orgs,   "Organizations", AMBER, y - 4)
    y = _entity_block(c, places, "Places", GREEN, y - 4)

    _page_num(c, 3)
    c.showPage()


def _page_events(c: rl_canvas.Canvas, timeline: list):
    events = [t for t in timeline if t.get("type") == "event"]

    _rect(c, 0, 0, W, H, OFFWHT)
    _rect(c, 0, H - 5, W, 5, RED_C)

    y = H - MY
    y = _section_header(c, y, "Key Moments", f"{len(events)} events recorded across your timeline")

    if not events:
        _text(c, MX, y - 20, "No events have been logged yet.", FONT_OBLIQUE, 11, GRAY)
        _page_num(c, 4)
        c.showPage()
        return

    # Vertical timeline line
    line_x = MX + 24
    _line(c, line_x, MY + 20, line_x, y - 10, LINE, 1.5)

    for ev in events[:18]:
        if y < MY + 50:
            break

        turn  = ev.get("turn") or 0
        title = ev.get("title") or ev.get("description") or ""
        date  = ev.get("date") or f"Turn {turn}"

        # Circle on timeline
        c.saveState()
        c.setFillColor(RED_C)
        c.setStrokeColor(OFFWHT)
        c.setLineWidth(2)
        c.circle(line_x, y, 5, fill=1, stroke=1)
        c.restoreState()

        # Turn / date tag
        tag_w = 52
        _rect(c, MX + 34, y - 7, tag_w, 16, RED_C, radius=4)
        _text(c, MX + 34 + tag_w / 2, y - 2, str(date)[:12], FONT_BOLD, 7, WHITE, "center")

        # Event title
        title_x = MX + 34 + tag_w + 10
        wrapped = _wrap(title, 54)
        _text(c, title_x, y + 2, wrapped[0][:72], FONT_BOLD, 11, INK)
        if len(wrapped) > 1:
            _text(c, title_x, y - 10, wrapped[1][:72], FONT_REGULAR, 10, GRAY_D)

        y -= 38

    _page_num(c, 4)
    c.showPage()


def _page_knowledge(c: rl_canvas.Canvas, nodes: list):
    knowledge = [n for n in nodes if n.get("type") == "Knowledge"]

    _rect(c, 0, 0, W, H, WHITE)
    _rect(c, 0, H - 5, W, 5, TEAL)

    y = H - MY
    y = _section_header(c, y, "What You've Learned",
                         f"{len(knowledge)} knowledge nodes in your memory graph")

    if not knowledge:
        _text(c, MX, y - 20, "No knowledge nodes recorded yet.", FONT_OBLIQUE, 11, GRAY)
        _page_num(c, 5)
        c.showPage()
        return

    card_h  = 76
    gap     = 10
    col_w   = (CW - gap) / 2

    for i, kn in enumerate(knowledge[:12]):
        if y - card_h < MY + 10:
            break

        col = i % 2
        cx  = MX + col * (col_w + gap)

        if col == 0 and i > 0:
            y -= (card_h + gap)

        cy = y - card_h

        _rect(c, cx, cy, col_w, card_h, OFFWHT, LINE, 6)

        # Teal left accent
        _rect(c, cx, cy + 4, 3, card_h - 8, TEAL, radius=2)

        topic   = kn.get("topic") or kn.get("name") or "Insight"
        content = kn.get("content") or ""

        _text(c, cx + 14, cy + card_h - 20, topic[:48], FONT_BOLD, 11, INK)

        for j, line in enumerate(_wrap(content, 48)[:3]):
            _text(c, cx + 14, cy + card_h - 34 - j * 13, line, FONT_REGULAR, 9, GRAY_D)

    _page_num(c, 5)
    c.showPage()


def _page_profile(c: rl_canvas.Canvas, stats: dict):
    _rect(c, 0, 0, W, H, OFFWHT)
    _rect(c, 0, H - 5, W, 5, AMBER)

    profile     = stats.get("profile", {})
    preferences = stats.get("preferences", [])

    y = H - MY
    y = _section_header(c, y, "Know Yourself", "Your profile, preferences, goals, and constraints")

    # Profile key-value
    if profile:
        _text(c, MX, y, "PROFILE", FONT_BOLD, 7, AMBER)
        y -= 6
        _line(c, MX, y, MX + CW, y, AMBER, 0.4)
        y -= 16

        for k, v in list(profile.items())[:12]:
            if y < MY + 120:
                break
            label = k.replace("_", " ").title()
            val   = str(v)

            # Label
            _rect(c, MX, y - 4, 100, 16, colors.HexColor('#FEF3C7'), radius=3)
            _text(c, MX + 6, y, label[:20], FONT_BOLD, 8, colors.HexColor('#92400E'))

            # Value
            for ln in _wrap(val, 56)[:2]:
                _text(c, MX + 114, y, ln, FONT_REGULAR, 10, INK)
                y -= 13
            y -= 10

        y -= 12

    # Preferences by category
    if preferences:
        cats = {}
        for p in preferences:
            cat = (p.get("category") or "preference").lower()
            cats.setdefault(cat, []).append(p.get("value", ""))

        cat_colors = {
            "preference": BLUE,
            "goal":       GREEN,
            "allergy":    RED_C,
            "constraint": PURPLE,
        }

        _text(c, MX, y, "PREFERENCES & GOALS", FONT_BOLD, 7, BLUE)
        y -= 6
        _line(c, MX, y, MX + CW, y, BLUE, 0.4)
        y -= 18

        for cat, values in cats.items():
            if y < MY + 40:
                break
            col = cat_colors.get(cat, GRAY_D)
            _text(c, MX, y, cat.upper(), FONT_BOLD, 8, col)
            y -= 14

            pill_x = MX
            for val in values[:8]:
                text  = str(val)[:40]
                tw    = len(text) * 5.6 + 16
                if pill_x + tw > MX + CW:
                    pill_x = MX
                    y -= 22
                if y < MY + 40:
                    break
                _rect(c, pill_x, y - 5, tw, 18, colors.HexColor(
                    col.hexval().replace('#', '') if hasattr(col, 'hexval') else '#EEF2FF'
                ) if False else _alpha_hex(col, 0.12), radius=9)
                _text(c, pill_x + 8, y, text, FONT_REGULAR, 9, col)
                pill_x += tw + 8

            y -= 30

    _page_num(c, 6)
    c.showPage()


def _alpha_hex(base_color, alpha: float) -> colors.Color:
    """Return a lightened/pastel version of a color by blending with white."""
    r = base_color.red   + (1 - base_color.red)   * (1 - alpha)
    g = base_color.green + (1 - base_color.green) * (1 - alpha)
    b = base_color.blue  + (1 - base_color.blue)  * (1 - alpha)
    return colors.Color(r, g, b)


def _page_closing(c: rl_canvas.Canvas, user_name: str, year: int):
    _rect(c, 0, 0, W, H, NAVY)
    _rect(c, 0, H - 5, W, 5, BLUE)
    _rect(c, 0, 0, W, 5, BLUE)

    # Large decorative quote marks
    c.saveState()
    c.setFont(FONT_BOLD, 180)
    c.setFillColor(colors.HexColor('#0D1A35'))
    c.drawString(MX - 20, H / 2 - 40, "\u201c")
    c.restoreState()

    _text(c, W / 2, H / 2 + 60, "Memory is the treasury",       FONT_BOLD,    22, WHITE,  "center")
    _text(c, W / 2, H / 2 + 32, "and guardian of all things.",  FONT_REGULAR, 22, SLATE,  "center")
    _text(c, W / 2, H / 2,      "— Cicero",                     FONT_OBLIQUE, 11, GRAY_D, "center")

    _text(c, W / 2, H / 2 - 60, f"{user_name}  ·  {year}",     FONT_REGULAR, 10, GRAY,   "center")

    _text(c, W / 2, MY + 30, "MemoryOS  —  Your personal memory operating system",
          FONT_BOLD, 9, BLUE_L, "center")
    _text(c, W / 2, MY + 14, "Keep building your story.",
          FONT_REGULAR, 8, GRAY_D, "center")

    c.showPage()


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_report(stats: dict, timeline: list, graph_nodes: list) -> bytes:
    """
    Build and return the full PDF as bytes.

    Args:
        stats:       result of GraphManager.get_stats()
        timeline:    result of GraphManager.get_timeline()
        graph_nodes: nodes list from GraphManager.get_graph_data()["nodes"]
    """
    year      = datetime.now().year
    user_name = stats.get("user_name") or stats.get("profile", {}).get("name") or "Your Story"
    buf       = io.BytesIO()
    c         = rl_canvas.Canvas(buf, pagesize=A4)

    c.setTitle(f"Your Year in Memory · {year}")
    c.setAuthor("MemoryOS")
    c.setSubject(f"Annual memory report for {user_name}")

    _page_cover(c, user_name, stats, year)
    _page_numbers(c, stats)
    _page_people(c, graph_nodes)
    _page_events(c, timeline)
    _page_knowledge(c, graph_nodes)
    _page_profile(c, stats)
    _page_closing(c, user_name, year)

    c.save()
    buf.seek(0)
    return buf.read()
