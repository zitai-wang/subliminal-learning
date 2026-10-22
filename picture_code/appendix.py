import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from textwrap import fill

# =========================
# Output
# =========================
OUT_PATH = "case_study.pdf"

# =========================
# Case data
# =========================
cases = [
    {
        "category": "State\nMaintenance\nFailure",
        "case": "Case 1",
        "title": "State update loss",
        "question": "Brian has 20 video games but lost 5 before comparison. Bobby has 5 fewer than 3 times Brian's games. How many does Bobby have?",
        "baseline": "Uses outdated state:\nBrian = 20\n3 × 20 − 5 = 55",
        "lift": "Updates state first:\nBrian = 20 − 5 = 15\n3 × 15 − 5 = 40",
        "answer_bad": "55",
        "answer_good": "40",
        "takeaway": "Repairs outdated-state reuse."
    },
    {
        "category": "State\nMaintenance\nFailure",
        "case": "Case 2",
        "title": "State transition drift",
        "question": "18 pink flamingos are placed. One third are painted white, then 18 pink flamingos are added. How many more pink than white?",
        "baseline": "Forgets updated pink count:\nPink = 18 + 18 = 36\nWhite = 6\n36 − 6 = 30",
        "lift": "Maintains updated state:\nPink = 12 + 18 = 30\nWhite = 6\n30 − 6 = 24",
        "answer_bad": "30",
        "answer_good": "24",
        "takeaway": "Keeps intermediate quantities consistent."
    },
    {
        "category": "Reasoning\nControl\nFailure",
        "case": "Case 3",
        "title": "Over-expansion",
        "question": "Great Grandma has 3 children; each child has 3 children; each grandchild has 3 babies. How many great-grandbabies?",
        "baseline": "Reaches target then continues:\nChildren: 3\nGrandchildren: 9\nGreat-grandbabies: 27\nExtra step: 27 × 3 = 81",
        "lift": "Stops at target depth:\nGrandchildren = 3 × 3 = 9\nGreat-grandbabies = 9 × 3 = 27",
        "answer_bad": "81",
        "answer_good": "27",
        "takeaway": "Suppresses unnecessary recursive expansion."
    },
    {
        "category": "Reasoning\nControl\nFailure",
        "case": "Case 4",
        "title": "Unit / variable drift",
        "question": "An alarm rang 4 times, then 3 times as long, then half as long as the second. How many rings in total?",
        "baseline": "Changes count into duration:\nFirst ring = x seconds\nOutputs irrelevant condition: 3",
        "lift": "Keeps count variable:\nFirst = 4\nSecond = 12\nThird = 6\nTotal = 22",
        "answer_bad": "3",
        "answer_good": "22",
        "takeaway": "Prevents variable-binding and unit drift."
    },
    {
        "category": "Question-\nCondition\nAlignment\nFailure",
        "case": "Case 5",
        "title": "Constraint omission",
        "question": "A bag has 35 marshmallows. John and DeSean use 9 each, and 3 are dropped. How many more s'mores can each kid make?",
        "baseline": "Omits dropped condition:\n35 − 18 = 17\n17 ÷ 2 ≈ 8",
        "lift": "Uses all constraints:\n35 − 18 − 3 = 14\n14 ÷ 2 = 7",
        "answer_bad": "8",
        "answer_good": "7",
        "takeaway": "Preserves stated conditions before answering."
    },
    {
        "category": "Question-\nCondition\nAlignment\nFailure",
        "case": "Case 6",
        "title": "Visual-symbolic mismatch",
        "question": "Find x in a triangle. The model must pair the correct side with its opposite angle before applying the Law of Sines.",
        "baseline": "Uses wrong side-angle pair:\nx / sin 97° = 16 / sin 62°",
        "lift": "Restores correct mapping:\nx / sin 21° = 16 / sin 97°\nx ≈ 5.8",
        "answer_bad": "Wrong option",
        "answer_good": "5.8",
        "takeaway": "Aligns visual geometry with symbolic formula."
    },
]

# =========================
# Style
# =========================
COLORS = {
    "State\nMaintenance\nFailure": {
        "main": "#D7191C",
        "light": "#FFF1F1",
        "edge": "#F2A3A3",
    },
    "Reasoning\nControl\nFailure": {
        "main": "#F28E2B",
        "light": "#FFF5E8",
        "edge": "#F2C08A",
    },
    "Question-\nCondition\nAlignment\nFailure": {
        "main": "#6A3DCC",
        "light": "#F4F0FF",
        "edge": "#BDADEB",
    },
}

TEXT = "#111827"
MUTED = "#4B5563"
GREEN = "#147A3D"
GREEN_BG = "#ECFDF3"
RED = "#C62828"
RED_BG = "#FFF1F1"
GRAY_BG = "#FFFFFF"


def add_round_box(ax, x, y, w, h, fc="white", ec="#DDDDDD", lw=1.0, radius=0.02):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw
    )
    ax.add_patch(box)
    return box


def add_text(ax, x, y, s, size=9, weight="normal", color=TEXT,
             ha="left", va="top", max_chars=None, linespacing=1.2):
    if max_chars is not None:
        s = fill(s, max_chars)
    ax.text(
        x, y, s,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        family="DejaVu Sans",
        linespacing=linespacing
    )


def draw_case(ax, case, x, y, w, h):
    style = COLORS[case["category"]]

    # outer case box
    add_round_box(ax, x, y, w, h, fc="white", ec=style["edge"], lw=1.1, radius=0.018)

    # case label
    add_text(ax, x + 0.015*w, y + h - 0.055*h, case["case"],
             size=10, weight="bold", color=style["main"])
    add_text(ax, x + 0.145*w, y + h - 0.055*h, case["title"],
             size=9.5, weight="bold", color=TEXT)

    # inner columns
    margin_x = 0.025 * w
    top = y + h - 0.16 * h
    inner_h = 0.64 * h
    gap = 0.018 * w
    col_w = (w - 2 * margin_x - 2 * gap) / 3

    qx = x + margin_x
    bx = qx + col_w + gap
    lx = bx + col_w + gap

    # Question box
    add_round_box(ax, qx, top - inner_h, col_w, inner_h, fc=GRAY_BG, ec="#D1D5DB", lw=0.8, radius=0.012)
    add_text(ax, qx + 0.04*col_w, top - 0.04*inner_h, "Question / Target",
             size=7.4, weight="bold", color=TEXT)
    add_text(ax, qx + 0.04*col_w, top - 0.16*inner_h, case["question"],
             size=6.8, color=TEXT, max_chars=25, linespacing=1.18)

    # Baseline box
    add_round_box(ax, bx, top - inner_h, col_w, inner_h, fc=RED_BG, ec="#F4B4B4", lw=0.8, radius=0.012)
    add_text(ax, bx + 0.04*col_w, top - 0.04*inner_h, "Baseline",
             size=7.4, weight="bold", color=RED)
    add_text(ax, bx + 0.04*col_w, top - 0.16*inner_h, case["baseline"],
             size=6.8, color=TEXT, max_chars=24, linespacing=1.18)

    # LIFT box
    add_round_box(ax, lx, top - inner_h, col_w, inner_h, fc=GREEN_BG, ec="#A6D8B8", lw=0.8, radius=0.012)
    add_text(ax, lx + 0.04*col_w, top - 0.04*inner_h, "+ Reasoning Shift",
             size=7.4, weight="bold", color=GREEN)
    add_text(ax, lx + 0.04*col_w, top - 0.16*inner_h, case["lift"],
             size=6.8, color=TEXT, max_chars=24, linespacing=1.18)

    # answer tags
    tag_h = 0.105 * h
    tag_y = y + 0.08 * h

    add_round_box(ax, bx + 0.08*col_w, tag_y, 0.84*col_w, tag_h,
                  fc="#FFE8E8", ec="#F4A6A6", lw=0.8, radius=0.012)
    add_text(ax, bx + 0.50*col_w, tag_y + 0.067*h,
             f"Answer: {case['answer_bad']} ×",
             size=7.5, weight="bold", color=RED, ha="center", va="center")

    add_round_box(ax, lx + 0.08*col_w, tag_y, 0.84*col_w, tag_h,
                  fc="#E8F7EE", ec="#8ECFA5", lw=0.8, radius=0.012)
    add_text(ax, lx + 0.50*col_w, tag_y + 0.067*h,
             f"Answer: {case['answer_good']} ✓",
             size=7.5, weight="bold", color=GREEN, ha="center", va="center")

    # takeaway
    add_text(ax, x + 0.03*w, y + 0.035*h, case["takeaway"],
             size=6.8, color=MUTED, max_chars=62)


def draw_category(ax, category, x, y, w, h):
    style = COLORS[category]
    add_round_box(ax, x, y, w, h, fc=style["light"], ec=style["edge"], lw=1.0, radius=0.018)

    # simple icon-like mark
    cx = x + 0.50*w
    cy = y + 0.67*h
    ax.text(cx, cy, "○", fontsize=42, color=style["main"], ha="center", va="center", family="DejaVu Sans")
    ax.text(cx, cy, "↘", fontsize=22, color=style["main"], ha="center", va="center", family="DejaVu Sans")

    add_text(ax, x + 0.50*w, y + 0.43*h, category,
             size=12, weight="bold", color=style["main"],
             ha="center", va="center")


# =========================
# Figure canvas
# =========================
fig = plt.figure(figsize=(18, 12))
ax = plt.axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# title
add_text(
    ax, 0.5, 0.965,
    "Representative Case Studies of Reasoning Repairs",
    size=21, weight="bold", color=TEXT, ha="center", va="top"
)

add_text(
    ax, 0.5, 0.925,
    "Baseline VLMs often follow a plausible path but lose state, over-expand reasoning, or misalign conditions with the target. "
    "Reasoning Shifts repair these failures by stabilizing language-side reasoning.",
    size=9.5, color=MUTED, ha="center", va="top", max_chars=155
)

# layout
left_margin = 0.035
right_margin = 0.035
top_y = 0.875
bottom_margin = 0.055

cat_w = 0.12
gap_cat_case = 0.012
case_gap_x = 0.012
row_gap = 0.012

case_area_x = left_margin + cat_w + gap_cat_case
case_area_w = 1 - right_margin - case_area_x
case_w = (case_area_w - case_gap_x) / 2

row_h = (top_y - bottom_margin - 2 * row_gap) / 3

for r in range(3):
    y = top_y - (r + 1) * row_h - r * row_gap
    category = cases[2*r]["category"]

    draw_category(ax, category, left_margin, y, cat_w, row_h)

    draw_case(ax, cases[2*r], case_area_x, y, case_w, row_h)
    draw_case(ax, cases[2*r + 1], case_area_x + case_w + case_gap_x, y, case_w, row_h)

# save
plt.savefig(OUT_PATH, bbox_inches="tight")
plt.savefig("case_study.png", dpi=300, bbox_inches="tight")
print(f"Saved to {OUT_PATH} and case_study.png")