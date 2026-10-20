"""Generate the synthetic evaluation corpus.

Four documents, designed so every answer in ``eval/datasets/questions.jsonl`` is checkable and
each retrieval path is exercised:

- northwind_annual_report_fy2025.pdf  born-digital text, two tables, two charts (chart-only facts)
- northwind_board_minutes_mar2026.pdf image-only "scan" (facts reachable only through OCR)
- invoice_inv_2026_0457.png           photographed invoice (OCR on an image upload)
- northwind_employee_handbook.pdf     long policy text with section headings

Usage: python eval/corpus/generate.py [output_dir]
"""

import io
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "files"
STYLES = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=STYLES["BodyText"], fontSize=10.5, leading=15)
H1, H2 = STYLES["Heading1"], STYLES["Heading2"]
random.seed(7)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _table(rows: list[list[str]], col_widths: list[float]) -> Table:
    t = Table(rows, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ]
        )
    )
    return t


def _chart_png(fig: plt.Figure) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def annual_report(path: Path) -> None:
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, title="Northwind Analytics Annual Report FY2025"
    )
    s: list[object] = []

    # Page 1
    s += [
        Paragraph("Northwind Analytics Ltd.", H1),
        Paragraph("Annual Report — Fiscal Year 2025", H2),
        Spacer(1, 0.4 * cm),
        Paragraph("Letter from the Chief Executive Officer", H2),
        Paragraph(
            "Dear shareholders, fiscal year 2025 was the strongest year in Northwind's history. "
            "Total revenue grew 18% year over year to $482.6 million, driven by sustained demand "
            "for our Cloud Platform. We ended the year with 2,340 employees across five offices, "
            "and our headquarters remains in Pune, India, where the company was founded in 2011.",
            BODY,
        ),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "Net retention among enterprise customers reached 124%, and we signed 61 new enterprise "
            "accounts during the year. I want to thank our customers and our teams for their trust.",
            BODY,
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("— Meera Iyer, Chief Executive Officer", BODY),
        PageBreak(),
    ]

    # Page 2
    s += [
        Paragraph("Business Segments", H1),
        Paragraph("Cloud Platform", H2),
        Paragraph(
            "The Cloud Platform segment provides managed analytics infrastructure on a subscription "
            "basis. It is our largest segment and contributed the majority of growth in FY2025. "
            "The segment launched real-time streaming ingestion in the third quarter.",
            BODY,
        ),
        Paragraph("Data Services", H2),
        Paragraph(
            "Data Services covers data engineering, governance, and quality tooling. The segment "
            "introduced an automated data-lineage product used by 140 customers at year end.",
            BODY,
        ),
        Paragraph("Consulting", H2),
        Paragraph(
            "Consulting delivers implementation and advisory engagements. Utilisation of billable "
            "consultants averaged 78% during the year.",
            BODY,
        ),
        PageBreak(),
    ]

    # Page 3 — revenue table
    s += [
        Paragraph("Revenue by Segment", H1),
        Paragraph("Quarterly revenue by segment for FY2025, in US$ millions.", BODY),
        Spacer(1, 0.3 * cm),
        _table(
            [
                ["Segment", "Q1", "Q2", "Q3", "Q4", "FY2025"],
                ["Cloud Platform", "52.4", "56.1", "60.3", "64.8", "233.6"],
                ["Data Services", "31.2", "33.0", "34.7", "36.5", "135.4"],
                ["Consulting", "27.0", "28.1", "28.9", "29.6", "113.6"],
                ["Total", "110.6", "117.2", "123.9", "130.9", "482.6"],
            ],
            [4.2 * cm] + [2.2 * cm] * 5,
        ),
        Spacer(1, 0.4 * cm),
        Paragraph("Fourth-quarter revenue was the highest of the year in every segment.", BODY),
        PageBreak(),
    ]

    # Page 4 — charts (facts appear only in the images)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    years = ["2021", "2022", "2023", "2024", "2025"]
    margins = [11.2, 13.5, 15.1, 17.8, 19.4]
    bars = ax.bar(years, margins, color="#2f6690")
    ax.bar_label(bars, labels=[f"{m}%" for m in margins])
    ax.set_title("Operating margin by fiscal year")
    ax.set_ylabel("Operating margin (%)")
    ax.set_ylim(0, 24)
    margin_png = _chart_png(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.pie(
        [38, 34, 20, 8],
        labels=["India 38%", "North America 34%", "Europe 20%", "Rest of APAC 8%"],
        colors=["#2f6690", "#3a7ca5", "#81c3d7", "#d9dcd6"],
        startangle=90,
    )
    ax.set_title("FY2025 revenue by region")
    region_png = _chart_png(fig)

    s += [
        Paragraph("Profitability and Geographic Mix", H1),
        RLImage(margin_png, width=14 * cm, height=7.4 * cm),
        Spacer(1, 0.3 * cm),
        RLImage(region_png, width=10.5 * cm, height=7.3 * cm),
        PageBreak(),
    ]

    # Page 5 — risk and ESG
    s += [
        Paragraph("Sustainability and People", H1),
        Paragraph(
            "We reduced absolute carbon emissions by 22% compared with the 2023 baseline. "
            "Renewable electricity now powers 64% of the data centres we operate. Voluntary "
            "employee attrition improved to 13.1% from 16.4% in the prior year.",
            BODY,
        ),
        Paragraph("Principal Risks", H2),
        Paragraph(
            "Key risks include concentration of revenue among our top ten customers, who account "
            "for 27% of revenue; currency fluctuation between the Indian rupee and the US dollar; "
            "and competition from hyperscale cloud providers.",
            BODY,
        ),
        PageBreak(),
    ]

    # Page 6 — outlook
    s += [
        Paragraph("Outlook for FY2026", H1),
        Paragraph(
            "For fiscal year 2026 we expect revenue between $560 million and $580 million. We "
            "plan capital expenditure of $41 million, mainly for new data-centre capacity, and "
            "we intend to open a Singapore office in the second quarter of FY2026.",
            BODY,
        ),
        PageBreak(),
    ]

    # Page 7 — headcount table
    s += [
        Paragraph("Headcount by Function", H1),
        _table(
            [
                ["Function", "Employees"],
                ["Engineering", "1,120"],
                ["Consulting", "520"],
                ["Sales & Marketing", "410"],
                ["General & Administrative", "290"],
                ["Total", "2,340"],
            ],
            [6 * cm, 3 * cm],
        ),
    ]
    doc.build(s)


def _scan_page(lines: list[str], size: tuple[int, int] = (1654, 2339)) -> Image.Image:
    """Render text like a printed page, then degrade it like a phone/flatbed scan."""
    img = Image.new("L", size, 250)
    draw = ImageDraw.Draw(img)
    y = 140
    for line in lines:
        font = _font(46 if line.isupper() else 34)
        draw.text((140, y), line, fill=25, font=font)
        y += 70 if line else 40
    img = img.rotate(random.uniform(-1.2, 1.2), fillcolor=245, resample=Image.Resampling.BICUBIC)
    # Sensor noise + a little blur, like a phone or flatbed scan
    rng = np.random.default_rng(7)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 14, (size[1], size[0]))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.7)
    )
    return img.convert("RGB")


def board_minutes(path: Path) -> None:
    page1 = [
        "NORTHWIND ANALYTICS LTD.",
        "Minutes of the Board of Directors Meeting",
        "Date: 14 March 2026        Venue: Pune headquarters",
        "",
        "Present: 9 directors, including the Chair, Arvind Rao.",
        "",
        "1. Dividend",
        "The Board approved a final dividend of Rs 4.50 per equity share",
        "for fiscal year 2025, payable on 30 April 2026.",
        "",
        "2. Acquisition of DataLoom",
        "The Board approved the acquisition of DataLoom Technologies",
        "for a total consideration of USD 28 million, subject to",
        "regulatory approval. Closing is expected in Q1 FY2027.",
    ]
    page2 = [
        "3. Appointment",
        "Priya Natarajan was appointed Chief Financial Officer",
        "with effect from 1 April 2026.",
        "",
        "4. Cybersecurity review",
        "The audit committee reported zero critical findings in the",
        "external penetration test completed in February 2026.",
        "",
        "5. Next meeting",
        "The next Board meeting will be held on 12 June 2026.",
        "",
        "Signed: Arvind Rao, Chair",
    ]
    pages = [_scan_page(page1), _scan_page(page2)]
    pages[0].save(path, "PDF", resolution=200.0, save_all=True, append_images=pages[1:])


def invoice(path: Path) -> None:
    lines = [
        "NORTHWIND ANALYTICS LTD.",
        "Tax Invoice",
        "Invoice No: INV-2026-0457      Date: 16 October 2026",
        "Bill to: Acme Retail Pvt Ltd, Bengaluru",
        "",
        "Item                                   Qty    Amount (USD)",
        "Cloud Platform subscription (annual)    1       9,600.00",
        "Onboarding workshop                     2       2,400.00",
        "Premium support add-on                  1         980.00",
        "",
        "Subtotal                                       12,980.00",
        "GST 18%                                         2,336.40",
        "TOTAL DUE                                      15,316.40",
        "",
        "Payment due by 15 November 2026.",
    ]
    img = _scan_page(lines, size=(1654, 1400))
    img.save(path, "PNG")


HANDBOOK_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "1. Working Hours and Remote Work",
        [
            "Standard working hours are 9:30 to 18:30, Monday to Friday, with a one-hour break.",
            "Employees may work remotely up to three days per week with their manager's approval. "
            "Fully remote arrangements require approval from the department head and HR.",
        ],
    ),
    (
        "2. Probation and Notice Period",
        [
            "New employees serve a probation period of six months. During probation the notice "
            "period is 15 days.",
            "After confirmation, the notice period is 60 days for all roles.",
        ],
    ),
    (
        "3. Leave Policy",
        [
            "Full-time employees receive 24 days of paid annual leave per calendar year, accrued "
            "monthly. A maximum of 10 unused days may be carried forward to the next year.",
            "Employees also receive 12 days of sick leave per year, which cannot be carried forward.",
            "Primary caregivers are entitled to 26 weeks of paid parental leave; secondary "
            "caregivers are entitled to 4 weeks.",
        ],
    ),
    (
        "4. Travel and Expenses",
        [
            "Expense claims must be submitted within 30 days of the expense with itemised receipts.",
            "The meal allowance during business travel is capped at Rs 1,500 per day. Economy class "
            "is mandatory for flights shorter than six hours.",
        ],
    ),
    (
        "5. Information Security",
        [
            "Multi-factor authentication is mandatory for all company systems. Passwords must be "
            "at least 14 characters and are rotated every 90 days.",
            "Lost or stolen devices must be reported to the security team within 4 hours.",
        ],
    ),
    (
        "6. Learning and Development",
        [
            "Each employee has an annual learning budget of Rs 40,000 for courses, certifications, "
            "and conferences. Unused budget does not roll over.",
        ],
    ),
]


def handbook(path: Path) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="Northwind Employee Handbook")
    s: list[object] = [
        Paragraph("Northwind Analytics — Employee Handbook", H1),
        Paragraph("Version 4.2, effective 1 January 2026", BODY),
        Spacer(1, 0.5 * cm),
    ]
    for i, (title, paras) in enumerate(HANDBOOK_SECTIONS):
        s.append(Paragraph(title, H2))
        s += [Paragraph(p, BODY) for p in paras]
        # Padding text so sections span pages, as in a real handbook
        s.append(
            Paragraph(
                "Questions about this section should be directed to your HR business partner. "
                "Policies are reviewed annually and changes are communicated by email.",
                BODY,
            )
        )
        if i % 2 == 1:
            s.append(PageBreak())
    doc.build(s)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    annual_report(OUT / "northwind_annual_report_fy2025.pdf")
    board_minutes(OUT / "northwind_board_minutes_mar2026.pdf")
    invoice(OUT / "invoice_inv_2026_0457.png")
    handbook(OUT / "northwind_employee_handbook.pdf")
    import distractors

    distractors.build(OUT)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:45s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
