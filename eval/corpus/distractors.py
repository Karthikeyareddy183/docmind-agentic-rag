"""Hard-negative documents for the evaluation corpus.

They share layout and vocabulary with the gold documents but carry different facts, so a
retriever only scores well if it matches the *specific* entity, year, and audience asked about:

- Northwind FY2024 annual report      same company and wording, previous year's numbers
- Southwind / Eastwind FY2025 reports competitors using the same report template
- Northwind board minutes, Dec 2025   earlier scanned minutes (interim dividend, other dates)
- Northwind contractor IT policy      security rules for contractors, not employees
"""

from pathlib import Path

import matplotlib.pyplot as plt
from generate import BODY, H1, H2, _chart_png, _scan_page, _table  # type: ignore[attr-defined]
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

REPORTS = [
    {
        "file": "northwind_annual_report_fy2024.pdf",
        "company": "Northwind Analytics Ltd.",
        "year": "2024",
        "ceo": "Meera Iyer",
        "hq": "Pune, India",
        "founded": "2011",
        "revenue": "409.0",
        "growth": "14%",
        "employees": "2,010",
        "offices": "four",
        "retention": "117%",
        "new_accounts": "48",
        "segments": [
            ("Cloud Platform", ["44.1", "46.9", "49.8", "53.2"], "194.0"),
            ("Data Services", ["27.6", "28.9", "30.1", "31.4"], "118.0"),
            ("Consulting", ["23.1", "23.9", "24.6", "25.4"], "97.0"),
        ],
        "totals": ["94.8", "99.7", "104.5", "110.0"],
        "margin_years": ["2020", "2021", "2022", "2023", "2024"],
        "margins": [9.8, 11.2, 13.5, 15.1, 17.8],
        "regions": [36, 35, 21, 8],
        "emissions": "We reduced absolute carbon emissions by 9% compared with the 2022 baseline. Renewable electricity powered 51% of our data centres. Voluntary employee attrition was 16.4%, compared with 18.0% the year before.",
        "risks": "Our top ten customers accounted for 29% of revenue.",
        "outlook": "For fiscal year 2025 we expect revenue between $470 million and $490 million, with capital expenditure of $33 million. We plan to open a Berlin office during FY2025.",
        "headcount": [
            ("Engineering", "960"),
            ("Consulting", "470"),
            ("Sales & Marketing", "350"),
            ("General & Administrative", "230"),
        ],
        "total_headcount": "2,010",
        "segment_notes": "The Cloud Platform launched multi-region failover in the second quarter. Data Services released its data-quality monitor, adopted by 85 customers. Consultant utilisation averaged 74%.",
    },
    {
        "file": "southwind_annual_report_fy2025.pdf",
        "company": "Southwind Analytics Pvt. Ltd.",
        "year": "2025",
        "ceo": "Rahul Menon",
        "hq": "Hyderabad, India",
        "founded": "2014",
        "revenue": "301.4",
        "growth": "11%",
        "employees": "1,580",
        "offices": "three",
        "retention": "109%",
        "new_accounts": "33",
        "segments": [
            ("Cloud Platform", ["30.2", "31.8", "33.0", "34.9"], "129.9"),
            ("Data Services", ["22.4", "23.1", "24.0", "24.8"], "94.3"),
            ("Consulting", ["18.6", "19.1", "19.5", "20.0"], "77.2"),
        ],
        "totals": ["71.2", "74.0", "76.5", "79.7"],
        "margin_years": ["2021", "2022", "2023", "2024", "2025"],
        "margins": [7.4, 8.1, 9.9, 10.6, 12.2],
        "regions": [52, 21, 17, 10],
        "emissions": "We reduced absolute carbon emissions by 14% compared with the 2022 baseline. Renewable electricity powers 47% of our data centres. Voluntary employee attrition was 15.2%.",
        "risks": "Our top ten customers account for 35% of revenue.",
        "outlook": "For fiscal year 2026 we expect revenue between $330 million and $345 million, with capital expenditure of $22 million. We plan to open a Dubai office in the third quarter of FY2026.",
        "headcount": [
            ("Engineering", "740"),
            ("Consulting", "390"),
            ("Sales & Marketing", "270"),
            ("General & Administrative", "180"),
        ],
        "total_headcount": "1,580",
        "segment_notes": "The Cloud Platform introduced a serverless query engine. Data Services launched automated PII discovery, used by 60 customers. Consultant utilisation averaged 81%.",
    },
    {
        "file": "eastwind_annual_report_fy2025.pdf",
        "company": "Eastwind Data Systems Ltd.",
        "year": "2025",
        "ceo": "Ananya Bose",
        "hq": "Chennai, India",
        "founded": "2009",
        "revenue": "655.2",
        "growth": "9%",
        "employees": "3,900",
        "offices": "eight",
        "retention": "112%",
        "new_accounts": "74",
        "segments": [
            ("Cloud Platform", ["70.5", "72.8", "75.1", "77.9"], "296.3"),
            ("Data Services", ["48.0", "49.2", "50.6", "51.9"], "199.7"),
            ("Consulting", ["38.7", "39.4", "40.1", "41.0"], "159.2"),
        ],
        "totals": ["157.2", "161.4", "165.8", "170.8"],
        "margin_years": ["2021", "2022", "2023", "2024", "2025"],
        "margins": [16.0, 15.2, 14.8, 15.9, 16.7],
        "regions": [29, 41, 22, 8],
        "emissions": "We reduced absolute carbon emissions by 18% compared with the 2021 baseline. Renewable electricity powers 70% of our data centres. Voluntary employee attrition was 11.8%.",
        "risks": "Our top ten customers account for 22% of revenue.",
        "outlook": "For fiscal year 2026 we expect revenue between $700 million and $720 million, with capital expenditure of $58 million. We plan to open a Toronto office in the first quarter of FY2026.",
        "headcount": [
            ("Engineering", "1,850"),
            ("Consulting", "1,020"),
            ("Sales & Marketing", "640"),
            ("General & Administrative", "390"),
        ],
        "total_headcount": "3,900",
        "segment_notes": "The Cloud Platform added GPU-backed analytics clusters. Data Services expanded its catalogue product to 210 customers. Consultant utilisation averaged 76%.",
    },
]


def annual_report(path: Path, d: dict) -> None:
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, title=f"{d['company']} Annual Report FY{d['year']}"
    )
    s: list[object] = [
        Paragraph(d["company"], H1),
        Paragraph(f"Annual Report — Fiscal Year {d['year']}", H2),
        Spacer(1, 0.4 * cm),
        Paragraph("Letter from the Chief Executive Officer", H2),
        Paragraph(
            f"Dear shareholders, in fiscal year {d['year']} total revenue grew {d['growth']} year over year "
            f"to ${d['revenue']} million. We ended the year with {d['employees']} employees across "
            f"{d['offices']} offices, and our headquarters remains in {d['hq']}, where the company was "
            f"founded in {d['founded']}.",
            BODY,
        ),
        Spacer(1, 0.3 * cm),
        Paragraph(
            f"Net retention among enterprise customers reached {d['retention']}, and we signed "
            f"{d['new_accounts']} new enterprise accounts during the year.",
            BODY,
        ),
        Paragraph(f"— {d['ceo']}, Chief Executive Officer", BODY),
        PageBreak(),
        Paragraph("Business Segments", H1),
        Paragraph(d["segment_notes"], BODY),
        PageBreak(),
        Paragraph("Revenue by Segment", H1),
        Paragraph(f"Quarterly revenue by segment for FY{d['year']}, in US$ millions.", BODY),
        Spacer(1, 0.3 * cm),
        _table(
            [["Segment", "Q1", "Q2", "Q3", "Q4", f"FY{d['year']}"]]
            + [[name, *q, total] for name, q, total in d["segments"]]
            + [["Total", *d["totals"], d["revenue"]]],
            [4.2 * cm] + [2.2 * cm] * 5,
        ),
        PageBreak(),
    ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(d["margin_years"], d["margins"], color="#6c8ead")
    ax.bar_label(bars, labels=[f"{m}%" for m in d["margins"]])
    ax.set_title("Operating margin by fiscal year")
    ax.set_ylabel("Operating margin (%)")
    ax.set_ylim(0, 24)
    margin_png = _chart_png(fig)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    labels = ["India", "North America", "Europe", "Rest of APAC"]
    ax.pie(
        d["regions"],
        labels=[f"{n} {p}%" for n, p in zip(labels, d["regions"], strict=True)],
        startangle=90,
    )
    ax.set_title(f"FY{d['year']} revenue by region")
    region_png = _chart_png(fig)

    s += [
        Paragraph("Profitability and Geographic Mix", H1),
        RLImage(margin_png, width=14 * cm, height=7.4 * cm),
        Spacer(1, 0.3 * cm),
        RLImage(region_png, width=10.5 * cm, height=7.3 * cm),
        PageBreak(),
        Paragraph("Sustainability and People", H1),
        Paragraph(d["emissions"], BODY),
        Paragraph("Principal Risks", H2),
        Paragraph(d["risks"], BODY),
        PageBreak(),
        Paragraph(f"Outlook for FY{int(d['year']) + 1}", H1),
        Paragraph(d["outlook"], BODY),
        PageBreak(),
        Paragraph("Headcount by Function", H1),
        _table(
            [
                ["Function", "Employees"],
                *[list(r) for r in d["headcount"]],
                ["Total", d["total_headcount"]],
            ],
            [6 * cm, 3 * cm],
        ),
    ]
    doc.build(s)


def board_minutes_dec2025(path: Path) -> None:
    page = [
        "NORTHWIND ANALYTICS LTD.",
        "Minutes of the Board of Directors Meeting",
        "Date: 10 December 2025        Venue: Mumbai office",
        "",
        "Present: 8 directors, including the Chair, Arvind Rao.",
        "",
        "1. Interim dividend",
        "The Board approved an interim dividend of Rs 3.75 per equity share.",
        "",
        "2. Budget",
        "The Board approved the FY2026 operating budget and a capital",
        "expenditure envelope of USD 41 million.",
        "",
        "3. Next meeting",
        "The next Board meeting will be held on 14 March 2026.",
    ]
    _scan_page(page).save(path, "PDF", resolution=200.0)


def contractor_policy(path: Path) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="Northwind Contractor IT Access Policy")
    s: list[object] = [
        Paragraph("Northwind Analytics — Contractor IT Access Policy", H1),
        Paragraph(
            "Applies to contractors, vendors, and temporary staff. Effective 1 March 2026.", BODY
        ),
        Paragraph("1. Accounts and Passwords", H2),
        Paragraph(
            "Contractor accounts expire after 90 days unless renewed by the sponsoring manager. "
            "Contractor passwords must be at least 16 characters and are rotated every 60 days.",
            BODY,
        ),
        Paragraph("2. Devices", H2),
        Paragraph(
            "Contractors must use company-managed virtual desktops; personal devices may not store "
            "company data. Lost devices must be reported to the vendor manager within 24 hours.",
            BODY,
        ),
        Paragraph("3. Remote Access", H2),
        Paragraph(
            "Remote access is permitted only through the contractor VPN between 07:00 and 22:00 IST.",
            BODY,
        ),
    ]
    doc.build(s)


def build(out: Path) -> None:
    for report in REPORTS:
        annual_report(out / report["file"], report)
    board_minutes_dec2025(out / "northwind_board_minutes_dec2025.pdf")
    contractor_policy(out / "northwind_contractor_it_policy.pdf")
