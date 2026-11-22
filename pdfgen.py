"""pdfgen.py - ATS-friendly single-column A4 resume PDF in pure Python (reportlab). No browser needed, so it runs on Vercel."""
import io
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_REPL = {"\u2192": "->", "\u2022": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ", "\u2011": "-", "\u2212": "-"}


def _t(s) -> str:
    s = str(s or "")
    for k, v in _REPL.items():
        s = s.replace(k, v)
    return escape(s.encode("cp1252", "replace").decode("cp1252"))


def build(d: dict, contact: str, fs: float) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
                            title=f"{d['name']} - Resume", author=d["name"])
    ld = fs * 1.32
    base = ParagraphStyle("b", fontName="Helvetica", fontSize=fs, leading=ld, spaceAfter=1.5)
    h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=fs * 2, leading=fs * 2.3, alignment=TA_CENTER, spaceAfter=0)
    hl = ParagraphStyle("hl", parent=base, fontName="Helvetica-Bold", fontSize=fs * 1.1, leading=fs * 1.4, alignment=TA_CENTER)
    ct = ParagraphStyle("ct", parent=base, fontSize=fs * 0.95, alignment=TA_CENTER, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=fs * 1.1, leading=fs * 1.35, spaceBefore=8, spaceAfter=1)
    bl = ParagraphStyle("bl", parent=base, leftIndent=12, bulletIndent=2, spaceAfter=1.2)
    jt = ParagraphStyle("jt", parent=base, fontName="Helvetica-Bold", spaceAfter=0)
    jd = ParagraphStyle("jd", parent=base, alignment=2, spaceAfter=0)

    def head(txt):
        return [Paragraph(_t(txt), h2), HRFlowable(width="100%", thickness=0.8, color="#333333", spaceBefore=0, spaceAfter=3)]

    def row(left, right):
        t = Table([[Paragraph(left, jt), Paragraph(_t(right), jd)]], colWidths=[None, 42 * mm])
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    st = [Paragraph(_t(d["name"]), h1), Paragraph(_t(d["headline"]), hl), Paragraph(_t(contact), ct)]
    st += head("PROFESSIONAL SUMMARY") + [Paragraph(_t(d["summary"]), base)]
    st += head("TECHNICAL SKILLS")
    for k, v in d["skills"].items():
        st.append(Paragraph(f"<b>{_t(k)}:</b> {_t(', '.join(v))}", base))
    st += head("PROFESSIONAL EXPERIENCE")
    for j in d["experience"]:
        blk = [row(_t(f"{j['title']} - {j['company']}"), j["dates"])]
        items = [Paragraph(_t(b), bl, bulletText="-") for b in j["bullets"]]
        st.append(KeepTogether(blk + items[:2]))
        st += items[2:]
    st += head("EDUCATION")
    for e in d.get("education", []):
        st.append(row(f"{_t(e['degree'])} <font name='Helvetica'>- {_t(e['place'])}</font>", e["dates"]))
    if d.get("certifications"):
        st += head("CERTIFICATIONS") + [Paragraph(_t(" | ".join(d["certifications"])), base)]
    if d.get("awards") or d.get("languages"):
        st += head("ADDITIONAL INFORMATION")
        if d.get("languages"):
            st.append(Paragraph(f"<b>Languages:</b> {_t(', '.join(d['languages']))}", base))
        if d.get("awards"):
            st.append(Paragraph(f"<b>Awards:</b> {_t(', '.join(d['awards']))}", base))
    doc.build(st)
    return buf.getvalue()
