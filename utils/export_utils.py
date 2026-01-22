from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Dict

import pandas as pd


def _write_excel(named_frames: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in named_frames.items():
            safe_name = name[:31] if len(name) > 31 else name
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(
                writer, sheet_name=safe_name, index=False
            )
    buffer.seek(0)
    return buffer.read()


def export_excel(filtered_df: pd.DataFrame, kpis: Dict[str, float], cruise_df: pd.DataFrame, closures_df: pd.DataFrame) -> bytes:
    """Build an Excel export aligned with the new app usage.

    Sheets:
    - Donnees_filtrees
    - KPIs
    - Rythme_de_croisiere
    - Fermetures_probables
    """
    # KPIs as two-column table
    kpis_df = pd.DataFrame(
        {"KPI": list(kpis.keys()), "Valeur": [kpis[k] for k in kpis.keys()]}
    ) if isinstance(kpis, dict) else pd.DataFrame()

    named_frames: Dict[str, pd.DataFrame] = {
        "Donnees_filtrees": filtered_df.copy() if isinstance(filtered_df, pd.DataFrame) else pd.DataFrame(),
        "KPIs": kpis_df,
        "Rythme_de_croisiere": cruise_df.copy() if isinstance(cruise_df, pd.DataFrame) else pd.DataFrame(),
        "Fermetures_probables": closures_df.copy() if isinstance(closures_df, pd.DataFrame) else pd.DataFrame(),
    }
    return _write_excel(named_frames)


def default_output_filename(prefix: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    return os.path.join(outputs_dir, f"{prefix}_{ts}.{ext}")


def export_pdf_simple(filtered_df: pd.DataFrame, kpis: Dict[str, float], cruise_df: pd.DataFrame, closures_df: pd.DataFrame) -> bytes:
    """Build a simple PDF memo aligned with the new app usage.

    Contains a title and short sections summarizing KPIs and key tables sizes.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception:
        return b""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title = "Lapaire – Synthèse d'analyse"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    # KPIs section
    story.append(Paragraph("KPIs exécutifs", styles["Heading2"]))
    if isinstance(kpis, dict) and kpis:
        for k, v in kpis.items():
            story.append(Paragraph(f"{k}: {v}", styles["BodyText"]))
    else:
        story.append(Paragraph("Aucun KPI disponible.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Tables sizes
    story.append(Paragraph("Tailles des tableaux", styles["Heading2"]))
    try:
        story.append(Paragraph(f"Données filtrées: {len(filtered_df)} lignes", styles["BodyText"]))
    except Exception:
        story.append(Paragraph("Données filtrées indisponibles", styles["BodyText"]))
    try:
        story.append(Paragraph(f"Rythme de croisière: {len(cruise_df)} lignes", styles["BodyText"]))
    except Exception:
        pass
    try:
        story.append(Paragraph(f"Fermetures probables: {len(closures_df)} lignes", styles["BodyText"]))
    except Exception:
        pass

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


