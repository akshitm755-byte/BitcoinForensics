import io
import os
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from contracts.schemas import Alert, Report


class ForensicDossierPDFExporter:
    """
    Builds an extensive, agency-grade Forensic Investigative Dossier (.pdf) embedding:
      1. High-DPI GNNExplainer Minimal Subgraph Visualization (NetworkX + Matplotlib)
      2. Ranked Feature Attribution Horizontal Bar Chart
      3. Grounded Investigative Claims Table with Verified Evidence Citations
      4. OSINT Attribution Appendix & Scientific Audit Disclaimer
    Automatically uploads to GCS when `vault_bucket` is configured so stateless Cloud Run
    services can serve PDFs cleanly across containers.
    """

    @classmethod
    def export_pdf(
        cls,
        alert: Alert,
        report: Report,
        output_dir: str = "/tmp/generated_dossiers",
        vault_bucket: Optional[str] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        local_path = os.path.join(output_dir, f"{report.report_id}.pdf")

        subgraph_buf = cls._render_subgraph_chart(alert, report)
        feat_buf = cls._render_feature_chart(report)

        doc = SimpleDocTemplate(
            local_path, pagesize=A4,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "BureauTitle", parent=styles["Heading1"],
            fontSize=15, textColor=colors.HexColor("#0f172a"), spaceAfter=4
        )
        sub_style = ParagraphStyle(
            "BureauSub", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#475569"), spaceAfter=12
        )
        h2_style = ParagraphStyle(
            "SectionH2", parent=styles["Heading2"],
            fontSize=11, textColor=colors.HexColor("#1e3a8a"), spaceBefore=10, spaceAfter=6
        )
        body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=13)

        story = []
        story.append(Paragraph("AUTONOMOUS BLOCKCHAIN INVESTIGATOR — FORENSIC CASE DOSSIER", title_style))
        story.append(Paragraph(
            f"Classification: <b>INVESTIGATIVE LEAD</b> &nbsp;|&nbsp; "
            f"Report ID: <b>{report.report_id}</b> &nbsp;|&nbsp; "
            f"Alert ID: <b>{report.alert_id}</b> &nbsp;|&nbsp; "
            f"Correlation ID: {report.correlation_id}",
            sub_style
        ))

        # Executive Metadata Table
        meta_data = [
            ["Predicted Typology", alert.predicted_typology.value.upper(), "Illicit Risk Score", f"{alert.risk_score:.2%}"],
            ["Model Version", alert.model_version, "Checkpoint Status", alert.model_checkpoint_status],
            ["Compliance Verdict", report.critic_verdict, "Faithfulness Score", f"{report.faithfulness_score:.1%}"],
            ["Explainer Engine", report.evidence.explainer_version, "OSINT Attribution", "MATCHED" if report.evidence.attribution_available else "UNAVAILABLE"],
        ]
        meta_tbl = Table(meta_data, colWidths=[1.4 * inch, 2.2 * inch, 1.4 * inch, 2.2 * inch])
        meta_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(meta_tbl)
        story.append(Spacer(1, 10))

        # Section 1: Visual Forensic Evidence
        story.append(Paragraph("1. GNNExplainer Minimal Subgraph & Feature Attribution Visuals", h2_style))
        img_tbl = Table(
            [[Image(subgraph_buf, width=3.6 * inch, height=2.45 * inch),
              Image(feat_buf, width=3.5 * inch, height=2.45 * inch)]],
            colWidths=[3.65 * inch, 3.55 * inch]
        )
        story.append(img_tbl)
        story.append(Spacer(1, 8))

        # Section 2: Evidence-Grounded Claims
        story.append(Paragraph("2. Compliance-Verified Investigative Narrative", h2_style))
        claim_rows = [["Claim ID", "Evidence-Grounded Statement", "Resolved Evidence Citations"]]
        for c in report.claims:
            claim_rows.append([
                Paragraph(f"<b>{c.claim_id}</b>", body_style),
                Paragraph(c.sentence, body_style),
                Paragraph(", ".join(c.evidence_refs), body_style),
            ])
        c_tbl = Table(claim_rows, colWidths=[0.8 * inch, 4.3 * inch, 2.1 * inch])
        c_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(c_tbl)
        story.append(Spacer(1, 10))

        # Section 3: Threat Intelligence Appendix
        story.append(Paragraph("3. Real-World Threat Intelligence Attribution", h2_style))
        if report.evidence.attribution_available and report.evidence.doc_refs:
            for d in report.evidence.doc_refs:
                story.append(Paragraph(
                    f"<b>[{d.doc_id}] {d.source_title}</b> (Entity: <i>{d.entity_id}</i>, Span: {d.span_start}-{d.span_end})<br/>{d.excerpt}",
                    body_style
                ))
                story.append(Spacer(1, 4))
        else:
            story.append(Paragraph(
                "No public OSINT or sanctions registry matches were found for the wallets in this anonymized cluster. "
                "Attribution is explicitly marked unavailable in accordance with anti-hallucination protocol.",
                body_style
            ))

        story.append(Spacer(1, 14))
        story.append(Paragraph(
            "<i>Disclaimer: This dossier is an auditable, source-cited investigative lead generated by Temporal Bit-CHetG "
            "and PyTorch GNNExplainer to support expert human review. It is not standalone legally admissible evidence.</i>",
            sub_style
        ))

        doc.build(story)
        report.pdf_local_path = local_path

        if vault_bucket:
            try:
                from google.cloud import storage
                client = storage.Client()
                blob_name = f"dossiers/{report.report_id}.pdf"
                client.bucket(vault_bucket).blob(blob_name).upload_from_filename(local_path)
                report.pdf_gcs_uri = f"gs://{vault_bucket}/{blob_name}"
            except Exception as exc:
                print(f"[PDFExporter] GCS upload skipped ({exc}); kept local file.")

        return local_path

    @staticmethod
    def _render_subgraph_chart(alert: Alert, report: Report) -> io.BytesIO:
        fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")

        G = nx.DiGraph()
        min_nodes = set(report.evidence.minimal_nodes)
        min_edges = set(report.evidence.minimal_edges)

        for n in alert.subgraph_nodes[:24]:
            G.add_node(n.node_id)
        for e in alert.subgraph_edges:
            if e.source_id in G and e.target_id in G:
                G.add_edge(e.source_id, e.target_id, flagged=(e.edge_id in min_edges))

        pos = nx.spring_layout(G, seed=42, k=0.7) if len(G) > 0 else {}
        node_colors = ["#ef4444" if n in min_nodes else "#38bdf8" for n in G.nodes()]
        edge_colors = ["#ef4444" if G[u][v].get("flagged") else "#475569" for u, v in G.edges()]
        edge_widths = [2.2 if G[u][v].get("flagged") else 0.9 for u, v in G.edges()]

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, arrows=True, arrowsize=8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=140, edgecolors="#f8fafc", linewidths=0.7)
        short_labels = {n: (n[:11] + "…" if len(n) > 12 else n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=short_labels, ax=ax, font_size=5.5, font_color="#e2e8f0")

        ax.set_title("Minimal Flagged Subgraph (Crimson = GNNExplainer Mask)", color="#f8fafc", fontsize=8, pad=6)
        ax.axis("off")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf

    @staticmethod
    def _render_feature_chart(report: Report) -> io.BytesIO:
        fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=150)
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")

        items = sorted(report.evidence.feature_attributions.items(), key=lambda kv: kv[1])
        labels = [k.replace("feat:", "") for k, _ in items]
        vals = [v for _, v in items]

        bars = ax.barh(labels, vals, color="#38bdf8", edgecolor="#0284c7", height=0.6)
        if bars:
            bars[-1].set_color("#ef4444")
            bars[-1].set_edgecolor("#b91c1c")
        ax.set_title("GNNExplainer Feature Attributions", color="#f8fafc", fontsize=8, pad=6)
        ax.tick_params(colors="#cbd5e1", labelsize=7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#334155")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf
