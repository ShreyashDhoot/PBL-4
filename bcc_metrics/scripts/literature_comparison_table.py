#!/usr/bin/env python3
"""
literature_comparison_table.py
================================
Notes reference: Section 5 ("Accuracy / agreement numbers for real lab
analyzers, with sources") and Section 6, Step 6 ("drop the result straight
into the Section 5 table").

The literature rows below are transcribed directly from Comparison_Metrics_Notes.docx
Section 5 (high-end/mid-tier and low-end/point-of-care analyzer tables), each
with its numbered source. Run agreement_stats.py first so
output/tables/section5_comparison_row.csv exists; this script appends that
"Our system" row underneath the literature rows, producing the exact table
the notes describe: not a converted accuracy percentage, but r / regression /
Bland-Altman numbers in the same statistical language as every other row.

Outputs:
  output/tables/literature_comparison_full.csv
  output/reports/literature_comparison_full.md   (paper-ready markdown table)
"""

import pandas as pd

from common import TABLES_DIR, REPORTS_DIR, log

LITERATURE_ROWS = [
    # High-end / mid-tier analyzers
    {"Tier": "High-end/mid-tier", "Analyzer": "Sysmex XN-1000",
     "Compared against": "Atellica HEMA 580 (CLSI EP09c method comparison, n=100)",
     "Reported numbers": "r = 0.999 (RBC), 0.998 (HGB), 0.996 (WBC), 0.995 (PLT-I)",
     "Source": "[1]"},
    {"Tier": "High-end/mid-tier", "Analyzer": "Sysmex XR-series",
     "Compared against": "Established Sysmex XN reference system",
     "Reported numbers": "r = 0.998 (RBC), 1.000 (WBC), 1.000 (PLT), 0.999 (HGB); "
                          "Bland-Altman bias ~0 except PLT (-1.6, LoA -11.8 to 8.6)",
     "Source": "[2]"},
    {"Tier": "High-end/mid-tier", "Analyzer": "Beckman Coulter DxH 900",
     "Compared against": "Manual 400-cell microscopic differential (n=100)",
     "Reported numbers": "R >= 0.88 for leukocyte differential, all parameters except basophils; "
                          "flag sensitivity >90%",
     "Source": "[3]"},
    {"Tier": "High-end/mid-tier", "Analyzer": "Beckman Coulter DxH 900",
     "Compared against": "Sysmex XN20 (reference: manual microscopy for abnormal-cell flags)",
     "Reported numbers": "\"Excellent degree of association\" for leukocyte differential; "
                          "IG% bias 1.2% (95% CI 0.9-1.5), r = 0.90",
     "Source": "[4]"},
    {"Tier": "High-end/mid-tier", "Analyzer": "Dymind DH76 (mid/5-part)",
     "Compared against": "Sysmex XN-1000",
     "Reported numbers": "R^2 = 1.000 (WBC), 0.999 (RBC), 0.999 (HGB), 0.994 (PLT>50e9/L), 0.910 (PLT<50e9/L)",
     "Source": "[5]"},
    # Low-end / 3-part and point-of-care analyzers
    {"Tier": "Low-end/POC", "Analyzer": "Sysmex XQ-320 (3-part)",
     "Compared against": "Sysmex XN-9000, used as in-lab gold standard (n=493)",
     "Reported numbers": "r > 0.94 for most of 20 parameters; weaker for MXD# (0.891), "
                          "MXD% (0.898), MCHC (0.849)",
     "Source": "[6]"},
    {"Tier": "Low-end/POC", "Analyzer": "ERBA ELite 580 (budget 5-part)",
     "Compared against": "Beckman Coulter LH 780",
     "Reported numbers": "R^2 > 0.90 for most CBC/differential parameters, except MCHC (0.35) and basophils",
     "Source": "[7]"},
    {"Tier": "Low-end/POC", "Analyzer": "HemoScreen (point-of-care, microfluidic)",
     "Compared against": "Sysmex XN (multi-site)",
     "Reported numbers": "Bland-Altman bias & 95% LoA reported per parameter; agreement good overall "
                          "but LoA noticeably wider than high-end-vs-high-end comparisons",
     "Source": "[8]"},
    {"Tier": "Low-end/POC", "Analyzer": "Cito POC CBC (microfluidic, CLIA-waived)",
     "Compared against": "In-laboratory reference hematology analyzer (n=551)",
     "Reported numbers": "Mean % bias <2% for WBC, RBC, PLT, HGB, HCT; larger bias for monocytes "
                          "(+25.9%), basophils (-15%); overall abnormal-flag agreement 93.3%",
     "Source": "[9]"},
    {"Tier": "Low-end/POC", "Analyzer": "QBC Autoread Plus (basic point-of-care)",
     "Compared against": "Trained lab staff, same device, operated by non-medical staff",
     "Reported numbers": "r = 0.91 (Hct), 0.96 (WBC), 0.92 (Plt); Bland-Altman bias 0.5% (Hct), "
                          "0.1x10^9/L (WBC), 10x10^9/L (Plt)",
     "Source": "[10]"},
    {"Tier": "Low-end/POC", "Analyzer": "Mindray BC-5180 (budget 5-part)",
     "Compared against": "Sysmex XN-1000",
     "Reported numbers": "Passing-Bablok regression + Pearson correlation reported; strong agreement, "
                          "consistent with the mid-tier analyzers above",
     "Source": "[11]"},
]

LITERATURE_REFERENCES_MD = """
**References (as cited in Comparison_Metrics_Notes.docx, Section 5):**

[1] Evaluation of the New Beckman Coulter Analyzer DxH 900 Compared to Sysmex XN20: Analytical
Performance and Flagging Efficiency. Int J Lab Hematol / PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8535162/

[2] Performance evaluation of the new Sysmex XR-Series haematology analyser. ScienceDirect
(Practical Laboratory Medicine). https://www.sciencedirect.com/science/article/pii/S2352551724000167

[3] Performance evaluation of the new hematology analyzer UniCel DxH 900. PubMed.
https://pubmed.ncbi.nlm.nih.gov/33389827/

[4] Evaluation of the New Beckmann Coulter Analyzer DxH 900 Compared to Sysmex XN20 (IG% Bland-Altman
detail). PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8535162/

[5] Evaluation of automated hematology analyzer DYMIND DH76 compared to SYSMEX XN 1000 system. PMC.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8451233/

[6] Evaluation of the Sysmex XQ-320 three-part differential haematology analyser and its flagging
capabilities. PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10943257/

[7] Performance Evaluation of Fully Automated 5-Part Differential Hematology Analyzer ELite 580
(ERBA). PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC8804059/

[8] Clinical Evaluation of a Novel Point-of-Care Hematology Analyzer for Complete Blood Count With
Differential. Int J Lab Hematol, 2026. https://onlinelibrary.wiley.com/doi/10.1111/ijlh.70032

[9] Analytical performance of a point-of-care CBC hematology analyzer, including a 5-part
differential. American Journal of Clinical Pathology. https://doi.org/10.1093/ajcp/aqae149

[10] The accuracy of a point-of-care test among different operators using the QBC Autoread Plus
Analyzer. PubMed. https://pubmed.ncbi.nlm.nih.gov/31865958/

[11] Analytical comparison between two hematological analyzer systems: Mindray BC-5180 vs
Sysmex XN-1000. PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6805265/
"""


def main():
    lit_df = pd.DataFrame(LITERATURE_ROWS)

    section5_path = TABLES_DIR / "section5_comparison_row.csv"
    if section5_path.exists():
        ours = pd.read_csv(section5_path)
        our_row = {
            "Tier": "This study",
            "Analyzer": ours.loc[0, "System"],
            "Compared against": ours.loc[0, "Compared against"],
            "Reported numbers": (
                f"Pearson {ours.loc[0, 'Correlation (r)']}; "
                f"Bland-Altman {ours.loc[0, 'Bland-Altman bias / LoA']}; "
                f"Passing-Bablok {ours.loc[0, 'Passing-Bablok slope/intercept']}"
            ),
            "Source": ours.loc[0, "Source"],
            "Sample-size caveat": ours.loc[0, "Sample-size caveat"],
        }
        full_df = pd.concat([lit_df, pd.DataFrame([our_row])], ignore_index=True)
    else:
        log("output/tables/section5_comparison_row.csv not found -- run agreement_stats.py first. "
            "Writing literature-only table for now.", tag="WARN")
        full_df = lit_df

    full_df.to_csv(TABLES_DIR / "literature_comparison_full.csv", index=False)

    md_lines = ["# Method-Comparison Table (Section 5/6 of Comparison_Metrics_Notes.docx)", ""]
    md_lines.append(full_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append(LITERATURE_REFERENCES_MD)
    with open(REPORTS_DIR / "literature_comparison_full.md", "w") as f:
        f.write("\n".join(md_lines))

    log(f"Wrote {TABLES_DIR / 'literature_comparison_full.csv'} "
        f"and {REPORTS_DIR / 'literature_comparison_full.md'}")


if __name__ == "__main__":
    main()
