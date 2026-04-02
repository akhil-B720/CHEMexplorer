# -*- coding: utf-8 -*-
"""
ChemExplorer X — chemistry learning platform (Flask + RDKit + PubChem + 3Dmol.js)
"""

from __future__ import annotations

import html
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, render_template_string, request

from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/compound/"

# ---------------------------------------------------------------------------
# PubChem
# ---------------------------------------------------------------------------


def fetch_pubchem(
    compound_name: Optional[str] = None,
    smiles_input: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns: {smiles, iupac, formula} or {error: <user-friendly message>}
    """
    name = (compound_name or "").strip()
    smi = (smiles_input or "").strip()

    if not name and not smi:
        return {"error": "Invalid input"}

    try:
        if smi:
            # Resolve SMILES via PubChem for metadata (may still fail if exotic)
            enc = urllib.parse.quote(smi, safe="")
            url = (
                f"{PUBCHEM_BASE}/compound/smiles/{enc}/property/"
                "IsomericSMILES,IUPACName,MolecularFormula/JSON"
            )
            r = requests.get(url, timeout=25)
            if r.status_code != 200:
                return {"error": "Compound not found"}
            data = r.json()
            if "PropertyTable" not in data or not data["PropertyTable"].get("Properties"):
                return {"error": "Compound not found"}
            props = data["PropertyTable"]["Properties"][0]
            return {
                "smiles": props.get("IsomericSMILES") or smi,
                "iupac": props.get("IUPACName", ""),
                "formula": props.get("MolecularFormula", ""),
            }

        # Name search
        enc_name = urllib.parse.quote(name, safe="")
        cid_url = f"{PUBCHEM_BASE}/compound/name/{enc_name}/cids/JSON"
        cid_r = requests.get(cid_url, timeout=25)
        if cid_r.status_code != 200:
            return {"error": "Compound not found"}
        cid_json = cid_r.json()
        if "IdentifierList" not in cid_json or not cid_json["IdentifierList"].get("CID"):
            return {"error": "Compound not found"}

        cid = cid_json["IdentifierList"]["CID"][0]
        prop_url = (
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
            "IsomericSMILES,IUPACName,MolecularFormula/JSON"
        )
        prop_r = requests.get(prop_url, timeout=25)
        if prop_r.status_code != 200:
            return {"error": "PubChem failed"}
        pjson = prop_r.json()
        if "PropertyTable" not in pjson or not pjson["PropertyTable"].get("Properties"):
            return {"error": "PubChem failed"}
        props = pjson["PropertyTable"]["Properties"][0]
        smi_out = props.get("IsomericSMILES") or smi
        if not smi_out:
            return {"error": "Compound not found"}
        return {
            "smiles": smi_out,
            "iupac": props.get("IUPACName", ""),
            "formula": props.get("MolecularFormula", ""),
        }
    except requests.RequestException:
        return {"error": "PubChem failed"}
    except (KeyError, ValueError, TypeError):
        return {"error": "PubChem failed"}


# ---------------------------------------------------------------------------
# RDKit: 2D / 3D
# ---------------------------------------------------------------------------


def mol2img(mol: Chem.Mol) -> str:
    """Return SVG string for <img src='data:image/svg+xml;base64,...'> or inline."""
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(520, 360)
    drawer.drawOptions().padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def prepare_3d_mol(mol: Chem.Mol) -> Chem.Mol:
    """Add hydrogens, embed 3D coordinates, optimize, assign stereochemistry from 3D."""
    m = Chem.AddHs(Chem.Mol(mol))
    try:
        params = AllChem.ETKDGv3()
    except Exception:
        params = AllChem.ETKDG()
    params.randomSeed = 0xC0FFEE
    err = AllChem.EmbedMolecule(m, params)
    if err == -1:
        AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE)
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(m)
        except Exception:
            pass
    try:
        Chem.AssignStereochemistryFrom3D(m)
    except Exception:
        Chem.AssignStereochemistry(m, force=True, cleanIt=True)
    return m


def mol3d(mol: Chem.Mol) -> str:
    """Return MOL block with 3D coordinates for 3Dmol.js."""
    return Chem.MolToMolBlock(prepare_3d_mol(mol))


def explain_chirality(mol: Chem.Mol) -> List[Dict[str, Any]]:
    """
    Chiral centers with R/S (when available), neighbor priorities, and teaching text.
    """
    return _explain_chirality_m3d(prepare_3d_mol(mol))


def _explain_chirality_m3d(m3d: Chem.Mol) -> List[Dict[str, Any]]:
    """Build chirality explanations from an embedded 3D molecule."""
    try:
        centers_raw = Chem.FindMolChiralCenters(
            m3d,
            force=False,
            includeUnassigned=True,
            includeCIP=True,
            useLegacyImplementation=False,
        )
    except TypeError:
        centers_raw = Chem.FindMolChiralCenters(m3d, includeUnassigned=True)

    results: List[Dict[str, Any]] = []
    for atom_idx, stereo in centers_raw:
        atom = m3d.GetAtomWithIdx(atom_idx)
        neighbors = [n for n in atom.GetNeighbors()]
        # CIP-style priority: higher atomic number first; ties need deeper rules (simplified here)
        priority_list: List[Tuple[int, str, int]] = []
        for n in neighbors:
            priority_list.append((n.GetIdx(), n.GetSymbol(), n.GetAtomicNum()))
        priority_list.sort(key=lambda x: x[2], reverse=True)

        why = (
            "This atom is sp³-like with four different substituents, so it cannot "
            "have a plane of symmetry and is a stereogenic (chiral) center."
        )
        priority_expl = (
            "CIP priority assigns the highest atomic number directly attached to the "
            "chiral center first; ties are broken by comparing atoms at the next "
            "level outward (double/triple bonds count as duplicate attachments)."
        )
        rs_expl = (
            "Orient the molecule so the lowest-priority group points away. "
            "Trace 1→2→3: if the path is clockwise, the configuration is R; "
            "if counterclockwise, S (for the usual CIP ordering)."
        )
        if stereo == "?":
            rs_expl += (
                " (Stereochemistry could not be fully resolved from the generated 3D conformer.)"
            )

        results.append(
            {
                "atom_index": atom_idx,
                "element": atom.GetSymbol(),
                "configuration": stereo,
                "priority_neighbors": [
                    {"idx": i, "element": s, "Z": z} for i, s, z in priority_list
                ],
                "why_chiral": why,
                "priority_explanation": priority_expl,
                "rs_explanation": rs_expl,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Quiz bank
# ---------------------------------------------------------------------------


QUIZ_BANK: Dict[str, List[Dict[str, Any]]] = {
    "easy": [
        {
            "question": "What is a stereoisomer?",
            "options": [
                "Isomers with the same connectivity but different spatial arrangement",
                "Isomers with different molecular formulas",
                "Isomers that differ only in bond order",
                "Atoms with the same mass number",
            ],
            "correct": "Isomers with the same connectivity but different spatial arrangement",
            "explanation": "Stereoisomers share the same molecular formula and atom connectivity but differ in how atoms are arranged in 3D space.",
        },
        {
            "question": "A chiral center is typically:",
            "options": [
                "An sp³ carbon with four different groups",
                "Any carbon with a double bond",
                "An atom with only two substituents",
                "A hydrogen atom",
            ],
            "correct": "An sp³ carbon with four different groups",
            "explanation": "Classic tetrahedral carbon chirality requires four different substituents.",
        },
        {
            "question": "Enantiomers are:",
            "options": [
                "Mirror images that are not superimposable",
                "Isomers with different connectivity",
                "Cis/trans isomers in a ring",
                "Identical molecules in different solvents",
            ],
            "correct": "Mirror images that are not superimposable",
            "explanation": "Enantiomers are non-superimposable mirror images of each other.",
        },
        {
            "question": "A sigma (σ) bond is best described as:",
            "options": [
                "End-on overlap of orbitals along the internuclear axis",
                "Side-by-side overlap of p orbitals",
                "A hydrogen bond",
                "A purely ionic interaction",
            ],
            "correct": "End-on overlap of orbitals along the internuclear axis",
            "explanation": "σ bonds arise from head-on overlap of orbitals along the bond axis.",
        },
        {
            "question": "The bond angle in an sp-hybridized carbon is about:",
            "options": ["180°", "120°", "109.5°", "90°"],
            "correct": "180°",
            "explanation": "sp hybridization gives linear geometry with ~180° bond angles.",
        },
    ],
    "medium": [
        {
            "question": "Diastereomers are stereoisomers that:",
            "options": [
                "Are not mirror images of each other",
                "Are always mirror images",
                "Have different molecular formulas",
                "Cannot be separated",
            ],
            "correct": "Are not mirror images of each other",
            "explanation": "Diastereomers are stereoisomers that are not related as object and mirror image.",
        },
        {
            "question": "In CIP rules, the first point of difference is used when:",
            "options": [
                "Two atoms directly attached have the same atomic number",
                "A molecule has no chiral centers",
                "A bond is purely ionic",
                "The molecule is achiral",
            ],
            "correct": "Two atoms directly attached have the same atomic number",
            "explanation": "Tie-breaking walks outward along substituents until a difference in atomic number appears.",
        },
        {
            "question": "A π (pi) bond in an alkene is formed from:",
            "options": [
                "Side-by-side overlap of p orbitals",
                "Only s orbitals",
                "Pure hydrogen bonding",
                "Metal d-orbitals only",
            ],
            "correct": "Side-by-side overlap of p orbitals",
            "explanation": "The π bond in alkenes comes from side-by-side overlap of p orbitals perpendicular to the σ framework.",
        },
        {
            "question": "sp² hybridization corresponds to bond angles near:",
            "options": ["120°", "109.5°", "180°", "90°"],
            "correct": "120°",
            "explanation": "Trigonal planar sp² centers have ~120° angles between substituents.",
        },
        {
            "question": "If the lowest-priority group is pointed toward you in the R/S convention, you should:",
            "options": [
                "Invert the final rotation sense (clockwise ↔ counterclockwise)",
                "Ignore the lowest-priority group entirely",
                "Always assign R",
                "Assign Z/E instead",
            ],
            "correct": "Invert the final rotation sense (clockwise ↔ counterclockwise)",
            "explanation": "When the lowest-priority group is toward you, the apparent rotation is reversed relative to the standard view.",
        },
    ],
    "hard": [
        {
            "question": "Multiple bonds in CIP are treated as:",
            "options": [
                "Duplicate single bonds to each duplicated atom",
                "Ignored entirely",
                "Always highest priority regardless of atoms",
                "Equivalent to a single bond to a ghost atom",
            ],
            "correct": "Duplicate single bonds to each duplicated atom",
            "explanation": "CIP expands multiple bonds into duplicate single bonds to phantom atoms for priority comparison.",
        },
        {
            "question": "Meso compounds are:",
            "options": [
                "Achiral despite having stereocenters",
                "Always enantiomers",
                "Always optically active",
                "Isomers with different formulas",
            ],
            "correct": "Achiral despite having stereocenters",
            "explanation": "Meso compounds have internal symmetry that cancels optical activity even with stereocenters.",
        },
        {
            "question": "A molecule with a plane of symmetry and two stereocenters in (R,S) vs (S,R) may be:",
            "options": [
                "A meso form if the molecule is superimposable on its mirror image",
                "Always chiral",
                "Always two enantiomers",
                "Always optically active",
            ],
            "correct": "A meso form if the molecule is superimposable on its mirror image",
            "explanation": "Meso diastereomers have a plane of symmetry and are achiral.",
        },
        {
            "question": "Hybridization of sp³ corresponds to:",
            "options": [
                "Tetrahedral geometry (~109.5°)",
                "Linear geometry",
                "Trigonal planar geometry",
                "Square planar geometry",
            ],
            "correct": "Tetrahedral geometry (~109.5°)",
            "explanation": "sp³ hybridization combines one s and three p orbitals to give four orbitals toward tetrahedral angles.",
        },
        {
            "question": "Which statement about σ and π bonds in a double bond is most accurate?",
            "options": [
                "One σ plus one π; rotation is restricted by the π bond",
                "Two π bonds only",
                "Two σ bonds only",
                "No π bond",
            ],
            "correct": "One σ plus one π; rotation is restricted by the π bond",
            "explanation": "A carbon–carbon double bond is one σ and one π; π overlap must stay aligned, so rotation is hindered.",
        },
    ],
}


def get_quiz(level: str) -> List[Dict[str, Any]]:
    lvl = (level or "easy").lower().strip()
    if lvl not in QUIZ_BANK:
        return []
    return QUIZ_BANK[lvl]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(MAIN_HTML, version="1.0.0")


@app.route("/quiz", methods=["GET"])
def quiz_route():
    level = request.args.get("level", "easy")
    items = get_quiz(level)
    if not items:
        return jsonify({"error": "Invalid input"}), 400
    out = []
    for q in items:
        out.append(
            {
                "question": q["question"],
                "options": q["options"],
                "correct_answer": q["correct"],
                "explanation": q["explanation"],
            }
        )
    return jsonify({"level": level, "questions": out})


@app.route("/analyze", methods=["POST"])
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("compound_name") or payload.get("name") or "").strip()
    smi_in = (payload.get("smiles") or "").strip()

    if not name and not smi_in:
        return jsonify({"error": "Invalid input"}), 400

    pub: Dict[str, Any] = {}
    if smi_in:
        mol_try = Chem.MolFromSmiles(smi_in)
        if mol_try is None:
            return jsonify({"error": "Invalid input"}), 400
        pub = fetch_pubchem(smiles_input=smi_in)
        if "error" in pub:
            return jsonify(pub), 400
        smiles = pub["smiles"]
    else:
        pub = fetch_pubchem(compound_name=name)
        if "error" in pub:
            return jsonify(pub), 400
        smiles = pub["smiles"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid input"}), 400

    svg = mol2img(mol)
    m3d = prepare_3d_mol(mol)
    molblock = Chem.MolToMolBlock(m3d)
    chiral = _explain_chirality_m3d(m3d)

    return jsonify(
        {
            "pubchem": {
                "smiles": pub.get("smiles", smiles),
                "iupac": pub.get("iupac", ""),
                "formula": pub.get("formula", ""),
            },
            "svg_2d": svg,
            "molblock_3d": molblock,
            "chiral_centers": chiral,
            "chiral_indices": [c["atom_index"] for c in chiral],
        }
    )


# ---------------------------------------------------------------------------
# Single-page UI (glassmorphism + tabs + 3Dmol.js)
# ---------------------------------------------------------------------------


MAIN_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ChemExplorer X</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    :root {
      --bg0: #070b12;
      --bg1: #0d1320;
      --glass: rgba(255,255,255,0.06);
      --glass2: rgba(255,255,255,0.1);
      --stroke: rgba(0,255,255,0.18);
      --text: #e8eefc;
      --muted: #94a3b8;
      --neon-cyan: #22d3ee;
      --neon-mint: #34d399;
      --neon-violet: #a78bfa;
      --danger: #fb7185;
      --ok: #4ade80;
      --radius: 18px;
      --shadow: 0 20px 60px rgba(0,0,0,0.45);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: 'DM Sans', system-ui, sans-serif;
      color: var(--text);
      background:
        radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,0.18), transparent 55%),
        radial-gradient(900px 500px at 90% 0%, rgba(167,139,250,0.16), transparent 50%),
        linear-gradient(165deg, var(--bg0), var(--bg1) 40%, #0a0f18);
    }
    a { color: var(--neon-cyan); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 80px; }
    .hero {
      text-align: center;
      padding: 36px 8px 28px;
    }
    .badge {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--stroke);
      background: var(--glass);
      backdrop-filter: blur(14px);
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.02em;
    }
    h1 {
      margin: 18px 0 10px;
      font-size: clamp(2rem, 4vw, 2.75rem);
      font-weight: 700;
      letter-spacing: -0.03em;
      background: linear-gradient(90deg, var(--neon-mint), var(--neon-cyan), var(--neon-violet));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .sub {
      color: var(--muted);
      max-width: 640px;
      margin: 0 auto;
      line-height: 1.55;
    }
    .panel {
      margin-top: 26px;
      padding: 22px;
      border-radius: var(--radius);
      border: 1px solid rgba(255,255,255,0.08);
      background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
    }
    .input-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      justify-content: center;
      align-items: center;
    }
    label { font-size: 13px; color: var(--muted); display: block; margin-bottom: 6px; }
    .field {
      position: relative;
      min-width: 240px;
      flex: 1 1 240px;
    }
    input[type="text"] {
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.1);
      background: rgba(0,0,0,0.35);
      color: var(--text);
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
    }
    input[type="text"]:focus {
      border-color: rgba(34,211,238,0.55);
      box-shadow: 0 0 0 3px rgba(34,211,238,0.12);
    }
    .btn {
      border: none;
      cursor: pointer;
      padding: 13px 26px;
      border-radius: 999px;
      font-weight: 600;
      letter-spacing: 0.02em;
      color: #041016;
      background: linear-gradient(120deg, var(--neon-mint), var(--neon-cyan));
      box-shadow: 0 12px 30px rgba(34,211,238,0.25);
      transition: transform 0.15s ease, filter 0.2s;
    }
    .btn:hover { transform: translateY(-1px); filter: brightness(1.05); }
    .btn:active { transform: translateY(0); }
    .btn:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }

    .tabs {
      display: none;
      margin-top: 22px;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: center;
    }
    .tabs.visible { display: flex; }
    .tab {
      padding: 10px 18px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.1);
      background: rgba(0,0,0,0.25);
      color: var(--muted);
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 14px;
    }
    .tab:hover {
      border-color: rgba(34,211,238,0.35);
      color: var(--text);
      transform: translateY(-1px);
    }
    .tab.active {
      border-color: rgba(34,211,238,0.55);
      color: var(--text);
      background: linear-gradient(145deg, rgba(34,211,238,0.18), rgba(167,139,250,0.12));
      box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }

    .section { display: none; margin-top: 18px; }
    .section.active { display: block; animation: fade 0.35s ease; }
    @keyframes fade { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 18px;
    }
    .card {
      background: rgba(0,0,0,0.28);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: var(--radius);
      padding: 18px;
      backdrop-filter: blur(12px);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card:hover {
      border-color: rgba(34,211,238,0.22);
      box-shadow: 0 16px 40px rgba(0,0,0,0.35);
    }
    .card h3 {
      margin: 0 0 12px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      font-weight: 600;
    }
    .mol2d-wrap {
      background: radial-gradient(circle at 30% 20%, rgba(34,211,238,0.08), transparent 50%);
      border-radius: 14px;
      padding: 12px;
      border: 1px solid rgba(255,255,255,0.06);
      text-align: center;
    }
    .mol2d-wrap svg { max-width: 100%; height: auto; }
    .viewer3d {
      height: 380px;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(34,211,238,0.15);
      background: radial-gradient(circle at 50% 50%, #0a0f18, #0b0f14);
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      justify-content: center;
      align-items: center;
      margin-top: 12px;
    }
    .toolbar button, .toolbar select {
      padding: 8px 14px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      color: var(--text);
      cursor: pointer;
      font-size: 13px;
    }
    .toolbar button:hover { border-color: rgba(34,211,238,0.35); }

    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
      font-size: 13px;
      color: var(--muted);
    }
    .pill {
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.1);
      background: rgba(255,255,255,0.04);
    }
    .pill strong { color: var(--text); font-weight: 600; }

    .chiral-card {
      margin-bottom: 14px;
      padding: 14px;
      border-radius: 14px;
      border: 1px solid rgba(167,139,250,0.16);
      background: linear-gradient(145deg, rgba(167,139,250,0.08), rgba(0,0,0,0.2));
    }
    .chiral-card h4 { margin: 0 0 8px; font-size: 16px; color: var(--neon-violet); }
    .chiral-card p { margin: 8px 0; line-height: 1.55; color: #cbd5e1; font-size: 14px; }

    .theory {
      margin-top: 18px;
      padding: 20px;
      border-radius: var(--radius);
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(0,0,0,0.35);
      line-height: 1.65;
      color: #cbd5e1;
      font-size: 14px;
    }
    .theory h2 { margin-top: 0; color: var(--text); font-size: 1.15rem; }
    .theory h3 { margin-top: 18px; color: var(--neon-cyan); font-size: 1rem; }
    .theory ul { padding-left: 1.2rem; }

    .quiz-head {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }
    .score {
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: var(--neon-mint);
    }
    .quiz-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }
    .quiz-controls select, .quiz-controls button {
      padding: 8px 12px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      color: var(--text);
      cursor: pointer;
    }
    .q-block {
      margin-bottom: 16px;
      padding: 16px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(0,0,0,0.25);
    }
    .q-block h4 { margin: 0 0 10px; font-size: 15px; color: var(--text); }
    .opt {
      display: block;
      width: 100%;
      text-align: left;
      margin: 6px 0;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .opt:hover { border-color: rgba(34,211,238,0.35); }
    .opt.correct { border-color: rgba(74,222,128,0.6); background: rgba(74,222,128,0.12); }
    .opt.wrong { border-color: rgba(251,113,133,0.6); background: rgba(251,113,133,0.1); }
    .feedback {
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .feedback strong { color: var(--neon-cyan); }

    .learn-more {
      margin-top: 22px;
      padding: 18px;
      border-radius: var(--radius);
      border: 1px dashed rgba(34,211,238,0.25);
      background: rgba(34,211,238,0.05);
    }
    .learn-more h3 { margin: 0 0 10px; font-size: 15px; color: var(--text); }
    .learn-more ul { margin: 0; padding-left: 1.2rem; color: var(--muted); }

    footer {
      margin-top: 40px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
    }
    .alert {
      display: none;
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid rgba(251,113,133,0.45);
      background: rgba(251,113,133,0.1);
      color: #fecdd3;
      font-size: 14px;
    }
    .alert.show { display: block; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="badge">Flask · RDKit · PubChem · 3Dmol.js</div>
      <h1>ChemExplorer X</h1>
      <p class="sub">Analyze compounds from PubChem, explore 2D/3D structure, learn stereochemistry with guided theory, and test your knowledge with adaptive quizzes.</p>
    </div>

    <div class="panel">
      <div class="input-row">
        <div class="field">
          <label for="compound_name">Compound name</label>
          <input type="text" id="compound_name" placeholder="e.g. caffeine, aspirin" autocomplete="off" />
        </div>
        <div class="field">
          <label for="smiles_in">SMILES</label>
          <input type="text" id="smiles_in" placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O" autocomplete="off" />
        </div>
        <div style="align-self:flex-end;">
          <button class="btn" id="analyzeBtn" type="button">Analyze</button>
        </div>
      </div>
      <div id="alertBox" class="alert" role="alert"></div>
    </div>

    <div id="tabs" class="tabs">
      <button class="tab active" data-tab="structure" type="button">Structure</button>
      <button class="tab" data-tab="chirality" type="button">Chirality &amp; Theory</button>
      <button class="tab" data-tab="quiz" type="button">Quiz</button>
    </div>

    <div id="meta" class="meta-row" style="display:none;"></div>

    <div id="section-structure" class="section">
      <div class="grid-2">
        <div class="card">
          <h3>2D structure (RDKit SVG)</h3>
          <div id="mol2d" class="mol2d-wrap"></div>
        </div>
        <div class="card">
          <h3>3D interactive viewer</h3>
          <div id="viewer3d" class="viewer3d"></div>
          <div class="toolbar">
            <button type="button" id="spinToggle">Spin: on</button>
            <label style="font-size:13px;color:var(--muted);">Style</label>
            <select id="styleSelect">
              <option value="stick">Stick</option>
              <option value="sphere">Sphere</option>
              <option value="line">Wireframe</option>
            </select>
            <button type="button" id="zoomIn">Zoom +</button>
            <button type="button" id="zoomOut">Zoom −</button>
          </div>
        </div>
      </div>
    </div>

    <div id="section-chirality" class="section">
      <div class="grid-2">
        <div class="card">
          <h3>Chiral centers (RDKit + 3D)</h3>
          <div id="chiralList"></div>
        </div>
        <div class="card">
          <h3>3D — chiral highlights</h3>
          <div id="viewer3d_chiral" class="viewer3d"></div>
          <p style="font-size:13px;color:var(--muted);margin-top:10px;">Chiral centers are emphasized as colored spheres; the rest stays in stick style.</p>
        </div>
      </div>
      <div class="theory">
        <h2>Theory — stereochemistry &amp; bonding</h2>
        <p><strong>Stereoisomers</strong> have the same molecular formula and connectivity but differ in the spatial arrangement of atoms. They are not constitutional isomers.</p>
        <h3>Types of stereoisomers</h3>
        <ul>
          <li><strong>Enantiomers</strong> are non-superimposable mirror images (like left and right hands).</li>
          <li><strong>Diastereomers</strong> are stereoisomers that are not mirror images of each other (e.g., cis vs trans alkenes, or multiple stereocenters with different configurations).</li>
        </ul>
        <h3>Chiral vs achiral</h3>
        <ul>
          <li><strong>Chiral</strong> molecules (or centers) lack an improper rotation axis / plane of symmetry that would make the object superimposable on its mirror image.</li>
          <li><strong>Achiral</strong> molecules often have symmetry elements that make them superimposable on their mirror image.</li>
        </ul>
        <h3>R / S configuration</h3>
        <p><strong>Cahn–Ingold–Prelog (CIP)</strong> assigns priorities 1–4 to substituents at a stereocenter. With the lowest-priority group oriented away, trace 1→2→3: clockwise = <strong>R</strong> (rectus), counterclockwise = <strong>S</strong> (sinister).</p>
        <h3>CIP rules (summary)</h3>
        <ul>
          <li><strong>Atomic number priority</strong>: higher atomic number at the first point of attachment wins.</li>
          <li><strong>Tie-breaking</strong>: move outward atom-by-atom until a difference is found.</li>
          <li><strong>Multiple bonds</strong>: treated as duplicate single bonds to duplicate atoms (ghost atoms).</li>
          <li><strong>Orientation</strong>: if the lowest-priority group points toward you, invert the sense of rotation when assigning R/S.</li>
        </ul>
        <h3>Hybridization</h3>
        <ul>
          <li><strong>sp</strong> — linear, ~180° angles (two electron domains).</li>
          <li><strong>sp²</strong> — trigonal planar, ~120° angles (three domains).</li>
          <li><strong>sp³</strong> — tetrahedral, ~109.5° angles (four domains).</li>
        </ul>
        <h3>Bond types</h3>
        <ul>
          <li><strong>σ (sigma)</strong> bond: end-on orbital overlap along the internuclear axis.</li>
          <li><strong>π (pi)</strong> bond: side-by-side overlap of p orbitals, present in multiple bonds (along with one σ).</li>
        </ul>
      </div>
    </div>

    <div id="section-quiz" class="section">
      <div class="card">
        <div class="quiz-head">
          <div>
            <h3 style="margin:0;">Quiz</h3>
            <span class="score" id="scoreDisplay">Score: 0 / 0</span>
          </div>
          <div class="quiz-controls">
            <label>Level</label>
            <select id="quizLevel">
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
            <label>View</label>
            <select id="quizView">
              <option value="one">One question at a time</option>
              <option value="all">All questions</option>
            </select>
            <button type="button" id="reloadQuiz">Reload quiz</button>
          </div>
        </div>
        <div id="quizArea"></div>
        <div id="quizFinal" class="theory" style="display:none;"></div>
      </div>
    </div>

    <div class="learn-more">
      <h3>Learn more</h3>
      <ul>
        <li><a href="https://pubchem.ncbi.nlm.nih.gov/" target="_blank" rel="noopener noreferrer">PubChem — compound database</a></li>
        <li><a href="https://chem.libretexts.org/" target="_blank" rel="noopener noreferrer">LibreTexts Chemistry</a></li>
        <li><a href="https://www.khanacademy.org/science/organic-chemistry" target="_blank" rel="noopener noreferrer">Khan Academy — Organic Chemistry</a></li>
      </ul>
    </div>

    <footer>
      ChemExplorer X · Version {{ version }} · Educational chemistry learning platform
    </footer>
  </div>

  <script>
  (function() {
    const $ = (id) => document.getElementById(id);
    let viewer = null;
    let viewerChiral = null;
    let spinEnabled = true;
    let currentStyle = 'stick';
    let chiralIdx = [];
    let quizData = [];
    let quizLevel = 'easy';
    let quizView = 'one';
    let qIndex = 0;
    let score = 0;
    let answered = 0;

    function showAlert(msg) {
      const el = $('alertBox');
      el.textContent = msg;
      el.classList.add('show');
    }
    function hideAlert() {
      $('alertBox').classList.remove('show');
    }

    function switchTab(name) {
      document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === name);
      });
      document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
      const map = {
        structure: 'section-structure',
        chirality: 'section-chirality',
        quiz: 'section-quiz'
      };
      $(map[name]).classList.add('active');
      if (name === 'chirality' && viewerChiral) {
        viewerChiral.resize();
        viewerChiral.render();
      }
      if (name === 'structure' && viewer) {
        viewer.resize();
        viewer.render();
      }
    }

    document.querySelectorAll('.tab').forEach(t => {
      t.addEventListener('click', () => switchTab(t.dataset.tab));
    });

    function applyStyle(v, targetViewer) {
      const style = {};
      if (v === 'stick') style.stick = { radius: 0.13 };
      else if (v === 'sphere') style.sphere = { scale: 0.23 };
      else style.line = { radius: 0.05 };
      targetViewer.setStyle({}, style);
    }

    function highlightChiral(v, indices) {
      applyStyle(currentStyle, v);
      const colors = ['#22d3ee', '#a78bfa', '#34d399', '#fb7185', '#fbbf24'];
      indices.forEach((idx, i) => {
        v.setStyle({ atom: [idx] }, { sphere: { scale: 0.35, color: colors[i % colors.length] } });
      });
      v.zoomTo();
      v.render();
    }

    function initViewers(molblock, indices) {
      const el = $('viewer3d');
      el.innerHTML = '';
      viewer = $3Dmol.createViewer(el, { backgroundColor: '#0b0f14' });
      viewer.addModel(molblock, 'mol');
      applyStyle(currentStyle, viewer);
      viewer.zoomTo();
      viewer.spin(spinEnabled);
      viewer.render();

      const el2 = $('viewer3d_chiral');
      el2.innerHTML = '';
      viewerChiral = $3Dmol.createViewer(el2, { backgroundColor: '#0b0f14' });
      viewerChiral.addModel(molblock, 'mol');
      highlightChiral(viewerChiral, indices);
      viewerChiral.spin(spinEnabled);
    }

    $('spinToggle').addEventListener('click', () => {
      spinEnabled = !spinEnabled;
      $('spinToggle').textContent = 'Spin: ' + (spinEnabled ? 'on' : 'off');
      if (viewer) { viewer.spin(spinEnabled); viewer.render(); }
      if (viewerChiral) { viewerChiral.spin(spinEnabled); viewerChiral.render(); }
    });
    $('styleSelect').addEventListener('change', (e) => {
      currentStyle = e.target.value;
      if (viewer) {
        applyStyle(currentStyle, viewer);
        viewer.render();
      }
      if (viewerChiral && chiralIdx.length) {
        highlightChiral(viewerChiral, chiralIdx);
        viewerChiral.spin(spinEnabled);
      }
    });
    $('zoomIn').addEventListener('click', () => {
      if (viewer) { viewer.zoom(1.2); viewer.render(); }
    });
    $('zoomOut').addEventListener('click', () => {
      if (viewer) { viewer.zoom(0.8); viewer.render(); }
    });

    $('analyzeBtn').addEventListener('click', async () => {
      hideAlert();
      const name = $('compound_name').value.trim();
      const smiles = $('smiles_in').value.trim();
      if (!name && !smiles) {
        showAlert('Invalid input');
        return;
      }
      $('analyzeBtn').disabled = true;
      try {
        const res = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ compound_name: name, smiles: smiles })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showAlert(data.error || 'PubChem failed');
          return;
        }
        $('tabs').classList.add('visible');
        $('meta').style.display = 'flex';
        $('meta').innerHTML = `
          <span class="pill">Formula: <strong>${escapeHtml(data.pubchem.formula || '—')}</strong></span>
          <span class="pill">IUPAC: <strong>${escapeHtml(data.pubchem.iupac || '—')}</strong></span>
          <span class="pill">SMILES: <strong>${escapeHtml(data.pubchem.smiles || '')}</strong></span>
        `;
        $('mol2d').innerHTML = data.svg_2d;
        chiralIdx = data.chiral_indices || [];
        initViewers(data.molblock_3d, chiralIdx);

        const list = $('chiralList');
        if (!data.chiral_centers || data.chiral_centers.length === 0) {
          list.innerHTML = '<p style="color:var(--muted);">No tetrahedral chiral centers detected in this structure (or stereochemistry could not be assigned).</p>';
        } else {
          list.innerHTML = data.chiral_centers.map((c, i) => `
            <div class="chiral-card">
              <h4>Center ${i + 1} — Atom ${c.atom_index} (${escapeHtml(c.element)}) · ${escapeHtml(String(c.configuration))}</h4>
              <p><strong>Why it is chiral:</strong> ${escapeHtml(c.why_chiral)}</p>
              <p><strong>Priority (CIP, simplified):</strong> ${escapeHtml(c.priority_explanation)}</p>
              <p><strong>Neighbor order (atomic number):</strong> ${c.priority_neighbors.map(n => `${n.element} (Z=${n.Z})`).join(' &gt; ')}</p>
              <p><strong>R / S:</strong> ${escapeHtml(c.rs_explanation)}</p>
            </div>
          `).join('');
        }
        switchTab('structure');
        loadQuiz();
      } catch (e) {
        showAlert('PubChem failed');
      } finally {
        $('analyzeBtn').disabled = false;
      }
    });

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, c => ({
        '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
      }[c]));
    }

    async function loadQuiz() {
      quizLevel = $('quizLevel').value;
      quizView = $('quizView').value;
      const res = await fetch('/quiz?level=' + encodeURIComponent(quizLevel));
      const data = await res.json();
      if (!res.ok) {
        $('quizArea').innerHTML = '<p style="color:var(--danger);">Could not load quiz.</p>';
        return;
      }
      quizData = data.questions || [];
      qIndex = 0;
      score = 0;
      answered = 0;
      $('quizFinal').style.display = 'none';
      renderQuiz();
    }

    function renderQuiz() {
      $('scoreDisplay').textContent = 'Score: ' + score + ' / ' + answered + ' (max ' + quizData.length + ')';
      const area = $('quizArea');
      area.innerHTML = '';
      if (!quizData.length) {
        area.innerHTML = '<p style="color:var(--muted);">No questions.</p>';
        return;
      }
      const qs = quizView === 'one' ? [quizData[qIndex]] : quizData;
      qs.forEach((q, qi) => {
        const globalIdx = quizView === 'one' ? qIndex : (quizData.indexOf(q));
        const block = document.createElement('div');
        block.className = 'q-block';
        block.dataset.idx = String(globalIdx);
        block.innerHTML = `
          <h4>${escapeHtml(q.question)}</h4>
          ${q.options.map((o) => `
            <button type="button" class="opt" data-ans="${escapeHtml(o)}">${escapeHtml(o)}</button>
          `).join('')}
          <div class="feedback" id="fb-${globalIdx}" style="display:none;"></div>
        `;
        block.querySelectorAll('.opt').forEach(btn => {
          btn.addEventListener('click', () => onAnswer(globalIdx, q, btn));
        });
        area.appendChild(block);
      });
      if (quizView === 'one' && quizData.length > 1) {
        const nav = document.createElement('div');
        nav.style.marginTop = '12px';
        nav.innerHTML = `
          <button type="button" class="opt" style="display:inline-block;width:auto;margin-right:8px;" id="prevQ">Previous</button>
          <button type="button" class="opt" style="display:inline-block;width:auto;" id="nextQ">Next</button>
        `;
        area.appendChild(nav);
        $('prevQ').addEventListener('click', () => {
          if (qIndex > 0) { qIndex--; renderQuiz(); }
        });
        $('nextQ').addEventListener('click', () => {
          if (qIndex < quizData.length - 1) { qIndex++; renderQuiz(); }
        });
      }
    }

    function onAnswer(idx, q, btn) {
      const block = btn.closest('.q-block');
      if (block.dataset.answered === '1') return;
      block.dataset.answered = '1';
      const chosen = btn.dataset.ans;
      const ok = chosen === q.correct_answer;
      if (ok) score++;
      answered++;
      block.querySelectorAll('.opt').forEach(b => {
        b.disabled = true;
        if (b.dataset.ans === q.correct_answer) b.classList.add('correct');
        else if (b === btn && !ok) b.classList.add('wrong');
      });
      const fb = $('fb-' + idx);
      fb.style.display = 'block';
      fb.innerHTML = '<strong>' + (ok ? 'Correct.' : 'Incorrect.') + '</strong> ' + escapeHtml(q.explanation);
      $('scoreDisplay').textContent = 'Score: ' + score + ' / ' + answered + ' (max ' + quizData.length + ')';
      if (answered === quizData.length) {
        $('quizFinal').style.display = 'block';
        $('quizFinal').innerHTML = '<h2>Final score</h2><p>You scored <strong>' + score + '</strong> out of <strong>' + quizData.length + '</strong> on <strong>' + quizLevel + '</strong> level.</p>';
      }
    }

    $('quizLevel').addEventListener('change', loadQuiz);
    $('quizView').addEventListener('change', loadQuiz);
    $('reloadQuiz').addEventListener('click', loadQuiz);
  })();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
