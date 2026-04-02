# ===========================
# ChemExplorer PRO (Learning Version)
# ===========================

from flask import Flask, request, jsonify, render_template_string
import requests, base64, random

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

app = Flask(__name__)

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# ---------------------------
# PubChem
# ---------------------------
def fetch_pubchem(name):
    try:
        cid = requests.get(f"{PUBCHEM}/compound/name/{name}/cids/JSON").json()["IdentifierList"]["CID"][0]

        props = requests.get(
            f"{PUBCHEM}/compound/cid/{cid}/property/IsomericSMILES,IUPACName,MolecularFormula/JSON"
        ).json()["PropertyTable"]["Properties"][0]

        return {
            "smiles": props["IsomericSMILES"],
            "iupac": props["IUPACName"],
            "formula": props["MolecularFormula"]
        }
    except:
        return {"error": "PubChem failed"}

# ---------------------------
# 2D Image
# ---------------------------
def mol2img(mol):
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

# ---------------------------
# 3D Model
# ---------------------------
def mol3d(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    return Chem.MolToMolBlock(mol)

# ---------------------------
# Chemistry Analysis
# ---------------------------
def hybridization(mol):
    return [(a.GetIdx(), str(a.GetHybridization())) for a in mol.GetAtoms()]

def bonds(mol):
    return [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())) for b in mol.GetBonds()]

def functional_groups(mol):
    groups = []
    if mol.HasSubstructMatch(Chem.MolFromSmarts("CO")):
        groups.append("Alcohol")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C=O")):
        groups.append("Carbonyl")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1")):
        groups.append("Aromatic Ring")
    return groups

# ---------------------------
# Chirality Explanation
# ---------------------------
def explain_chirality(mol):
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    data = []

    for idx, config in centers:
        atom = mol.GetAtomWithIdx(idx)
        neighbors = atom.GetNeighbors()

        priority = sorted(
            [(n.GetSymbol(), n.GetAtomicNum()) for n in neighbors],
            key=lambda x: x[1],
            reverse=True
        )

        data.append({
            "atom": idx,
            "element": atom.GetSymbol(),
            "config": config,
            "priority": priority,
            "reason": "Higher atomic number = higher priority"
        })

    return data

# ---------------------------
# Dynamic Quiz
# ---------------------------
def generate_quiz(mol, chiral):
    q = []

    q.append({
        "q": "How many chiral centers?",
        "options": ["0", "1", "2", "3"],
        "answer": str(len(chiral))
    })

    atom = mol.GetAtomWithIdx(0)
    q.append({
        "q": f"Hybridization of atom 0 ({atom.GetSymbol()})?",
        "options": ["SP", "SP2", "SP3"],
        "answer": str(atom.GetHybridization())
    })

    if mol.GetNumBonds():
        bond = mol.GetBondWithIdx(0)
        q.append({
            "q": "Bond type between first atoms?",
            "options": ["SINGLE", "DOUBLE", "TRIPLE"],
            "answer": str(bond.GetBondType())
        })

    return random.sample(q, min(3, len(q)))

# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    name = data.get("name")
    smiles = data.get("smiles")

    if name:
        pub = fetch_pubchem(name)
        if "error" in pub:
            return jsonify(pub)
        smiles = pub["smiles"]
    else:
        pub = {}

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return jsonify({"error": "Invalid input"})

    chiral = explain_chirality(mol)

    return jsonify({
        "img": mol2img(mol),
        "molblock": mol3d(mol),
        "hyb": hybridization(mol),
        "bonds": bonds(mol),
        "groups": functional_groups(mol),
        "chiral": chiral,
        "quiz": generate_quiz(mol, chiral),
        "pub": pub
    })

# ---------------------------
# FRONTEND (FIXED)
# ---------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>ChemExplorer X</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap" rel="stylesheet">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>

<style>
body {
  font-family: 'Poppins', sans-serif;
  margin:0;
  background: radial-gradient(circle at top, #0f172a, #020617);
  color: #e2e8f0;
}
.header {
  text-align:center;
  padding:20px;
  font-size:28px;
  font-weight:700;
  background: linear-gradient(90deg,#22c55e,#06b6d4);
  -webkit-background-clip: text;
  color: transparent;
}
.search-box {
  display:flex;
  justify-content:center;
  gap:10px;
  margin:20px;
}
input {
  padding:12px;
  border-radius:10px;
  border:none;
  width:220px;
  background:#0f172a;
  color:white;
}
button {
  padding:12px;
  border:none;
  border-radius:10px;
  background: linear-gradient(45deg,#22c55e,#06b6d4);
  color:white;
  cursor:pointer;
}
.grid {
  display:grid;
  grid-template-columns: repeat(auto-fit,minmax(320px,1fr));
  gap:20px;
  padding:20px;
}
.card {
  background: rgba(255,255,255,0.05);
  border-radius:20px;
  padding:20px;
}
</style>
</head>

<body>

<div class="header">🧪 ChemExplorer X</div>

<div class="search-box">
<input id="name" placeholder="Compound name">
<input id="smiles" placeholder="SMILES">
<button onclick="run()">Analyze</button>
</div>

<div id="out"></div>

<script>

let v, spinning=true;

async function run(){
let name=document.getElementById("name").value
let smiles=document.getElementById("smiles").value

let r=await fetch("/analyze",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({name,smiles})
})

let d=await r.json()

if(d.error){alert(d.error);return}

document.getElementById("out").innerHTML=`
<div class="grid">

<div class="card">
<h3>Structure</h3>
<img src="${d.img}" style="width:100%">
<div id="v" style="height:300px"></div>
</div>

<div class="card">
<h3>Chirality</h3>
${d.chiral.map(c=>`
<p>
Atom ${c.atom} (${c.element}) → <b>${c.config}</b><br>
Priority: ${c.priority.map(p=>p[0]).join(" > ")}<br>
Reason: ${c.reason}
</p>
`).join("")}
</div>

<div class="card">
<h3>Quiz</h3>
${d.quiz.map(q=>`
<p>${q.q}</p>
${q.options.map(o=>`<button onclick="check('${o}','${q.answer}')">${o}</button>`).join("")}
`).join("")}
</div>

</div>
`

v=$3Dmol.createViewer("v")
v.addModel(d.molblock,"mol")
v.setStyle({}, {stick:{}})
v.zoomTo()
v.spin(true)
v.render()
}

function check(a,b){
alert(a==b?"Correct":"Wrong")
}

</script>

</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)