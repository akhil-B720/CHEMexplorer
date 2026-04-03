def fetch_pubchem(
    compound_name: Optional[str] = None,
    smiles_input: Optional[str] = None,
) -> Dict[str, Any]:
    name = (compound_name or "").strip()
    smi = (smiles_input or "").strip()

    if not name and not smi:
        return {"error": "Invalid input"}

    try:
        # ---------------- SMILES INPUT ----------------
        if smi:
            enc = urllib.parse.quote(smi, safe="")

            url = f"{PUBCHEM_BASE}/compound/smiles/{enc}/property/IsomericSMILES,IUPACName,MolecularFormula/JSON"

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

        # ---------------- NAME INPUT ----------------
        enc_name = urllib.parse.quote(name, safe="")

        cid_url = f"{PUBCHEM_BASE}/compound/name/{enc_name}/cids/JSON"
        cid_r = requests.get(cid_url, timeout=25)

        if cid_r.status_code != 200:
            return {"error": "Compound not found"}

        cid_json = cid_r.json()

        if "IdentifierList" not in cid_json or not cid_json["IdentifierList"].get("CID"):
            return {"error": "Compound not found"}

        cid = cid_json["IdentifierList"]["CID"][0]

        prop_url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/IsomericSMILES,IUPACName,MolecularFormula/JSON"
        prop_r = requests.get(prop_url, timeout=25)

        if prop_r.status_code != 200:
            return {"error": "PubChem failed"}

        pjson = prop_r.json()

        if "PropertyTable" not in pjson or not pjson["PropertyTable"].get("Properties"):
            return {"error": "PubChem failed"}

        props = pjson["PropertyTable"]["Properties"][0]

        return {
            "smiles": props.get("IsomericSMILES"),
            "iupac": props.get("IUPACName", ""),
            "formula": props.get("MolecularFormula", ""),
        }

    except requests.RequestException:
        return {"error": "PubChem failed"}
    except (KeyError, ValueError, TypeError):
        return {"error": "PubChem failed"}
