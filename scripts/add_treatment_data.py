"""Enrich data/diseases.json with actionable treatment + pesticide guidance.

Provenance: each record already cites university-extension sources (Cornell,
UMass, Penn State, Oregon State, etc.). The treatment steps and active
ingredients added here follow the guidance those extension services and
equivalent plant-protection authorities (e.g., FAO/CIMMYT for maize) have
long published for these diseases; they are curated, generic, and
region-agnostic (active ingredients rather than brand names). Viral diseases
and citrus greening are marked incurable because no chemical cure exists.
This is educational reference data, not a prescribing system — the UI shows a
label-law/extension disclaimer alongside it.

Run once from the repo root:  python scripts/add_treatment_data.py
Idempotent: re-running re-applies the same content.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISEASES_PATH = ROOT / "data" / "diseases.json"

# {record id: {"treatment": [...], "pesticides": [...], "incurable"?: True}}
TREATMENT: dict[str, dict] = {
    "apple-apple-scab": {
        "treatment": [
            "Sanitation first: rake and destroy fallen leaves and remove mummy fruit — most inoculum overwinters on ground debris.",
            "Prune for an open canopy so leaves dry quickly after rain.",
            "Apply a protective fungicide program from green tip through the primary spring infection period; continue cover sprays while scab is active.",
            "Plant resistant cultivars for future plantings.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["captan", "myclobutanil", "dodine", "sulfur", "thiophanate-methyl"],
                "application": "Start at green tip and maintain cover sprays (typically every 7-14 days per label, shortened after rain) through the primary infection period. Rotate modes of action between groups and observe the pre-harvest interval.",
            }
        ],
    },
    "apple-black-rot": {
        "treatment": [
            "Prune out cankers, dead wood, and branch stubs during dry weather — branch cankers are the main inoculum source.",
            "Remove and destroy all mummified fruit and fallen leaves.",
            "Avoid tree wounds; manage bird and rodent damage that creates entry points.",
            "Apply fungicide during the growing season where the disease has been active, and protect fresh pruning wounds.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["myclobutanil", "captan", "thiophanate-methyl", "pyraclostrobin", "boscalid"],
                "application": "Spray at pink and again at petal fall where black rot or scab pressure exists, with follow-up cover sprays per label during warm, wet weather; dormant applications for canker control where the label allows.",
            }
        ],
    },
    "apple-cedar-apple-rust": {
        "treatment": [
            "Remove the alternate host where practical: cut down juniper/cedar trees within a few hundred metres — the fungus cannot complete its life cycle without both hosts.",
            "Remove galls from remaining junipers and prune out infected apple twigs.",
            "Collect and destroy fallen infected leaves and fruit.",
            "Apply protective fungicide on apple during the spring infection window (teliospore release follows rainy weather).",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["myclobutanil", "sulfur", "copper"],
                "application": "Myclobutanil is the most effective option: start at pink and continue per label through roughly petal fall plus 2-3 weeks. Sulfur and copper are protectants with shorter residual activity. Observe the pre-harvest interval.",
            }
        ],
    },
    "cherry-powdery-mildew": {
        "treatment": [
            "Prune for an open canopy and avoid excessive nitrogen fertilisation — succulent growth is most susceptible.",
            "Apply fungicide from shuck split through the summer where pressure is high (warm days with humid nights favour the disease).",
            "Prune out and destroy heavily infected shoots where practical.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["myclobutanil", "triflumizole", "sulfur", "potassium bicarbonate"],
                "application": "Cover sprays from shuck split at label intervals; rotate the systemic groups (myclobutanil, triflumizole) with sulfur or potassium bicarbonate to slow resistance. Sulfur can burn foliage/fruit at high heat, and pre-harvest intervals matter on ripe fruit.",
            }
        ],
    },
    "corn-common-rust": {
        "treatment": [
            "Plant resistant hybrids — the most economical control; resistance genes are widely available.",
            "Scout from early vegetative stages; rust usually appears first on lower leaves as cinnamon-brown pustules.",
            "Fungicide is rarely needed for field corn grown for grain; consider it for seed corn or sweet corn when disease is actively progressing before grain fill under cool, moist weather.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["pyraclostrobin", "azoxystrobin", "trifloxystrobin", "propiconazole", "tebuconazole"],
                "application": "Apply at early leaf stages if scouting shows active pustule development and yield potential is high; keep label intervals and rotate strobilurins with triazoles. Applications after grain fill are generally not economic for field corn.",
            }
        ],
    },
    "corn-cercospora-gray-leaf-spot": {
        "treatment": [
            "Manage crop residue: bury or remove infected stalks and leaves — the fungus overwinters on residue.",
            "Rotate with non-host crops (soybean, small grains) for 1-2 years.",
            "Plant resistant or tolerant hybrids where available.",
            "Scout the lower canopy from V6; apply fungicide when lesions reach the ear leaf before or at tasseling under warm, humid conditions.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["azoxystrobin", "pyraclostrobin", "trifloxystrobin", "propiconazole", "cyproconazole"],
                "application": "Time application at VT-R1 (tasseling to silking) when at least half of plants show lesions on the ear leaf or above, especially on hybrids with intermediate or low resistance. Rotate FRAC groups between applications — strobilurin resistance in Cercospora is documented in several regions.",
            }
        ],
    },
    "corn-northern-leaf-blight": {
        "treatment": [
            "Rotate with non-host crops and use tillage to bury infected residue — sporulation requires residue.",
            "Plant hybrids carrying Ht resistance genes.",
            "Scout for cigar-shaped lesions from V6; fungicide pays on continuous corn with susceptible hybrids when lesions appear before tasseling under cool, humid conditions with long leaf wetness.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["azoxystrobin", "pyraclostrobin", "propiconazole", "trifloxystrobin"],
                "application": "Apply at VT-R1 when the majority of plants have lesions reaching the ear leaf on susceptible hybrids; intervals are often 14-21 days per label. Rotate modes of action between applications.",
            }
        ],
    },
    "grape-black-rot": {
        "treatment": [
            "Prune out and destroy mummified berries, infected tendrils and trunk cankers — mummies are the overwintering inoculum.",
            "Remove wild grapes and keep an open canopy so clusters dry quickly.",
            "Apply a strict protectant program from bud swell through bunch closure (the critical window); sprays after infection do not cure already-infected berries.",
            "Bury or remove fallen leaves.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["myclobutanil", "mancozeb", "captan", "boscalid", "pyraclostrobin"],
                "application": "Protectant program: start at bud swell and continue at label intervals (7-14 days, shorter after rain) through bloom and bunch closure. Myclobutanil gives limited eradicant action early but cannot cure established infections. Rotate groups and respect the PHI before harvest.",
            }
        ],
    },
    "grape-esca": {
        "incurable": True,
        "treatment": [
            "No reliable chemical cure exists for this trunk disease; management is by vine surgery and wound protection.",
            "Prune out symptomatic wood during dormancy, cutting back to clean wood well below visible lesions; tiger-striped leaves signal affected cordons.",
            "Protect all pruning wounds immediately with a registered wound protectant (for example Trichoderma-based formulations where they are registered).",
            "Remove and replace vines with severe trunk lesions; avoid mechanical injury, drought stress and excessive cropping, which favour the disease.",
        ],
        "pesticides": [
            {
                "category": "Biological wound protectant",
                "active_ingredients": ["Trichoderma-based pruning-wound dressings"],
                "application": "Paint onto fresh pruning wounds immediately after cutting, per the product label. There is no curative trunk fungicide; trunk injections may only suppress symptoms in some regions and are not a cure.",
            }
        ],
    },
    "grape-leaf-blight-isariopsis": {
        "treatment": [
            "Remove and destroy infected leaf litter — the fungus overwinters on fallen leaves.",
            "Maintain an open canopy with positioning and leaf removal to lower humidity around the fruit zone.",
            "Apply protectant sprays during the growing season where the disease is active in warm, humid climates.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["copper (e.g., Bordeaux mixture)", "mancozeb", "difenoconazole"],
                "application": "Protectant sprays on green canopy at label intervals during warm, wet periods; rotate copper with mancozeb or a DMI group. Observe the PHI and copper limits under organic rules.",
            }
        ],
    },
    "orange-hlb-citrus-greening": {
        "incurable": True,
        "treatment": [
            "There is no known cure: infected trees remain infected and decline. Regional programs typically require roguing — remove and destroy infected trees.",
            "Control the Asian citrus psyllid vector (Diaphorina citri): monitor with yellow sticky traps and shoot inspections.",
            "Manage new flush to reduce psyllid breeding synchrony; remove weeds that host psyllids.",
            "Use only certified clean nursery stock; reflective mulches deter psyllids.",
            "Foliar micronutrient programs (zinc, manganese, iron) can maintain productivity of mildly infected trees but do not remove the bacterium.",
        ],
        "pesticides": [
            {
                "category": "Insecticide (psyllid vector control — does not cure infected trees)",
                "active_ingredients": ["imidacloprid", "thiamethoxam", "clothianidin", "carbaryl", "kaolin particle film"],
                "application": "Target psyllids on new flush: soil-applied neonicotinoids (imidacloprid, clothianidin) at the labelled seasonal soil rate, timed foliar carbaryl or thiamethoxam for adults during flush, plus kaolin or reflective mulch. Vector control protects healthy trees only. Neonicotinoid rules differ widely by country — confirm local registration.",
            }
        ],
    },
    "peach-bacterial-spot": {
        "treatment": [
            "Plant resistant cultivars — the most effective long-term control.",
            "Avoid working among trees when foliage is wet; bacteria spread on tools and clothing.",
            "Apply copper during dormancy to reduce overwintering inoculum, and streptomycin at shuck split where the local label allows it on stone fruit.",
            "Maintain balanced nutrition (avoid excess nitrogen) and provide windbreaks to reduce the leaf tearing that creates infection sites.",
        ],
        "pesticides": [
            {
                "category": "Copper bactericide",
                "active_ingredients": ["fixed copper (copper hydroxide, copper octanoate)"],
                "application": "Dormant applications plus cover sprays per label at infection periods; watch for phytotoxicity at high temperatures or concentrations.",
            },
            {
                "category": "Antibiotic bactericide (where registered)",
                "active_ingredients": ["streptomycin"],
                "application": "Apply at shuck split and per label during warm, wet weather. Antibiotics on fruit crops are restricted in many countries (not permitted in the EU); confirm local registration and resistance-management rules.",
            },
        ],
    },
    "pepper-bacterial-spot": {
        "treatment": [
            "Use certified healthy seed and transplants — seed and seedlings are the main entry route.",
            "Rotate with non-solanaceous crops for 1-2 years and remove crop debris after harvest.",
            "Avoid overhead irrigation and working wet foliage; irrigate at the base in the morning.",
            "Apply copper-based bactericides protectively during wet weather, before infection spreads through the canopy.",
        ],
        "pesticides": [
            {
                "category": "Copper bactericide",
                "active_ingredients": ["fixed copper (copper hydroxide, copper sulfate)"],
                "application": "Weekly or per-label cover sprays during warm, rainy periods; copper is protectant — it must be on the leaf before infection. Avoid tank-mixes that raise phytotoxicity and respect the PHI.",
            },
            {
                "category": "Antibiotic bactericide (where registered)",
                "active_ingredients": ["streptomycin", "oxytetracycline"],
                "application": "Mostly greenhouse or transplant programmes where registered; not permitted in the EU. Rotate with copper and follow local resistance-management rules.",
            },
        ],
    },
    "potato-early-blight": {
        "treatment": [
            "Rotate with non-host crops (not tomato, potato or pepper) for 2-3 years — inoculum builds on debris.",
            "Bury or remove infected vines and volunteer plants.",
            "Maintain a vigorous canopy with balanced nitrogen and regular irrigation; stress and senescence trigger infection.",
            "Hill or mulch to keep developing tubers covered from soil splash.",
            "Apply protectant fungicide once symptoms appear and continue on a label interval until vines senesce.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["chlorothalonil", "mancozeb", "azoxystrobin", "boscalid", "picoxystrobin"],
                "application": "Protectant schedule from first symptoms (often at tuber initiation) at 7-14 day intervals per label with full canopy coverage including lower leaves; alternate multi-site products (chlorothalonil, mancozeb) with single-site strobilurins to manage resistance.",
            }
        ],
    },
    "potato-late-blight": {
        "treatment": [
            "Destroy all inoculum sources: remove and bury or burn cull piles, volunteer potatoes and nightshade weeds — green inoculum near the field is a classic epidemic start.",
            "Plant certified seed and resistant or tolerant varieties suited to local races.",
            "Use ridges and drip irrigation; avoid overhead irrigation and prolonged leaf wetness.",
            "Scout daily in warm, humid weather and use forecast-driven warning systems; at first symptoms remove infected plants and tighten the spray interval.",
            "Delay harvest appropriately after haulm destruction to protect tubers.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["chlorothalonil", "mancozeb", "metalaxyl-mefenoxam", "dimethomorph", "cyazofamid", "mandipropamid"],
                "application": "This is an oomycete: multi-site protectants (chlorothalonil, mancozeb) must be on foliage before infection and cannot cure lesions. Rotate in oomycete-specific systemic products (mefenoxam, dimethomorph, cyazofamid, mandipropamid) — mefenoxam resistance is widespread, so never rely on it alone. Spray at 5-7 day intervals under epidemic pressure and observe the PHI.",
            }
        ],
    },
    "squash-powdery-mildew": {
        "treatment": [
            "Choose resistant or tolerant varieties — the most common control.",
            "Space plants for airflow; powdery mildew thrives in dense canopies with warm days and humid nights and, unlike most fungi, does not need free leaf wetness.",
            "Remove badly affected lower leaves as the season progresses and avoid excess nitrogen.",
            "Apply fungicide at the first sign on lower leaves and continue to protect fruit set; spray both leaf surfaces thoroughly.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["sulfur", "potassium bicarbonate", "myclobutanil", "difenoconazole", "horticultural oil", "Bacillus subtilis"],
                "application": "Start at first signs at label intervals (typically 7-14 days); rotate the systemic DMIs (myclobutanil, difenoconazole) with contact options (sulfur, potassium bicarbonate, oil, biologicals) to limit resistance. Avoid sulfur plus oil combinations and applications above about 32 C to prevent plant damage; respect the PHI.",
            }
        ],
    },
    "strawberry-leaf-scorch": {
        "treatment": [
            "Remove and destroy infected leaves; renovate heavily affected plantings after harvest (mow, remove old crowns, fertilise to stimulate healthy regrowth).",
            "Renew the planting on a 2-3 year cycle using certified clean transplants.",
            "Mulch to reduce splash and keep foliage dry; irrigate with drip rather than overhead.",
            "Apply fungicide protectively in fall and spring — infection occurs in both periods on young leaves.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["captan", "thiophanate-methyl", "myclobutanil", "boscalid", "pyraclostrobin"],
                "application": "Protectant sprays from post-harvest renovation through fall, and again from late dormancy, at label intervals (often 7-14 days); alternate multi-site captan with DMI or SDHI single-site groups. Mind rain-fastness and the PHI on ripe fruit.",
            }
        ],
    },
    "tomato-bacterial-spot": {
        "treatment": [
            "Use certified seed and healthy transplants; treat saved seed with the hot-water or copper protocol from your extension service.",
            "Rotate with non-host crops for 1-2 years and remove all tomato debris after harvest.",
            "Avoid overhead irrigation and splash; mulch at the base and irrigate in the morning so leaves dry.",
            "Apply copper protectively during warm, wet weather — before symptoms spread through the canopy.",
        ],
        "pesticides": [
            {
                "category": "Copper bactericide",
                "active_ingredients": ["fixed copper (copper hydroxide, copper sulfate)"],
                "application": "Apply protectively at 7-10 day intervals through warm, wet weather with thorough canopy coverage; copper must be present before infection and cannot cure infected tissue. Watch for phytotoxicity in hot weather; respect the PHI.",
            },
            {
                "category": "Antibiotic bactericide (where registered)",
                "active_ingredients": ["oxytetracycline", "streptomycin"],
                "application": "Used in greenhouse or transplant programmes where permitted (not in the EU); apply at label intervals and rotate with copper to limit resistance.",
            },
        ],
    },
    "tomato-early-blight": {
        "treatment": [
            "Mulch at planting and stake or prune up the plant to block soil splash — most infection comes from debris-borne spores splashed onto lower leaves.",
            "Remove the lowest infected leaves as they appear and destroy debris after harvest.",
            "Rotate with non-solanaceous crops for 2-3 years and avoid dense spacing.",
            "Apply protectant fungicide from the first lesions, focusing on the lower canopy, and continue on a label interval through fruit set.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["chlorothalonil", "mancozeb", "azoxystrobin", "boscalid", "picoxystrobin"],
                "application": "Start when first lesions appear (or preventively in humid regions) at 7-14 day intervals with excellent lower-canopy coverage; alternate multi-site protectants with single-site products (azoxystrobin, boscalid) to slow resistance and observe the chlorothalonil day-to-harvest restriction.",
            }
        ],
    },
    "tomato-late-blight": {
        "treatment": [
            "At first symptoms remove and destroy infected plants or leaves immediately — bag them, do not compost.",
            "Eliminate inoculum sources: rogue volunteer tomatoes, potatoes and nightshade weeds; remove cull piles.",
            "Avoid overhead irrigation; water at the base in the morning and space or prune for rapid drying.",
            "Apply protective fungicide before infection when cool, humid weather is forecast; tighten to a 5-7 day interval during active pressure.",
            "Choose resistant varieties for plantings where the disease recurs.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["chlorothalonil", "mancozeb", "copper", "metalaxyl-mefenoxam", "dimethomorph", "cyazofamid"],
                "application": "Oomycete programme: multi-site protectants (chlorothalonil, mancozeb, copper) applied before rain on a 5-7 day interval under epidemic pressure; rotate in oomycete-specific systemic products (mefenoxam, dimethomorph, cyazofamid) — mefenoxam resistance is common, so do not rely on it alone. Observe the harvest interval.",
            }
        ],
    },
    "tomato-leaf-mold": {
        "treatment": [
            "Reduce humidity — the driver of leaf mold: ventilate greenhouses, widen spacing, water at the base and avoid evening leaf wetness.",
            "Resistant varieties (Ve genes) exist for greenhouse production.",
            "Remove and destroy infected lower leaves as symptoms appear.",
            "Scout under the canopy: yellow blotches on the upper leaf surface with olive-brown velvety sporulation underneath signal the start of the epidemic.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["difenoconazole", "chlorothalonil", "copper", "cyprodinil-fludioxonil", "mandipropamid"],
                "application": "Greenhouse protectant programmes at 7-10 day intervals from first symptoms, with thorough underside coverage because sporulation occurs on the leaf underside; alternate DMI and contact groups; observe the PHI on fruiting crops.",
            }
        ],
    },
    "tomato-septoria-leaf-spot": {
        "treatment": [
            "Mulch immediately and stake or prune to block splash and speed leaf drying.",
            "At the first small round spots with grey centres, remove the lowest infected leaves and destroy debris after harvest.",
            "Rotate 2-3 years with non-hosts and avoid overhead or evening irrigation.",
            "Apply protectant fungicide once symptoms are present and repeat per label through the season.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["chlorothalonil", "mancozeb", "copper", "azoxystrobin", "boscalid"],
                "application": "7-14 day cover sprays from first symptoms with full canopy, especially lower-leaf, coverage; rotate multi-site protectants with single-site products and mind the PHI and re-entry intervals.",
            }
        ],
    },
    "tomato-spider-mites": {
        "treatment": [
            "Conserve predators: lady beetles, minute pirate bugs and especially predatory mites (Phytoseiulus persimilis) control two-spotted spider mite — avoid broad-spectrum insecticides that flare mite populations.",
            "Remove weeds and crop debris that harbour mites and wash dust off foliage — dust favours mites.",
            "Monitor with a beat test: tap foliage over white paper; moving mites plus pale stippling on leaves confirm presence.",
            "Act at threshold (stippling spreading upward); populations explode in hot, dry weather.",
        ],
        "pesticides": [
            {
                "category": "Insecticidal soap / horticultural oil",
                "active_ingredients": ["insecticidal soap (potassium salts of fatty acids)", "horticultural oil", "neem oil (azadirachtin)"],
                "application": "Spray thorough coverage of leaf undersides every 5-7 days per label while populations build; contact action only, so repeats are needed. Avoid applying in hot sun or soon after sulfur residues (phytotoxicity).",
            },
            {
                "category": "Miticide (when threshold exceeded)",
                "active_ingredients": ["abamectin", "spiromesifen", "bifenthrin (where registered)"],
                "application": "Apply to the leaf underside at labelled intervals, rotating modes of action with the softer options to delay resistance; in greenhouses, releasing predatory mites instead of miticides is preferred where possible. Some products differ by country — confirm local registration.",
            },
        ],
    },
    "tomato-target-spot": {
        "treatment": [
            "Remove and bury or destroy crop debris after harvest — the fungus survives on residue.",
            "Rotate with non-host crops for 1-2 years.",
            "Stake and prune for airflow, mulch to reduce splash and avoid working wet foliage.",
            "Apply protectant fungicide from first symptoms during warm, humid weather.",
        ],
        "pesticides": [
            {
                "category": "Fungicide",
                "active_ingredients": ["chlorothalonil", "mancozeb", "azoxystrobin", "boscalid"],
                "application": "7-14 day cover sprays per label from first symptoms with full canopy coverage, rotating multi-site and single-site groups; observe the PHI before harvest.",
            }
        ],
    },
    "tomato-yellow-leaf-curl-virus": {
        "incurable": True,
        "treatment": [
            "No cure exists: remove and destroy infected plants promptly to reduce inoculum in the field.",
            "Control whiteflies (Bemisia tabaci), which spread the virus; monitor with yellow sticky traps.",
            "Use reflective mulches and insect screens in protected cultivation to repel whiteflies.",
            "Plant resistant varieties where available, remove weed hosts around the field and delay planting to avoid peak whitefly pressure.",
        ],
        "pesticides": [
            {
                "category": "Insecticide (whitefly vector control — does not cure infected plants)",
                "active_ingredients": ["imidacloprid", "thiamethoxam", "flonicamid", "pyriproxyfen", "neem (azadirachtin)"],
                "application": "Treat transplants and new growth at label intervals, rotating neonicotinoids, insect growth regulators (pyriproxyfen) and botanicals to avoid resistance; combine with reflective mulch and screens. These protect healthy plants only. Resistance-breaking virus strains exist, so follow local varietal advice.",
            }
        ],
    },
    "tomato-mosaic-virus": {
        "incurable": True,
        "treatment": [
            "No cure and no pesticide: remove and destroy infected plants immediately (bag them; do not compost).",
            "Hygiene stops the spread: wash hands with soap and disinfect tools, stakes and hands with trisodium phosphate (10%) or a dilute bleach solution between plants.",
            "Use certified virus-free seed and transplants; mosaic viruses can be seed-borne, so test saved seed.",
            "Do not smoke or handle tobacco just before working in the crop — tobacco can carry the virus onto hands and leaves.",
            "Remove Solanaceae weed hosts around the field and plant resistant varieties where available.",
        ],
        "pesticides": [],
    },
}

HEALTHY_DEFAULT = {"treatment": [], "pesticides": []}


def main() -> None:
    with DISEASES_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    records = data.get("diseases", [])

    updated = 0
    missing: list[str] = []
    for rec in records:
        rid = rec.get("id", "")
        if rid in TREATMENT:
            spec = TREATMENT[rid]
            rec["treatment"] = list(spec.get("treatment", []))
            rec["pesticides"] = list(spec.get("pesticides", []))
            if spec.get("incurable"):
                rec["incurable"] = True
            updated += 1
        elif rec.get("condition") == "healthy":
            rec["treatment"] = list(HEALTHY_DEFAULT["treatment"])
            rec["pesticides"] = list(HEALTHY_DEFAULT["pesticides"])
        else:
            missing.append(rid)

    diseased_total = sum(1 for r in records if r.get("condition") == "diseased")
    with_treatment = sum(
        1
        for r in records
        if r.get("condition") == "diseased" and r.get("treatment")
    )
    with_pesticides = sum(
        1
        for r in records
        if r.get("condition") == "diseased" and r.get("pesticides")
    )

    print(f"records: {len(records)} (diseased: {diseased_total})")
    print(f"treatment applied: {updated}, healthy defaults: {len(records) - updated - len(missing)}")
    print(f"diseased with treatment: {with_treatment}/{diseased_total}")
    print(f"diseased with pesticides: {with_pesticides}/{diseased_total}")
    if missing:
        print(f"WARNING — diseased records without authored treatment: {missing}")

    if not missing:
        with DISEASES_PATH.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"written: {DISEASES_PATH}")
    else:
        print("NOT written — author the missing records first.")


if __name__ == "__main__":
    main()
