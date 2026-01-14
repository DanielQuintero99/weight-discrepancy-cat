import re
import io
import pandas as pd
import pdfplumber
import streamlit as st

# =========================
# CONFIG
# =========================
TOL_DEFAULT = 0.10  # 🔒 fijo a ±10% para la calculadora previa (igual que tu celda 0)
DEDUP_CASES_ACROSS_INVOICES_DEFAULT = True  # si un CASE/PACKAGE aparece en 2 invoices, tratar como 1 pieza física

# Anti-locura (pero GR manda cuando existe match)
BIG_MAX_RATIO = 1.25
BIG_MIN_RATIO = 0.80
TINY_GUARD_KG = 1.0
TINY_MAX_RATIO = 2.0
MAX_ABS_CHANGE_KG = 80.0
CUSHION_KG = 0.5
MIN_CHANGE_KG = 0.30

LBS_TO_KG = 0.45359237
KG_TO_LBS = 1.0 / LBS_TO_KG

PACKING_LIST_RE = re.compile(
    r"(P\s*A\s*C\s*K\s*I\s*N\s*G\s+L\s*I\s*S\s*T)|"
    r"(PACKING\s*LIST)|"
    r"(LISTA\s+DE\s+EMPAQUE)|"
    r"(LISTA\s+DE\s+EMBALAGEM)",
    re.IGNORECASE
)

# =========================
# Helpers Calculadora (CELDA 0)
# =========================
def band_from_gr(gr: float, tol: float = TOL_DEFAULT):
    # Regla CAT:
    # Invoice debe estar entre: [ GR/(1+tol) , GR/(1-tol) ]
    low = gr / (1 + tol)
    high = gr / (1 - tol)
    return low, high

# =========================
# PDF helpers
# =========================
def pdf_bytes_to_lines(pdf_bytes: bytes):
    lines = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            for ln in txt.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    return lines

def is_invoice(lines):
    return any(PACKING_LIST_RE.search(l) for l in lines)

def is_gr(lines):
    u = " ".join(lines).upper()
    return ("WAREHOUSE RECEIPT" in u) or ("ORDGR" in u) or re.search(r"\b[A-Z]{3}GR\d{6,}\b", u) is not None

# =========================
# PACKING LIST block (robusto)
# =========================
def extract_packing_list_block(lines, invoice_name):
    start = None
    for i, l in enumerate(lines):
        if PACKING_LIST_RE.search(l):
            start = i
            break
    if start is None:
        raise ValueError(f"Invoice '{invoice_name}': No encontré PACKING LIST.")

    block = []
    for l in lines[start:]:
        u = l.upper().strip()
        if "INVOICE TOTALS" in u or u.startswith("TOTALS:"):
            break
        block.append(l)

    if not block:
        raise ValueError(f"Invoice '{invoice_name}': PACKING LIST vacío o mal delimitado.")
    return block

# =========================
# Invoice parser: CASE (con o sin SRC) + PACKAGE#
# =========================
def parse_invoice_packing_list(lines, invoice_name):
    block = extract_packing_list_block(lines, invoice_name)

    stitched = []
    i = 0
    while i < len(block):
        cur = block[i]
        if i + 1 < len(block):
            nxt = block[i + 1]
            if (re.search(r"\b\d{4,}\b", cur) or "PACKAGE" in cur.upper()) and len(re.findall(r"\d+\.\d+", cur)) < 2:
                stitched.append(cur + " " + nxt)
                i += 2
                continue
        stitched.append(cur)
        i += 1
    block = stitched

    piece_re_src = re.compile(
        r"^(?P<src>[A-Z]{3})\s+(?P<id>\d{6,})\s+.+?\s+(?P<grosslbs>\d+\.\d+)\s+(?P<netlbs>\d+\.\d+)\b",
        re.IGNORECASE
    )
    piece_re_case = re.compile(
        r"^(?P<id>\d{6,})\s+.+?\s+(?P<grosslbs>\d+\.\d+)\s+(?P<netlbs>\d+\.\d+)\b",
        re.IGNORECASE
    )
    piece_re_pkg = re.compile(
        r"^(?P<id>\d{4,})\s+.+?\s+(?P<grosslbs>\d+(?:\.\d+)?)\s+(?P<netlbs>\d+(?:\.\d+)?)\b",
        re.IGNORECASE
    )

    def find_kg_line(idx, lookahead=4):
        for j in range(1, lookahead + 1):
            if idx + j >= len(block):
                return None
            s = block[idx + j].strip()
            if re.match(r"^\d+(\.\d+)?\s", s):
                return s
        return None

    rows = []
    i = 0
    while i < len(block):
        line = block[i].strip()
        u = line.upper()

        if u.startswith("SRC ") or u.startswith("--------") or ("PACKAGE#" in u and "KILOS" in u) or ("CASE NO" in u and "KILOS" in u):
            i += 1
            continue

        m = piece_re_src.match(line)
        if not m:
            m = piece_re_case.match(line)
        if not m:
            m = piece_re_pkg.match(line)

        if not m:
            i += 1
            continue

        piece_id = str(m.group("id"))
        gross_lbs = float(m.group("grosslbs"))

        gross_kg = None
        kg_line = find_kg_line(i, lookahead=4)
        if kg_line:
            nums = re.findall(r"\d+(?:\.\d+)?", kg_line)
            if nums:
                gross_kg = float(nums[0])
        if gross_kg is None:
            gross_kg = gross_lbs * LBS_TO_KG

        rows.append({
            "INVOICE_FILE": invoice_name,
            "CASE_NO": piece_id,
            "CAT_WEIGHT_LBS": gross_lbs,
            "CAT_WEIGHT_KG": gross_kg
        })

        if kg_line and (i + 1 < len(block)) and (kg_line.strip() == block[i + 1].strip()):
            i += 2
        else:
            i += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Invoice '{invoice_name}': No pude extraer piezas del PACKING LIST.")

    df["CASE_OCC"] = df.groupby(["INVOICE_FILE", "CASE_NO"]).cumcount() + 1
    df["PIECE_ID"] = df["INVOICE_FILE"].astype(str) + "|" + df["CASE_NO"].astype(str) + "|" + df["CASE_OCC"].astype(str)
    return df.reset_index(drop=True)

# =========================
# Dedupe entre invoices (misma pieza física)
# =========================
def collapse_duplicates_across_invoices(inv_df):
    inv_df = inv_df.copy().reset_index(drop=True)
    idx = inv_df.groupby("CASE_NO")["CAT_WEIGHT_KG"].idxmax()
    base = inv_df.loc[idx].copy()

    files = inv_df.groupby("CASE_NO")["INVOICE_FILE"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index()
    base = base.merge(files, on="CASE_NO", suffixes=("", "_ALL"))
    base["INVOICE_FILE"] = base["INVOICE_FILE_ALL"]
    base.drop(columns=["INVOICE_FILE_ALL"], inplace=True)

    base["CASE_OCC"] = 1
    base["PIECE_ID"] = base["CASE_NO"].astype(str)
    return base.sort_values("CASE_NO").reset_index(drop=True)

# =========================
# GR parser (tabla Package ID + fallback)
# =========================
def parse_gr(lines):
    start = None
    for i, l in enumerate(lines):
        u = l.upper()
        if u.startswith("PACKAGE ID") or ("PACKAGE ID" in u and "WGT" in u):
            start = i
            break

    gr_pieces = []

    if start is not None:
        window = lines[start:start + 800]
        for l in window:
            u = l.upper()
            if "HANDLING" in u or "CHARGES" in u or "SERVICE" in u:
                break
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*KGM\b", u):
                gr_pieces.append(float(m.group(1)))
            for m in re.finditer(r"(\d+(?:\.\d+)?)KGM\b", u.replace(" ", "")):
                gr_pieces.append(float(m.group(1)))

        if gr_pieces:
            return sum(gr_pieces), gr_pieces

    for l in lines:
        u = l.upper()
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*KGM\b", u):
            gr_pieces.append(float(m.group(1)))
        for m in re.finditer(r"(\d+(?:\.\d+)?)KGM\b", u.replace(" ", "")):
            gr_pieces.append(float(m.group(1)))

    if gr_pieces:
        return sum(gr_pieces), gr_pieces

    raise ValueError("GR: No pude extraer pesos KGM del GR.")

# =========================
# Matching por similitud (sorted-to-sorted)
# =========================
def match_gr_to_invoice_by_similarity(inv_df, gr_pieces):
    inv_sorted = inv_df.sort_values("CAT_WEIGHT_KG").reset_index(drop=True)
    gr_sorted = sorted(gr_pieces)
    n = min(len(inv_sorted), len(gr_sorted))
    mapping = {}
    for i in range(n):
        mapping[inv_sorted.loc[i, "PIECE_ID"]] = float(gr_sorted[i])
    return mapping, len(inv_sorted), len(gr_sorted)

# =========================
# Regla CAT: tolerancia basada en INVOICE
# invoice ∈ [GR/(1+tol), GR/(1-tol)]
# =========================
def invoice_band_from_gr(gr_total, tol=0.10):
    low = gr_total / (1 + tol)
    high = gr_total / (1 - tol)
    return low, high

# =========================
# Ajuste guiado por GR
# =========================
def gr_guided_adjust_auto(inv_df, gr_map, low, high):
    inv = inv_df.copy().reset_index(drop=True)
    n = len(inv)
    cur_total = float(inv["CAT_WEIGHT_KG"].sum())

    if low <= cur_total <= high:
        return {}

    if cur_total < low:
        need_up = True
        target = min(high, low + CUSHION_KG)
        gap = target - cur_total
    else:
        need_up = False
        target = max(low, high - CUSHION_KG)
        gap = cur_total - target

    def allowed_delta(cur, gr):
        if cur <= 0:
            return 0.0

        if gr is not None and isinstance(gr, (int, float)) and gr > 0:
            if need_up:
                return max(0.0, gr - cur)
            else:
                return max(0.0, cur - gr)

        if cur < TINY_GUARD_KG:
            if need_up:
                return cur * (TINY_MAX_RATIO - 1.0)
            else:
                return cur * (1.0 - (1.0 / TINY_MAX_RATIO))
        else:
            if need_up:
                return cur * (BIG_MAX_RATIO - 1.0)
            else:
                return cur * (1.0 - BIG_MIN_RATIO)

    cands = []
    for _, r in inv.iterrows():
        pid = r["PIECE_ID"]
        cur = float(r["CAT_WEIGHT_KG"])
        gr = gr_map.get(pid, None)

        cap = min(MAX_ABS_CHANGE_KG, allowed_delta(cur, gr))
        if cap < MIN_CHANGE_KG:
            continue

        score = (1000 * min(cap, gap)) + (50 * cap) + cur
        cands.append((score, pid, cur, gr, cap))

    cands.sort(key=lambda x: x[0], reverse=True)
    if not cands:
        return {}

    max_by_n = max(3, min(20, int(round(0.50 * n))))

    updates = {}
    remaining = gap

    for _, pid, cur, gr, cap in cands:
        if remaining <= 1e-9 or len(updates) >= max_by_n:
            break

        delta = min(remaining, cap)
        if delta < MIN_CHANGE_KG:
            continue

        new_kg = (cur + delta) if need_up else max(0.01, cur - delta)

        # clamp extra por GR si existe
        if gr is not None and isinstance(gr, (int, float)) and gr > 0:
            if need_up:
                new_kg = min(new_kg, gr)
            else:
                new_kg = max(new_kg, gr)

        updates[pid] = round(new_kg, 2)
        remaining -= delta

    return updates

# =========================
# Tablas
# =========================
def build_cat_tables(inv_df, updates):
    rows = []
    for _, r in inv_df.iterrows():
        pid = r["PIECE_ID"]
        old_lbs = float(r["CAT_WEIGHT_LBS"])
        old_kg = float(r["CAT_WEIGHT_KG"])

        if pid in updates:
            new_kg = float(updates[pid])
            new_lbs = new_kg * KG_TO_LBS
            changed = True
        else:
            new_kg = old_kg
            new_lbs = old_lbs
            changed = False

        rows.append({
            "CASE/BOX": r["CASE_NO"],
            "CAT WEIGHT lbs": round(old_lbs, 2),
            "NEW WEIGHT lbs": round(new_lbs, 2),
            "NEW WEIGHT kgs": round(new_kg, 2),
            "NEW LENGTH": "N/A",
            "NEW WIDTH": "N/A",
            "NEW HEIGHT": "N/A",
            "INVOICE #": r["INVOICE_FILE"],
            "__CHANGED__": changed
        })

    df = pd.DataFrame(rows)
    df_adj = df[df["__CHANGED__"]].drop(columns=["__CHANGED__"])
    df_full = df.drop(columns=["__CHANGED__"])
    return df_full, df_adj

def build_validation(inv_df, gr_map, updates):
    rows = []
    for _, r in inv_df.iterrows():
        pid = r["PIECE_ID"]
        inv_kg = float(r["CAT_WEIGHT_KG"])
        gr_kg = gr_map.get(pid, None)
        new_kg = float(updates.get(pid, inv_kg))
        rows.append({
            "CASE/BOX": r["CASE_NO"],
            "INVOICE ORIGINAL KG": round(inv_kg, 2),
            "GR KG (matched)": (round(gr_kg, 2) if gr_kg is not None else None),
            "NEW KG": round(new_kg, 2),
        })
    return pd.DataFrame(rows)

# =========================
# RUNNER
# =========================
def run_analysis(uploaded: dict, tol=TOL_DEFAULT, dedup=True):
    if len(uploaded) < 2:
        raise ValueError("Sube mínimo 2 PDFs: 1 GR + 1 o más Invoices.")

    invoices = []
    gr_lines = None
    gr_file = None

    for fname, fbytes in uploaded.items():
        lines = pdf_bytes_to_lines(fbytes)
        if is_gr(lines) and gr_lines is None:
            gr_lines = lines
            gr_file = fname
        elif is_invoice(lines):
            invoices.append((fname, lines))
        else:
            invoices.append((fname, lines))

    if gr_lines is None:
        raise ValueError("No encontré el GR.")
    if not invoices:
        raise ValueError("No encontré ninguna invoice con PACKING LIST.")

    gr_total, gr_pieces = parse_gr(gr_lines)

    inv_dfs = [parse_invoice_packing_list(lines, name) for name, lines in invoices]
    inv_df = pd.concat(inv_dfs, ignore_index=True)

    if dedup:
        inv_df = collapse_duplicates_across_invoices(inv_df)

    inv_total = float(inv_df["CAT_WEIGHT_KG"].sum())
    low, high = invoice_band_from_gr(gr_total, tol)

    in_before = (low <= inv_total <= high)

    gr_map, inv_n, gr_n = match_gr_to_invoice_by_similarity(inv_df, gr_pieces)

    updates = {}
    if not in_before:
        updates = gr_guided_adjust_auto(inv_df, gr_map, low, high)

    df_full, df_adj = build_cat_tables(inv_df, updates)
    new_total = float(df_full["NEW WEIGHT kgs"].sum())
    in_after = (low <= new_total <= high)

    summary = pd.DataFrame([{
        "GR file": gr_file,
        "Invoices files": ", ".join([x[0] for x in invoices]),
        "Invoice total (kg)": round(inv_total, 2),
        "GR total (kg)": round(gr_total, 2),
        "Allowed low (kg)": round(low, 2),
        "Allowed high (kg)": round(high, 2),
        "Pieces detected": int(len(inv_df)),
        "GR pieces extracted": int(gr_n),
        "Pieces changed": int(len(df_adj)),
        "New total (kg)": round(new_total, 2),
        "In tolerance BEFORE": bool(in_before),
        "In tolerance AFTER": bool(in_after),
    }])

    validation_df = build_validation(inv_df, gr_map, updates)

    return summary, df_full, df_adj, validation_df

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Weight Discrepancy + CAT Adjuster", layout="wide")
st.title("⚖️ Weight Discrepancy (±10%) + CAT Adjuster")

# Mantener estado
if "calc_done" not in st.session_state:
    st.session_state.calc_done = False
if "has_discrepancy" not in st.session_state:
    st.session_state.has_discrepancy = False
if "low" not in st.session_state:
    st.session_state.low = None
if "high" not in st.session_state:
    st.session_state.high = None

# -------------------------
# CELDA 0 (adaptada)
# -------------------------
st.subheader("🧮 Calculadora previa (±10% fijo, basado en GR)")

with st.container():
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        gr_val = st.number_input("GR (kg)", min_value=0.0, value=0.0, step=0.1)
    with colB:
        inv_val = st.number_input("Invoice (kg)", min_value=0.0, value=0.0, step=0.1)
    with colC:
        st.markdown("**Tolerancia:** ±10% (fija)")

calc_btn = st.button("Calcular", type="primary")

if calc_btn:
    if gr_val <= 0 or inv_val <= 0:
        st.error("⚠️ Ingresa valores > 0 para GR e Invoice.")
    else:
        low, high = band_from_gr(gr_val, tol=TOL_DEFAULT)
        in_tol = (low <= inv_val <= high)

        st.session_state.calc_done = True
        st.session_state.has_discrepancy = not in_tol
        st.session_state.low = low
        st.session_state.high = high

        # “Cuadro tipo Excel”
        df_calc = pd.DataFrame([{
            "-10.00% (LOW)": round(low, 3),
            "Commercial invoice weight -->": round(inv_val, 2),
            "+10.00% (HIGH)": round(high, 3),
            "enter GXD weight here -->": round(gr_val, 2),
        }])

        st.markdown("### Weight discrepancy")
        st.dataframe(df_calc, use_container_width=True)

        if in_tol:
            st.success("✅ NO hay weight discrepancy. No necesitas subir documentos.")
        else:
            st.warning("⚠️ SÍ hay weight discrepancy. Sube los PDFs para hacer la corrección.")

# -------------------------
# Control: permitir subir PDFs si hay discrepancy o si el usuario quiere forzar
# -------------------------
st.divider()
st.subheader("📄 Corrección con PDFs (GR + Invoices)")

force_upload = st.checkbox("Quiero subir PDFs aunque NO haya discrepancy", value=False)

can_upload = (st.session_state.calc_done and st.session_state.has_discrepancy) or force_upload

if not can_upload:
    st.info("Primero usa la calculadora. Si sale discrepancy, aquí se habilita la carga de PDFs.")
    st.stop()

with st.sidebar:
    st.header("Carga de PDFs")
    pdfs = st.file_uploader(
        "Sube 1 GR + 1 o más Invoices (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )
    dedup = st.checkbox("Deduplicar CASEs entre invoices", value=DEDUP_CASES_ACROSS_INVOICES_DEFAULT)
    run_btn = st.button("▶️ Ejecutar análisis", type="primary")

if run_btn:
    if not pdfs or len(pdfs) < 2:
        st.error("Sube mínimo 2 PDFs: 1 GR + 1 o más Invoices.")
        st.stop()

    uploaded = {f.name: f.read() for f in pdfs}

    with st.spinner("Analizando PDFs..."):
        try:
            summary, df_full, df_adjusted, validation_df = run_analysis(
                uploaded,
                tol=TOL_DEFAULT,   # 🔒 fijo a 10% (igual que tu calculadora y tu celda 0)
                dedup=dedup
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    st.subheader("📊 Resumen del shipment")
    st.dataframe(summary, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Suma NEW WEIGHT (lbs)", f"{df_full['NEW WEIGHT lbs'].sum():.2f}")
    with col2:
        st.metric("Suma NEW WEIGHT (kg)", f"{df_full['NEW WEIGHT kgs'].sum():.2f}")

    st.subheader("📦 Tabla completa (CAT)")
    st.dataframe(df_full, use_container_width=True, height=420)

    st.subheader("📦 Solo piezas ajustadas (CAT)")
    st.dataframe(df_adjusted, use_container_width=True, height=320)

    st.subheader("✅ Validación – Invoice vs GR vs Nuevo")
    st.dataframe(validation_df, use_container_width=True, height=420)

    # Descargar Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="SUMMARY", index=False)
        df_full.to_excel(writer, sheet_name="CAT_FULL", index=False)
        df_adjusted.to_excel(writer, sheet_name="CAT_ADJUSTED", index=False)
        validation_df.to_excel(writer, sheet_name="VALIDATION", index=False)
    output.seek(0)

    st.download_button(
        "⬇️ Descargar Excel",
        data=output,
        file_name="cat_adjustment_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
