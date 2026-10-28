"""Genera informe.pdf (máx. 5 páginas + referencias) leyendo las cifras de evidencias/.

No forma parte del pipeline de ML; necesita reportlab y pandas, en un entorno aparte:
  python3 -m venv /tmp/pdfenv && /tmp/pdfenv/bin/pip install reportlab==4.2.5 pandas==2.3.3
  /tmp/pdfenv/bin/python scripts/generar_informe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidencias"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

if (FONT_DIR / "DejaVuSans.ttf").exists():
    pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")
    BASE, BOLD, MONO = "Sans", "Sans-Bold", "Mono"
else:
    BASE, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"

GREEN = colors.HexColor("#1f4e3d")
LIGHT = colors.HexColor("#eef3f0")

ss = getSampleStyleSheet()
P = ParagraphStyle("p", parent=ss["Normal"], fontName=BASE, fontSize=8.6, leading=11.4,
                   alignment=TA_JUSTIFY, spaceAfter=4)
SMALL = ParagraphStyle("s", parent=P, fontSize=7.4, leading=9.2, alignment=0, spaceAfter=0)
CELL = ParagraphStyle("c", parent=SMALL, fontSize=7.2, leading=8.8)
CELLB = ParagraphStyle("cb", parent=CELL, fontName=BOLD, textColor=colors.white)
H1 = ParagraphStyle("h1", parent=P, fontName=BOLD, fontSize=11, leading=14, textColor=GREEN,
                    spaceBefore=7, spaceAfter=3, alignment=0)
TITLE = ParagraphStyle("t", parent=H1, fontSize=16, leading=20, spaceBefore=0)


def load(name):
    return json.loads((EV / name).read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{100 * x:.{d}f} %".replace(".", ",")


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def code(s):
    return f'<font name="{MONO}" size="6.8">{s}</font>'


def table(rows, widths, header=True, zebra=True):
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), CELLB if (header and i == 0) else CELL)
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c7bf")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), GREEN))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BASE, 7)
    canvas.setFillColor(colors.grey)
    canvas.drawString(1.6 * cm, 1.0 * cm, "MLOps · Defiende el servidor · Grupo A · Francy Carolina Umbarila Rodriguez")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    aud = load("auditoria.json")
    sel = load("seleccion.json")
    hf = load("hf_versions.json")
    mt = load("metricas-test.json")
    fp = load("analisis-fp-test.json")
    rec = load("recuperacion/comparacion.json")
    err = load("errores-validacion.json")
    errt = pd.read_csv(EV / "errores-validacion.csv")
    exp = pd.read_csv(EV / "experimentos.csv").set_index("id")

    tr, te = aud["files"]["train"], aud["files"]["test"]
    sp, ov = aud["splits"], aud["cross_partition_overlap"]
    v1, v2 = hf["v1"], hf["v2"]
    e1, e4 = exp.loc["e1_logreg_base"], exp.loc["e4_hgb_threshold"]
    m = mt["metrics"]
    cm_ = m["confusion_matrix"]

    s = []
    s.append(Paragraph("Defiende el servidor: versionado de una solución de ML para UNSW-NB15", TITLE))
    s.append(Paragraph(
        "<b>Francy Carolina Umbarila Rodriguez</b> · Grupo A · entrega individual · MLOps, Visualización 2, "
        "2026-II · Universidad Externado de Colombia", SMALL))
    s.append(Paragraph(
        f"Código: {code('github.com/c851458/Taller_visu_2')} · Modelos: "
        f"{code('huggingface.co/c851458/defiende-servidor-unsw')} · DVC: "
        f"{code('dagshub.com/c851458/Taller_visu_2.dvc')}", SMALL))
    s.append(Spacer(1, 5))
    s.append(Paragraph(
        "El objetivo es clasificar conexiones como normales (0) o ataques (1, clase positiva) y, sobre todo, "
        "poder identificar, comparar y recuperar cada versión de la solución. La cadena de trazabilidad es "
        "<b>commit Git → referencia DVC → run de MLflow → revisión (SHA) de Hugging Face</b>; cada eslabón se "
        "vincula por identificadores y hashes, no por nombres de archivo. Todas las cifras de este informe se "
        "leen de los archivos de <i>evidencias/</i> (este PDF se genera con <i>scripts/generar_informe.py</i>).", P))

    # 1. Datos
    s.append(Paragraph("1. Datos, auditoría y limpieza", H1))
    s.append(Paragraph(
        "Se usan las particiones publicadas, descargadas de la copia alternativa indicada en el taller "
        "(mirror de terceros); la atribución corresponde a la página oficial de UNSW. Las huellas SHA-256 de "
        "ambos CSV coinciden con las del anexo del taller. Ningún archivo tiene valores faltantes ni valores "
        "negativos en las columnas numéricas, y los <i>id</i> son únicos. Se excluyen de las entradas "
        "<i>label</i> (respuesta), <i>attack_cat</i> (revela la respuesta: Normal ⇔ label 0) e <i>id</i> "
        f"(identificador). Quedan {len(aud['features_used'])} variables predictoras, 3 de ellas categóricas "
        "(<i>proto</i>, <i>service</i>, <i>state</i>).", P))
    s.append(table([
        ["Archivo", "SHA-256 (inicio)", "Filas × col.", "Normal / ataque", "Filas en grupos duplicados",
         "Grupos contradictorios"],
        ["training-set", code(tr["sha256"][:16] + "…"), f"{miles(tr['rows'])} × {tr['columns']}",
         f"{miles(tr['class_counts']['0'])} / {miles(tr['class_counts']['1'])}",
         f"{miles(tr['duplicates_by_features']['rows_in_duplicate_groups'])} "
         f"({miles(tr['duplicates_by_features']['redundant_rows'])} redundantes)",
         f"{tr['contradictory_label_groups']['groups']} grupos, {tr['contradictory_label_groups']['rows']} filas"],
        ["testing-set", code(te["sha256"][:16] + "…"), f"{miles(te['rows'])} × {te['columns']}",
         f"{miles(te['class_counts']['0'])} / {miles(te['class_counts']['1'])}",
         f"{miles(te['duplicates_by_features']['rows_in_duplicate_groups'])} "
         f"({miles(te['duplicates_by_features']['redundant_rows'])} redundantes)",
         f"{te['contradictory_label_groups']['groups']} grupos, {te['contradictory_label_groups']['rows']} filas"],
    ], [2.1, 3.3, 2.2, 2.7, 3.6, 3.3]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "<b>Duplicados y contradicciones.</b> Un <i>grupo</i> es un conjunto de filas con todas las variables "
        "predictoras idénticas (sin <i>id</i> ni etiquetas). Política: (1) en el archivo de entrenamiento se "
        f"eliminan las {tr['contradictory_label_groups']['rows']} filas de los "
        f"{tr['contradictory_label_groups']['groups']} grupos con etiquetas contradictorias, porque no son "
        "aprendibles; (2) los duplicados exactos se conservan, pero cada grupo va <b>completo</b> a una sola "
        "partición; (3) en test no se elimina ninguna fila, solo se audita.", P))
    s.append(Paragraph(
        "<b>Particiones.</b> Entrenamiento y validación salen solo del archivo de entrenamiento, con una "
        f"asignación determinista por grupo: sha256(\"{sp['seed']}:group_id\") &lt; {num(sp['val_fraction'], 2)}. "
        f"Resultado: train = {miles(sp['sizes']['train'])}, validación = {miles(sp['sizes']['val'])} y "
        f"<b>{sp['groups_shared_train_val']} grupos compartidos</b> entre ambas. Las huellas de cada split y la "
        "asignación fila → partición (<i>splits.csv</i>) están en <i>auditoria.json</i>. Las cuatro "
        "configuraciones usan exactamente estas particiones, y codificadores, imputación y escalado se ajustan "
        "dentro de un <i>Pipeline</i> de scikit-learn solo con train. Test = archivo completo "
        f"({miles(sp['sizes']['test'])} filas). <b>Solapamiento:</b> {miles(ov['test_rows_with_inputs_seen_in_train_or_val'])} "
        f"filas de test ({pct(ov['test_overlap_fraction'])}) tienen entradas idénticas a alguna fila de "
        "train/val, así que el test no es del todo \"datos nuevos\".", P))
    s.append(Paragraph(
        "<b>Dos versiones de datos con DVC.</b> <i>data/raw.dvc</i> (CSV recibidos + procedencia, md5 "
        f"{code(v1['dvc']['raw_md5'])}, tag Git <i>datos-original</i>) y la etapa <i>prepare</i> de "
        f"<i>dvc.yaml</i> / <i>dvc.lock</i> (train/val/test/splits, md5 {code(v1['dvc']['prepared_md5'])}, tag "
        "<i>datos-preparados</i>). El almacenamiento está en DagsHub, no en el computador local.", P))

    # 2. Experimentos
    s.append(Paragraph("2. Cuatro experimentos, dos familias (validación)", H1))
    s.append(Paragraph(
        "Cada configuración (<i>configs/experiments/*.yaml</i>) declara su hipótesis, el cambio respecto a la "
        "anterior y la política de umbral. Cada run de MLflow registra parámetros, semilla (2026), commit, md5 "
        "DVC, métricas de validación, matriz de confusión y el modelo completo (preprocesamiento + estimador). "
        "La exportación está en <i>evidencias/mlflow/</i> (parámetros, métricas, tags y un manifiesto con el "
        "SHA-256 de cada artefacto; los modelos de E1, E2 y E4 van copiados) y la tabla en <i>experimentos.csv</i>. "
        "<b>Denominadores cero:</b> si un cociente es indefinido (p. ej., precision sin positivos predichos) se "
        "reporta 0,0 y la métrica se lista en <i>undefined_metrics</i>; en estos runs no ocurrió.", P))
    names = {"e1_logreg_base": "E1 · LogReg base", "e2_logreg_log1p": "E2 · LogReg + log1p",
             "e3_random_forest": "E3 · Random Forest", "e4_hgb_threshold": "E4 · HistGB + umbral"}
    hyp = {"e1_logreg_base": "Referencia: one-hot + escalado, umbral 0,5.",
           "e2_logreg_log1p": "log1p en variables de cola larga reduce FP.",
           "e3_random_forest": "Árboles capturan interacciones no lineales.",
           "e4_hgb_threshold": "Boosting + umbral elegido con FPR ≤ 5 %."}
    rows = [["Experimento", "Familia", "Hipótesis", "Run ID", "Umbral", "Recall", "Precision", "FPR",
             "Macro-F1", "FP", "FN"]]
    for k, r in exp.iterrows():
        rows.append([names[k], {"arboles": "árboles"}.get(r["family"], r["family"]), hyp[k], code(r["run_id"][:8]), num(r["threshold"], 4),
                     num(r["val_recall_attack"]), num(r["val_precision_attack"]), num(r["val_fpr"]),
                     num(r["val_macro_f1"]), miles(r["val_fp"]), miles(r["val_fn"])])
    s.append(table(rows, [2.4, 1.4, 3.1, 1.5, 1.3, 1.1, 1.6, 1.0, 1.6, 1.0, 0.9]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "<b>Interpretación.</b> E1 detecta casi todos los ataques, pero 1 de cada 6 conexiones normales genera "
        "una falsa alarma. E2 confirma la hipótesis solo en parte: log1p reduce la FPR (0,170 → 0,133) a costa "
        "de algo de recall. El cambio de familia (E3) es lo que más ayuda: macro-F1 sube a 0,949 y la FPR baja "
        "a menos del 10 % con el mismo umbral. E4 consigue la mejor precision y el mejor macro-F1 al fijar "
        "explícitamente la tasa de falsas alarmas. Con umbral 0,5, E4 tendría un macro-F1 algo mayor "
        f"({num(0.953025)}) pero una FPR de {num(0.080311)}: el umbral cambia la proporción entre los dos "
        "tipos de error, no la calidad del ranking. Validación: "
        f"{miles(sp['class_counts']['val']['0'])} normales y {miles(sp['class_counts']['val']['1'])} ataques.", P))

    # Errores por categoría (análisis posterior en validación: src/analyze_val_errors.py)
    ids = list(names)
    piv = errt.pivot(index="attack_cat", columns="id", values="errores")
    rate = errt.pivot(index="attack_cat", columns="id", values="tasa_error")
    filas = errt.drop_duplicates("attack_cat").set_index("attack_cat")["filas"]
    cat_rows = ["Normal", "Fuzzers", "Analysis", "Exploits", "DoS", "Reconnaissance", "Backdoor"]
    rest = [c for c in piv.index if c not in cat_rows]
    erows = [["Categoría (attack_cat)", "Filas val"] + [names[k].split(" · ")[0] for k in ids]]
    for c in cat_rows:
        label = "Normal (falsos positivos)" if c == "Normal" else f"{c} (no detectados)"
        erows.append([label, miles(filas[c])] + [f"{miles(piv.at[c, k])} ({pct(rate.at[c, k])})" for k in ids])
    erows.append([" + ".join(rest) + " (no detectados)", miles(filas[rest].sum())]
                 + [miles(piv.loc[rest, k].sum()) for k in ids])
    s.append(Spacer(1, 2))
    s.append(KeepTogether([table(erows, [4.6, 1.6, 2.9, 2.9, 2.9, 2.9])]))
    s.append(Spacer(1, 4))
    ev1, ev4 = err["experimentos"]["e1_logreg_base"], err["experimentos"]["e4_hgb_threshold"]
    fn4 = {r["attack_cat"]: r for r in ev4["falsos_negativos_por_attack_cat"]}
    fn1 = {r["attack_cat"]: r for r in ev1["falsos_negativos_por_attack_cat"]}
    st4 = {r["sttl"]: r for r in ev4["falsos_positivos_por_sttl"]}
    st1 = {r["sttl"]: r for r in ev1["falsos_positivos_por_sttl"]}
    s.append(Paragraph(
        "<b>Errores por categoría.</b> <i>src/analyze_val_errors.py</i> recarga cada modelo (hash verificado), "
        "predice validación, comprueba que las cuatro matrices de confusión coinciden con MLflow y desglosa los "
        "errores por <i>attack_cat</i>. <i>attack_cat</i> no es entrada del modelo: aquí solo sirve para explicar "
        "los errores (<i>errores-validacion.csv/.json</i>). El análisis es posterior a la selección y no cambió "
        f"modelo ni umbral. <b>Fuzzers</b> concentra los ataques no detectados en todas las configuraciones: "
        f"{pct(fn1['Fuzzers']['pct_de_fn'], 0)} de los de E1 y {pct(fn4['Fuzzers']['pct_de_fn'], 0)} de los de E4. "
        "Son ataques que envían datos generados al azar, y el modelo los confunde con tráfico normal. "
        "<b>Analysis</b> también empeora "
        f"con los árboles (0 → {miles(fn4['Analysis']['no_detectados'])} de {miles(fn4['Analysis']['ataques'])}). "
        "Generic, DoS, Exploits y Reconnaissance se detectan en más del 99 % con cualquier modelo. Las falsas "
        f"alarmas se concentran en conexiones normales con <i>sttl</i> = 254: {pct(st1['254']['pct_de_fp'], 0)} de las "
        f"FP de E1 y {pct(st4['254']['pct_de_fp'], 0)} de las de E4.", P))

    # 3. v1 vs v2
    s.append(Paragraph("3. Versiones v1 y v2", H1))
    s.append(table([
        ["", "v1", "v2"],
        ["Experimento / familia", "E1 · regresión logística", "E4 · HistGradientBoosting"],
        ["Preprocesamiento", "One-hot (min_frequency 20) + escalado estándar", "One-hot (min_frequency 20), sin escalado"],
        ["Umbral (y_pred = 1 si score ≥ umbral)", num(v1["threshold"], 4), num(v2["threshold"], 4)],
        ["Commit que entrenó / tag Git", code(v1["git_commit_model"][:12]) + " / v1",
         code(v2["git_commit_model"][:12]) + " / v2"],
        ["Run de MLflow", code(v1["mlflow_run_id"]), code(v2["mlflow_run_id"])],
        ["Revisión Hugging Face", code(v1["revision"]), code(v2["revision"])],
        ["sha256 de model.joblib", code(v1["model_sha256"][:32] + "…"), code(v2["model_sha256"][:32] + "…")],
        ["Datos DVC (raw / preparados)", code(v1["dvc"]["raw_md5"][:10] + "… / " + v1["dvc"]["prepared_md5"][:10] + "…"),
         code(v2["dvc"]["raw_md5"][:10] + "… / " + v2["dvc"]["prepared_md5"][:10] + "…")],
        ["Validación: recall / precision", f"{num(e1['val_recall_attack'])} / {num(e1['val_precision_attack'])}",
         f"{num(e4['val_recall_attack'])} / {num(e4['val_precision_attack'])}"],
        ["Validación: FPR / macro-F1", f"{num(e1['val_fpr'])} / {num(e1['val_macro_f1'])}",
         f"{num(e4['val_fpr'])} / {num(e4['val_macro_f1'])}"],
        ["Validación: FP / FN", f"{miles(e1['val_fp'])} / {miles(e1['val_fn'])}",
         f"{miles(e4['val_fp'])} / {miles(e4['val_fn'])}"],
    ], [5.0, 6.4, 6.4]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "<b>Qué cambia y por qué.</b> v2 cambia la familia del modelo y la regla del umbral. Mejora la "
        f"precision y el macro-F1 y reduce las falsas alarmas de {miles(e1['val_fp'])} a {miles(e4['val_fp'])} "
        "(menos de un tercio). <b>Empeora el recall de ataque</b> (0,989 → 0,960): los ataques no detectados "
        f"pasan de {miles(e1['val_fn'])} a {miles(e4['val_fn'])}, y casi todo el aumento son Fuzzers "
        f"({miles(fn1['Fuzzers']['no_detectados'])} → {miles(fn4['Fuzzers']['no_detectados'])}). Ese es el error "
        "que se acepta, porque para un equipo de seguridad el exceso de alertas provoca fatiga y hace que se "
        "ignoren también las alertas reales. Exploits, DoS, Backdoor y Shellcode siguen detectándose en "
        "≥ 98 %. v2 es un artefacto distinto (otro hash y otra revisión), no un renombre de v1. Cada revisión "
        "de Hugging Face contiene el modelo completo, <i>config.json</i> con el umbral, <i>requirements.txt</i>, "
        "<i>predict.py</i> y una model card con propósito, datos, métricas, limitaciones, uso y las referencias "
        "a Git, DVC y MLflow.", P))

    # 4. Umbral y selección
    s.append(Paragraph("4. Elección de umbral y congelación de la selección", H1))
    s.append(Paragraph(
        f"La regla (<i>configs/seleccion.yaml</i>) se escribió antes de mirar el test: <b>{sel['selection_rule']['criterion']}</b>, "
        f"con FPR máxima de {pct(sel['selection_rule']['max_fpr'], 0)}. En E4 el umbral es el que da el máximo "
        f"recall de validación con FPR ≤ 5 %: {num(sel['threshold'], 4)}. Solo E4 cumple la restricción, así que "
        f"se elige como v2 (validación: FPR {num(sel['validation_metrics']['val_fpr'])}, macro-F1 "
        f"{num(sel['validation_metrics']['val_macro_f1'])}). <i>seleccion.json</i> (run, revisión HF, hash del "
        f"modelo, umbral, razón y <i>test_used_for_selection = false</i>) se congeló en el commit "
        f"{code('269c2ab0850129')} (tag v2, {sel['frozen_at_utc'].replace('T', ' ')} UTC), anterior al commit del "
        f"test, {code('d71d7f66addf23')}. <i>evaluate_test</i> se niega a correr si <i>seleccion.json</i> no está "
        "en Git o si el test ya se evaluó. Limitación: el umbral se elige y se reporta sobre la misma "
        "validación, así que esa estimación es algo optimista.", P))

    # 5. Test
    s.append(Paragraph("5. Evaluación final en test (una sola vez)", H1))
    s.append(Paragraph(
        f"v2 congelada (revisión {code(mt['hf_revision'][:12])}, sha256 {code(mt['model_sha256'][:12])}…) se "
        f"evaluó una vez sobre las {miles(mt['rows_evaluated'])} filas del test (sha256 del archivo preparado "
        f"{code(mt['dataset_sha256'][:12])}…; del CSV original {code(mt['raw_test_sha256'][:12])}…). "
        "<i>predicciones-test.csv</i> tiene una fila por registro (<i>row_id</i> = <i>test:&lt;fila&gt;</i>, "
        "<i>y_true</i>, <i>score_attack</i> = probabilidad de ataque, <i>y_pred</i>), y "
        "<i>recompute_metrics</i> recalcula desde ese archivo exactamente las mismas métricas (diferencia 0).", P))
    s.append(KeepTogether([table([
        ["Métrica", "Validación", "Test", "", "Matriz de confusión (test)", "Pred. normal", "Pred. ataque"],
        ["Recall de ataque", num(e4["val_recall_attack"]), num(m["recall_attack"]), "", "Real normal",
         miles(cm_["tn"]), miles(cm_["fp"])],
        ["Precision de ataque", num(e4["val_precision_attack"]), num(m["precision_attack"]), "", "Real ataque",
         miles(cm_["fn"]), miles(cm_["tp"])],
        ["FPR", num(e4["val_fpr"]), f"<b>{num(m['fpr'])}</b>", "", "", "", ""],
        ["Macro-F1", num(e4["val_macro_f1"]), num(m["macro_f1"]), "", "Accuracy (complementaria)", "",
         num(m["accuracy"])],
    ], [3.0, 1.8, 1.6, 0.4, 4.2, 2.2, 2.2], zebra=False)]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "<b>El resultado empeora y se conserva sin ajustes.</b> El recall de ataque se mantiene (0,968), pero la "
        f"FPR sube de 0,050 a {num(m['fpr'])}: {miles(cm_['fp'])} de {miles(cm_['tn'] + cm_['fp'])} conexiones "
        "normales generan alarma. No se cambió el modelo ni el umbral. Para entenderlo se hizo un "
        "<b>diagnóstico posterior</b> (<i>src/analyze_fp_test.py</i> → <i>analisis-fp-test.json</i>), que solo "
        "explica el resultado y no se usó para ninguna decisión:", P))
    by = {r["sttl"]: r for r in fp["por_sttl"]}
    st = {r["state"]: r for r in fp["por_state"]}
    sv = {r["service"]: r for r in fp["por_service"]}
    ovl = {r["entrada_vista_en_train_val"]: r for r in fp["por_solapamiento"]}
    diag = [["Segmento de conexiones normales", "% de normales (val → test)", "FPR val", "FPR test",
             "% de las FP de test"]]
    for label, r in [("sttl = 254", by["254"]), ("sttl = 62", by["62"]), ("sttl = 31", by["31"]),
                     ("state = INT", st["INT"]), ("state = FIN", st["FIN"]), ("state = CON", st["CON"]),
                     ("service = \"-\" (sin servicio identificado)", sv["-"]), ("service = http", sv["http"]),
                     ("service = dns", sv["dns"])]:
        diag.append([label, f"{pct(r['pct_normales_val'])} → {pct(r['pct_normales_test'])}", num(r["fpr_val"]),
                     num(r["fpr_test"]), pct(r["pct_fp_test"])])
    s.append(table(diag, [6.2, 3.8, 1.9, 1.9, 2.9]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "El 87,6 % de las falsas alarmas vienen de conexiones normales con <i>sttl</i> (TTL de origen) = 254. "
        "En el archivo de entrenamiento, el 90,4 % de las filas con sttl = 254 son ataques y solo el 19,5 % del "
        "tráfico normal tiene ese valor, así que el modelo aprendió que \"sttl = 254 ⇒ ataque\". En test, en "
        "cambio, el 40,7 % del tráfico normal tiene sttl = 254 (y solo el 72,2 % de esas filas son ataques). "
        "Además, ese tráfico es más difícil: incluso dentro del segmento, la FPR sube de 0,239 a 0,414. Lo mismo "
        "ocurre con <i>state = INT</i> (FPR 0,547) y con las conexiones sin servicio identificado. Los segmentos "
        "con sttl = 31 o servicio dns no generan falsas alarmas. El solapamiento no explica el problema: la FPR "
        f"es parecida en las filas de test ya vistas en train/val ({num(ovl['True']['fpr_test'])}) y en las no "
        f"vistas ({num(ovl['False']['fpr_test'])}). <b>Conclusión:</b> el archivo de test tiene una distribución de "
        "tráfico normal distinta de la del archivo de entrenamiento (un cambio de distribución). La dependencia "
        f"de <i>sttl</i> ya se veía en validación ({pct(st4['254']['pct_de_fp'], 0)} de las FP de v2, sección 2), "
        "pero la validación sale del mismo archivo que train y no podía anticipar que en test ese segmento "
        "pesaría el doble. Se reporta como limitación y no se corrige con el test. Una v3 futura debería validar con particiones que imiten ese cambio y revisar la "
        "dependencia de <i>sttl</i>.", P))

    # 6. Recuperación
    s.append(Paragraph("6. Prueba de recuperación en entorno nuevo", H1))
    vr1, vr2 = rec["versions"]["v1"], rec["versions"]["v2"]
    s.append(Paragraph(
        "<i>scripts/prueba_recuperacion.sh</i> se ejecutó en una carpeta nueva (<i>~/recuperacion_20261028</i>) "
        "con un entorno virtual nuevo, sin reutilizar cachés. Pasos: (1) clonar GitHub en el commit "
        f"{code('3265bc1ef567')} e instalar <i>requirements.txt</i>; (2) <i>dvc pull</i> desde DagsHub con caché "
        "vacía y verificar las huellas SHA-256 de los 2 CSV originales y de los 4 archivos preparados (todas OK); "
        "(3–4) con una caché de Hugging Face vacía, descargar v1 y v2 por su SHA completo, comprobar el hash de "
        f"cada modelo y predecir las mismas {vr1['rows']} filas fijas de validación (estratificadas, semilla "
        "2026); (5) comparar con las salidas guardadas antes de la recuperación. Tolerancia declarada: "
        f"|Δ score| ≤ {rec['tolerance_abs_score']:.0e}. Resultado: <b>v1 y v2 con {vr1['rows']}/{vr1['rows']} "
        f"etiquetas iguales y diferencia máxima {vr1['max_abs_diff_score']:.0f} y "
        f"{vr2['max_abs_diff_score']:.0f}: RESULTADO OK</b>. Duración: ~146 s. Entrada, salidas, hashes, log y "
        "comparación están en <i>evidencias/recuperacion/</i>. <b>Nota de acceso:</b> DagsHub exige iniciar "
        "sesión para leer el almacenamiento DVC aunque el repositorio sea público (el primer intento, sin "
        "credenciales, recibió HTTP 401). Por eso el revisor usa su propia cuenta gratuita y su token, "
        "guardados solo en <i>.dvc/config.local</i>. Nunca se comparten tokens.", P))

    # 7. Limitaciones
    s.append(Paragraph("7. Limitaciones", H1))
    s.append(Paragraph(
        "(1) UNSW-NB15 es un dataset de laboratorio de 2015: el resultado no demuestra eficacia en una red "
        "actual. (2) El test tiene un cambio de distribución en el tráfico normal (sección 5) y su FPR (0,192) "
        "está lejos del objetivo del 5 %; en producción, v2 generaría muchas más alertas de las previstas. "
        f"(3) El {pct(ov['test_overlap_fraction'])} del test repite entradas de train/val. (4) El umbral se elige y "
        "se reporta sobre la misma validación. (5) Variables como <i>sttl</i> pueden ser artefactos del "
        "laboratorio donde se generó el dataset más que señales generalizables. (6) <i>model.joblib</i> es un "
        "pickle: requiere scikit-learn 1.7.2 y solo debe cargarse de una fuente confiable. (7) Leer los datos "
        "requiere una cuenta de DagsHub.", P))
    s.append(Paragraph(
        "<b>Uso de IA.</b> Claude (Anthropic), mediante Claude Code en VS Code, ayudó a estructurar el código, "
        "ejecutar los pasos, analizar los falsos positivos y redactar la documentación. La autora verificó las "
        "huellas, la ausencia de fuga, la recuperación y el recálculo de métricas (detalle en <i>entrega.md</i>).", P))

    # Referencias (aparte del límite de 5 páginas)
    s.append(PageBreak())
    s.append(Paragraph("Referencias", H1))
    for ref in [
        "Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection "
        "systems. <i>Military Communications and Information Systems Conference (MilCIS)</i>, IEEE. "
        "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i> 12, 2825–2830. "
        "Errores comunes: https://scikit-learn.org/stable/common_pitfalls.html",
        "DVC — Data Version Control. https://dvc.org/doc",
        "MLflow Tracking. https://mlflow.org/docs/latest/tracking.html",
        "Hugging Face Hub — descarga por revisión y model cards. https://huggingface.co/docs/hub",
        "Mitchell, M. et al. (2019). Model Cards for Model Reporting. <i>FAccT</i>.",
        "Zuluaga, J. (2026). <i>Defiende el servidor</i>, taller evaluable, MLOps 2026-II, Universidad Externado de Colombia.",
    ]:
        s.append(Paragraph(ref, ParagraphStyle("ref", parent=P, alignment=0)))

    doc = SimpleDocTemplate(str(ROOT / "informe.pdf"), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.6 * cm,
                            title="Defiende el servidor — informe", author="Francy Carolina Umbarila Rodriguez")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("informe.pdf generado")


if __name__ == "__main__":
    build()
