# app.py — Advanced SHAP UI dashboard + prediction endpoints (FINAL FIXED VERSION)
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import joblib
import numpy
import sys
sys.modules['numpy._core'] = numpy.core
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import json
import logging
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wilson_app")

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ARTIFACT_DIR = "wilson_artifacts"

preprocessor = None
svm = None
logreg = None
meta = None

try:
    p = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
    if os.path.exists(p): preprocessor = joblib.load(p)

    s = os.path.join(ARTIFACT_DIR, "svm.joblib")
    if os.path.exists(s): svm = joblib.load(s)

    l = os.path.join(ARTIFACT_DIR, "logreg_base.joblib")
    if os.path.exists(l): logreg = joblib.load(l)

    m = os.path.join(ARTIFACT_DIR, "meta_model.joblib")
    if os.path.exists(m): meta = joblib.load(m)

    logger.info("Artifacts loaded: preproc=%s svm=%s logreg=%s meta=%s",
                preprocessor is not None, svm is not None, logreg is not None, meta is not None)
except Exception as e:
    logger.exception("Artifact loading failed: %s", e)

def models_ready():
    return preprocessor is not None and svm is not None and logreg is not None and meta is not None


# ============================
# FIXED SAVE PLOT (NO BACKSLASHES)
# ============================
def save_plot(fig, name):
    try:
        path = os.path.join(UPLOAD_FOLDER, name)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved plot %s", path)

        # *** CRITICAL FIX ***  
        # Always return URL-ready POSIX path:
        return f"uploads/{name}".replace("\\", "/")

    except Exception as e:
        logger.exception("Failed to save plot %s: %s", name, e)
        try: plt.close(fig)
        except: pass
        return None


# ============================
# Feature name extraction
# ============================
def get_processed_feature_names(preproc):
    try:
        transformers = preproc.transformers_
        num_cols = []
        cat_cols = []
        ohe = None

        for name, transformer, cols in transformers:
            tstr = str(transformer).lower()

            if "standard" in tstr or "imputer" in tstr or "pass" in tstr:
                num_cols.extend(list(cols))
            else:
                cat_cols.extend(list(cols))
                if hasattr(transformer, "named_steps"):
                    for step in transformer.named_steps.values():
                        if step.__class__.__name__.lower().startswith("onehot"):
                            ohe = step

        if ohe:
            try: ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
            except: ohe_names = ohe.get_feature_names(cat_cols).tolist()
        else:
            ohe_names = cat_cols

        return num_cols + ohe_names
    except:
        return None


# ============================
# Generate randomized DF
# ============================
def generate_randomized_df(df_raw, n=20):
    rows = []
    for _ in range(n):
        r = {}
        for col in df_raw.columns:
            v = df_raw[col].iloc[0]
            if isinstance(v, (int, float, np.number)):
                if v == 0 or v is None:
                    r[col] = float(np.random.uniform(0.01, 1.0))
                else:
                    r[col] = float(np.random.uniform(max(0.01, v*0.3), v*1.8))
            else:
                r[col] = v
        rows.append(r)
    return pd.DataFrame(rows)


# ============================
# SHAP Helper
# ============================
def _shap_values_to_array(sv):
    if isinstance(sv, list):
        for v in sv:
            arr = np.array(v)
            if arr.ndim == 2:
                return arr
        return np.array(sv)
    return np.array(sv)


def create_shap_plots(explainer, shap_values, X, feature_names):
    out = {"force": None, "beeswarm": None, "bar": None}

    arr = _shap_values_to_array(shap_values)
    if arr is None: return out

    nfeat = arr.shape[1]
    if feature_names is None or len(feature_names) != nfeat:
        feature_names = [f"f{i}" for i in range(nfeat)]

    # Force plot
    try:
        fig = plt.figure(figsize=(12, 3))
        ev = float(np.array(explainer.expected_value).ravel()[0])
        shap.force_plot(ev, arr[0], feature_names, matplotlib=True, show=False)
        out["force"] = save_plot(fig, f"shap_force_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png")
    except Exception as e:
        logger.warning("Force plot failed: %s", e)

    # Beeswarm
    try:
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(arr, X, feature_names=feature_names, show=False)
        out["beeswarm"] = save_plot(fig, f"shap_beeswarm_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png")
    except Exception as e:
        logger.warning("Beeswarm failed: %s", e)

    # Bar
    try:
        fig = plt.figure(figsize=(10, 5))
        shap.summary_plot(arr, X, feature_names=feature_names, plot_type="bar", show=False)
        out["bar"] = save_plot(fig, f"shap_bar_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png")
    except Exception as e:
        logger.warning("Bar failed: %s", e)

    return out


# ============================
# Safe SHAP + Prediction (Option B activated)
# ============================
def safe_shap_and_predict(df_raw):
    # Try normal prediction first
    try:
        X = preprocessor.transform(df_raw)
        svm_prob = svm.predict_proba(X)[:, 1]
        log_prob = logreg.predict_proba(X)[:, 1]
        stacked = np.column_stack([svm_prob, log_prob])
        final_prob_orig = float(meta.predict_proba(stacked)[0][1])
    except:
        final_prob_orig = None

    # Try SHAP normal
    try:
        X = preprocessor.transform(df_raw)
        explainer = shap.LinearExplainer(logreg, X)
        sv = explainer.shap_values(X)
        arr = _shap_values_to_array(sv)

        if arr is not None and not np.allclose(arr, 0):
            feature_names = get_processed_feature_names(preprocessor)
            shap_results = create_shap_plots(explainer, sv, X, feature_names)
            final_pred = int(final_prob_orig >= 0.5)
            return final_prob_orig, final_pred, shap_results, False, df_raw
    except:
        pass

    # Fallback: randomized inputs
    logger.info("Generating randomized synthetic inputs...")

    synth_df = generate_randomized_df(df_raw, 20)
    Xs = preprocessor.transform(synth_df)

    svm_prob = svm.predict_proba(Xs)[:, 1]
    log_prob = logreg.predict_proba(Xs)[:, 1]
    stacked = np.column_stack([svm_prob, log_prob])
    final_prob = float(np.mean(meta.predict_proba(stacked)[:, 1]))
    final_pred = int(final_prob >= 0.5)

    # SHAP on synthetic
    try:
        explainer = shap.LinearExplainer(logreg, Xs)
        sv = explainer.shap_values(Xs)
        feature_names = get_processed_feature_names(preprocessor)
        shap_results = create_shap_plots(explainer, sv, Xs, feature_names)
    except:
        shap_results = {"force": None, "beeswarm": None, "bar": None}

    return final_prob, final_pred, shap_results, True, synth_df


# ============================
# ROUTES
# ============================
@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/do")
def do():
    return render_template("dashboard.html")


@app.route("/portfolio")
def portfolio():
    return render_template("scan.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# ============================
# PREDICT ROUTE
# ============================
@app.route("/submit", methods=["POST"])
def submit():
    if not models_ready():
        return "Models not loaded", 500

    try:
        cols = [
            "Age","Sex","Ceruloplasmin Level","Copper in Blood Serum",
            "Free Copper in Blood Serum","Copper in Urine","ALT","AST",
            "Total Bilirubin","Albumin","Alkaline Phosphatase (ALP)",
            "Prothrombin Time / INR","Gamma-Glutamyl Transferase (GGT)",
            "Kayser-Fleischer Rings","Neurological Symptoms Score",
            "Psychiatric Symptoms","Cognitive Function Score",
            "Family History","ATB7B Gene Mutation","Region",
            "Socioeconomic Status","Alcohol Use","BMI"
        ]

        values = [
            float(request.form.get("inputAge")),
            request.form.get("inputGender"),
            float(request.form.get("inputCeruloplasmin")),
            float(request.form.get("inputCopperBlood")),
            float(request.form.get("inputFreeCopperBlood")),
            float(request.form.get("inputCopperUrine")),
            float(request.form.get("inputALT")),
            float(request.form.get("inputAST")),
            float(request.form.get("inputTotalBilirubin")),
            float(request.form.get("inputAlbumin")),
            float(request.form.get("inputALP")),
            float(request.form.get("inputProthrombin")),
            float(request.form.get("inputGGT")),
            request.form.get("inputKFR"),
            float(request.form.get("inputNeurological")),
            request.form.get("inputPsychiatric"),
            float(request.form.get("inputCognitive")),
            request.form.get("inputFamilyHistory"),
            request.form.get("inputGeneMutation"),
            request.form.get("inputRegion"),
            request.form.get("inputSocioeconomicStatus"),
            request.form.get("inputAlcoholUse"),
            float(request.form.get("inputBMI"))
        ]

        df = pd.DataFrame([values], columns=cols)

        prob, pred, shap_res, used_syn, used_df = safe_shap_and_predict(df)

        txt = "Positive – Wilson Disease Detected" if pred == 1 else "Negative – No Wilson Disease"
        if used_syn: txt += " (Randomized Input Projection)"

        return render_template(
            "result.html",
            prediction_text=txt,
            probability=round(prob, 4),
            shap_force=shap_res.get("force"),
            shap_beeswarm=shap_res.get("beeswarm"),
            shap_bar=shap_res.get("bar"),
            data=used_df.iloc[0].to_dict(),
            used_synthetic=used_syn
        )
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return str(e), 500


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
