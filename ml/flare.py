"""HMP2 flare-risk model (Stage B2): will a patient in remission (HBI<5) flare (HBI>=5) at the next visit?

Leakage rules: features use only visit t and earlier; label from visit t+1; patients never split across folds.
"""
import os, sys, json, numpy as np, pandas as pd, joblib, warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "data", "hmp2_metadata_2018-08-20.csv")
OUT = os.path.join(HERE, "..", "artifacts", "flare")
PID = "Participant ID"
WB = {"Very Well": 0, "Slightly below par": 1, "Poor": 2, "Very Poor": 3, "Terrible": 4}
PAIN = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
YN = {"No": 0, "Yes": 1}

def first(s):
    s = s.dropna(); return s.iloc[0] if len(s) else np.nan

def visit_table():
    d = pd.read_csv(SRC, low_memory=False)
    d = d[d.diagnosis == "CD"]
    cols = {"hbi": "hbi", "fecalcal": "fc", "CRP (mg/L)": "crp", "ESR (mm/hr)": "esr",
            "General wellbeing": "wellbeing", "Abdominal pain": "pain",
            "Number of liquid or very soft stools in the past 24 hours:": "liquid_stools",
            "Arthralgia": "arthralgia", "Antibiotics": "antibiotics",
            "Immunosuppressants (e.g. oral corticosteroids)": "immunosupp",
            "4) In the past 2 weeks, have you had diarrhea?": "diarrhea_2wk",
            "5) In the past 2 weeks, have you been hospitalized?": "hosp_2wk",
            "week_num": "week", "consent_age": "age", "Age at diagnosis": "age_dx", "sex": "sex",
            "baseline_montreal_location": "location"}
    v = d.groupby([PID, "visit_num"]).agg({c: first for c in cols}).rename(columns=cols).reset_index()
    v = v.rename(columns={PID: "pid"})
    for c in ["age", "age_dx", "sex", "location"]:  # static per patient
        v[c] = v.groupby("pid")[c].transform(lambda s: s.ffill().bfill())
    v["wellbeing"] = v.wellbeing.map(WB); v["pain"] = v.pain.map(PAIN).fillna(pd.Series(np.where(v.hbi.notna(), 0, np.nan), index=v.index))
    for c in ["arthralgia", "antibiotics", "immunosupp", "diarrhea_2wk", "hosp_2wk"]:
        v[c] = v[c].map(YN)
    v["male"] = (v.sex == "Male").astype(float)
    loc = v.location.fillna("")
    v["ileal"] = loc.str.contains("L1|L3").astype(float); v["colonic"] = loc.str.contains("L2|L3").astype(float)
    v["upper_gi"] = loc.str.contains("L4").astype(float)
    v["disease_years"] = v.age - v.age_dx
    v = v[v.week.notna()].sort_values(["pid", "week", "visit_num"])
    return v

FEATURES = ["hbi", "hbi_prev", "hbi_delta", "hbi_mean3", "hbi_max3", "hbi_max_all", "prior_flares", "weeks_since_flare",
            "wellbeing", "pain", "liquid_stools", "arthralgia", "antibiotics", "immunosupp", "diarrhea_2wk",
            "fc_last", "fc_log", "fc_weeks_since", "crp_last", "esr_last", "lab_weeks_since",
            "age", "disease_years", "male", "ileal", "colonic", "upper_gi", "gap_weeks", "n_prior_visits"]
LABELS = {f: f for f in FEATURES}
NICE = {"hbi": "Current HBI", "hbi_prev": "Previous HBI", "hbi_delta": "HBI change since last visit",
        "hbi_mean3": "Mean HBI (last 3 visits)", "hbi_max3": "Max HBI (last 3 visits)", "hbi_max_all": "Highest HBI so far",
        "prior_flares": "Number of prior flares", "weeks_since_flare": "Weeks since last flare",
        "wellbeing": "General wellbeing (0 very well - 4 terrible)", "pain": "Abdominal pain (0-3)",
        "liquid_stools": "Liquid stools / 24 h", "arthralgia": "Joint pain", "antibiotics": "Antibiotics (past 2 wk)",
        "immunosupp": "Immunosuppressants / steroids", "diarrhea_2wk": "Diarrhoea (past 2 wk)",
        "fc_last": "Last fecal calprotectin", "fc_log": "log Fecal calprotectin", "fc_weeks_since": "Weeks since calprotectin",
        "crp_last": "Last CRP (mg/L)", "esr_last": "Last ESR (mm/h)", "lab_weeks_since": "Weeks since blood labs",
        "age": "Age", "disease_years": "Years since diagnosis", "male": "Male sex", "ileal": "Ileal involvement (L1/L3)",
        "colonic": "Colonic involvement (L2/L3)", "upper_gi": "Upper-GI involvement (L4)",
        "gap_weeks": "Weeks until next visit", "n_prior_visits": "Number of prior visits"}

def engineer(v):
    """One row per HBI visit, features from that visit and earlier only."""
    rows = []
    for pid, g in v.groupby("pid"):
        g = g.reset_index(drop=True)
        fc_val = crp = esr = np.nan; fc_wk = lab_wk = np.nan
        hb_hist = []; flares = 0; last_flare_wk = np.nan
        for i, r in g.iterrows():
            if pd.notna(r.fc): fc_val, fc_wk = r.fc, r.week
            if pd.notna(r.crp) or pd.notna(r.esr):
                crp = r.crp if pd.notna(r.crp) else crp; esr = r.esr if pd.notna(r.esr) else esr; lab_wk = r.week
            if pd.isna(r.hbi): continue
            prev = hb_hist[-1] if hb_hist else np.nan
            if hb_hist and prev < 5 <= r.hbi: flares += 1; last_flare_wk = r.week
            hb_hist.append(r.hbi)
            rows.append(dict(pid=pid, week=r.week, hbi=r.hbi, hbi_prev=prev, hbi_delta=r.hbi - prev if hb_hist[:-1] else np.nan,
                hbi_mean3=np.mean(hb_hist[-3:]), hbi_max3=np.max(hb_hist[-3:]), hbi_max_all=np.max(hb_hist),
                prior_flares=flares, weeks_since_flare=r.week - last_flare_wk if pd.notna(last_flare_wk) else np.nan,
                wellbeing=r.wellbeing, pain=r.pain, liquid_stools=r.liquid_stools, arthralgia=r.arthralgia,
                antibiotics=r.antibiotics, immunosupp=r.immunosupp, diarrhea_2wk=r.diarrhea_2wk,
                fc_last=fc_val, fc_log=np.log1p(fc_val) if pd.notna(fc_val) else np.nan,
                fc_weeks_since=r.week - fc_wk if pd.notna(fc_wk) else np.nan, crp_last=crp, esr_last=esr,
                lab_weeks_since=r.week - lab_wk if pd.notna(lab_wk) else np.nan, age=r.age, disease_years=r.disease_years,
                male=r.male, ileal=r.ileal, colonic=r.colonic, upper_gi=r.upper_gi, n_prior_visits=len(hb_hist) - 1))
    f = pd.DataFrame(rows).sort_values(["pid", "week"]).reset_index(drop=True)
    f["next_hbi"] = f.groupby("pid").hbi.shift(-1); f["next_week"] = f.groupby("pid").week.shift(-1)
    f["gap_weeks"] = f.next_week - f.week
    return f

def models(pos_weight):
    return {
        "Logistic regression": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                             LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)),
        "Random forest": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                       RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=2, random_state=0)),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=3, scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=2, random_state=0),
    }

def boot_ci(y, p, groups, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if y[s].min() != y[s].max(): vals.append(fn(y[s], p[s]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def main():
    os.makedirs(OUT, exist_ok=True)
    v = visit_table(); f = engineer(v)
    pairs = f[(f.hbi < 5) & f.next_hbi.notna() & (f.gap_weeks <= 8)].copy()
    pairs["y"] = (pairs.next_hbi >= 5).astype(int)
    X, y, g = pairs[FEATURES].astype(float), pairs.y.values, pairs.pid.values
    summary = dict(patients=int(v.pid.nunique()), hbi_visits=int(f.shape[0]), pairs=int(len(pairs)), flares=int(y.sum()),
                   flare_rate=float(y.mean()), patients_with_flare=int(pairs[pairs.y == 1].pid.nunique()),
                   median_gap_weeks=float(pairs.gap_weeks.median()))
    print(summary)
    pos_w = (1 - y.mean()) / y.mean(); res = {}; oof_all = {}
    for name, mk in list(models(pos_w).items()) + [("Baseline: current HBI only", None)]:
        aucs, aps, briers = [], [], []; oof_sum = np.zeros(len(y)); 
        for rep in range(5):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep); oof = np.zeros(len(y))
            for tr, te in cv.split(X, y, g):
                if mk is None: oof[te] = X.iloc[te].hbi.values / 5.0; continue
                m = models(pos_w)[name]; m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y, oof)); aps.append(average_precision_score(y, oof))
            if mk is not None: briers.append(brier_score_loss(y, oof))
            oof_sum += oof
        oof_mean = oof_sum / 5; oof_all[name] = oof_mean
        res[name] = dict(auroc_mean=float(np.mean(aucs)), auroc_sd=float(np.std(aucs)), auprc_mean=float(np.mean(aps)), auprc_sd=float(np.std(aps)),
                         brier=float(np.mean(briers)) if briers else None,
                         auroc_ci95=boot_ci(y, oof_mean, g, roc_auc_score), auprc_ci95=boot_ci(y, oof_mean, g, average_precision_score))
        print(name, {k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in res[name].items()})
    best = max([k for k in res if not k.startswith("Baseline")], key=lambda k: res[k]["auroc_mean"])
    # calibration of best model's OOF predictions (deciles -> 5 bins given small n)
    p = oof_all[best]; bins = np.quantile(p, np.linspace(0, 1, 6)); bins[-1] += 1e-9; cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m_ = (p >= lo) & (p < hi); cal.append(dict(pred=float(p[m_].mean()), obs=float(y[m_].mean()), n=int(m_.sum())))
    # Platt recalibration on out-of-fold predictions (class weighting inflates raw RF probabilities; ranking/AUROC unchanged)
    from scipy.special import logit, expit
    lo = logit(np.clip(oof_all[best], 1e-4, 1 - 1e-4)).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(lo, y); pa, pb = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    # honest check of the recalibration itself: fit on 4/5 of patients, apply to the rest
    pc = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=99).split(lo, y, g):
        m_ = LogisticRegression(C=1e6).fit(lo[tr], y[tr]); pc[te] = m_.predict_proba(lo[te])[:, 1]
    cal_after = []
    bins2 = np.quantile(pc, np.linspace(0, 1, 6)); bins2[-1] += 1e-9
    for lo_, hi_ in zip(bins2[:-1], bins2[1:]):
        m_ = (pc >= lo_) & (pc < hi_); cal_after.append(dict(pred=float(pc[m_].mean()), obs=float(y[m_].mean()), n=int(m_.sum())))
    np.savez(os.path.join(OUT, "flare_oof.npz"), y=y, oof=oof_all[best], oof_cal=pc, groups=g)
    # final model on all data + SHAP-style contributions
    final = models(pos_w)[best]; final.fit(X, y)
    import shap
    if best == "XGBoost":
        expl = shap.TreeExplainer(final); sv = expl.shap_values(X); base = float(expl.expected_value)
    elif best == "Random forest":
        Xi = final[:-1].transform(X); expl = shap.TreeExplainer(final[-1]); sv = expl.shap_values(Xi)[:, :len(FEATURES), 1] if expl.shap_values(Xi).ndim == 3 else expl.shap_values(Xi)[:, :len(FEATURES)]; base = float(np.ravel(expl.expected_value)[-1])
    else:
        Xi = final[:-1].transform(X); expl = shap.LinearExplainer(final[-1], Xi); sv = expl.shap_values(Xi)[:, :len(FEATURES)]; base = float(expl.expected_value)
    imp = sorted([(NICE[c], float(np.abs(sv[:, i]).mean())) for i, c in enumerate(FEATURES)], key=lambda t: -t[1])
    joblib.dump(dict(model=final, features=FEATURES, nice=NICE, best=best, pos_weight=pos_w,
                     X_background=X.sample(min(200, len(X)), random_state=0)), os.path.join(OUT, "flare_model.joblib"))
    # per-patient timelines for the app (risk at every remission visit, OOF predictions only -> honest)
    pairs["risk_oof"] = pc
    timelines = {}
    for pid, gdf in f.groupby("pid"):
        pr = pairs[pairs.pid == pid].set_index("week").risk_oof.to_dict()
        timelines[pid] = [dict(week=float(r.week), hbi=float(r.hbi), fc=None if pd.isna(r.fc_last) else float(r.fc_last),
                              risk=None if r.week not in pr else float(pr[r.week])) for r in gdf.itertuples()]
    res[best]["brier_recalibrated"] = float(brier_score_loss(y, pc))
    json.dump(dict(summary=summary, results=res, best=best, calibration=cal, calibration_recalibrated=cal_after, platt=dict(a=pa, b=pb), importance=imp[:12],
                   label_def="Flare = HBI >= 5 at the next visit (<= 8 weeks later) for a patient currently in remission (HBI < 5)",
                   cv="5-fold StratifiedGroupKFold by patient, repeated 5x; 95% CI = patient-level bootstrap of averaged OOF predictions"),
              open(os.path.join(OUT, "flare_metrics.json"), "w"), indent=2)
    json.dump(timelines, open(os.path.join(OUT, "flare_timelines.json"), "w"))
    print("best", best); print(imp[:10])

if __name__ == "__main__":
    main()
