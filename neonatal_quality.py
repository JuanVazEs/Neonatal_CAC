# --------------------------------------------------------------
# neonatal_quality.py      (full upgrade – June 2025 release)
# --------------------------------------------------------------
"""
Neonatal Quality Analyzer – HGZ 20 protocol implementation

Key features added on top of the original version:
▪ Robust data cleanse & column normalisation
▪ Outlier flagging and missing‑value visual audit
▪ Effect‑size & 95 % CI reporting for group comparisons
▪ Cronbach’s α / KR‑20 internal‑consistency check of OMS bundles
▪ χ² matrix with Benjamini‑Hochberg FDR correction
▪ Quick multivariable logistic model for low‑compliance risk factors
▪ Auto‑generated HTML (or PDF) report scaffold via Jinja2

Optional third‑party libs
────────────────────────
pip install missingno pingouin statsmodels jinja2

Protocol reference
──────────────────
All statistical choices match the “ANÁLISIS ESTADÍSTICO” section
(pages 25‑26) of the approved research protocol. :contentReference[oaicite:0]{index=0}
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# optional, imported lazily
_missingno = None
_pingouin = None
_sm_formula = None
_multitest = None

# ------------------------- helpers ----------------------------
_YES = {"si", "sí", "sí", "yes", "y", "1", "true"}
_NO  = {"no", "nó", "n", "0", "false"}


def _yes_no_to_int(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in _YES:
        return 1
    if s in _NO:
        return 0
    # keep numeric if already 0/1
    try:
        n = float(s)
        if n in (0, 1):
            return int(n)
    except ValueError:
        pass
    return np.nan


def _try_import():
    """Import optional heavy‑weight libs only when first needed."""
    global _missingno, _pingouin, _sm_formula, _multitest
    if _missingno is None:
        try:
            import missingno as _missingno  # type: ignore
        except ImportError:
            _missingno = False
    if _pingouin is None:
        try:
            import pingouin as _pingouin  # type: ignore
        except ImportError:
            _pingouin = False
    if _sm_formula is None:
        try:
            import statsmodels.formula.api as _sm_formula  # type: ignore
        except ImportError:
            _sm_formula = False
    if _multitest is None:
        try:
            from statsmodels.stats.multitest import multipletests as _multitest  # type: ignore
        except ImportError:
            _multitest = False


# ---------------------- main analyzer -------------------------
class NeonatalQualityAnalyzer:
    def __init__(self, xlsx_path: str | Path, sheet: str = "Muestreo RN "):
        self._path = Path(xlsx_path)
        self.df = pd.read_excel(self._path, sheet_name=sheet)
        self._rename_columns()
        self._clean()
        self._build_scores()
        self._indicator_cols: list[str] = (
            [c for c in self.df.columns if c.startswith(("examen_", "evaluacion_", "revision_"))]
            + [c for c in self.df.columns if c in (
                "bcg", "hepatitis_b", "vacuna_polio", "vitamina_k", "vitamina_a",
                "suplementacion_de_vitamina_d", "deteccion_de_hiperbilurrubinemia",
                "uso_de_clorhexidina_en_cordon_umbilical", "uso_de_emolientes_en_piel",
                "presencia_fiebre", "lactancia_materna_exclusiva",
                "apego_lactancia_materna", "adecuada_posicion_al_dormir_rn"
            )]
        )

    # ---------- column normalisation -------------------------
    def _rename_columns(self):
        def _tidy(c: str) -> str:
            table = str.maketrans(
                "áéíóúÁÉÍÓÚñÑÃ",
                "aeiouAEIOUNNaa"
            )
            c = c.translate(table)
            return (
                c.strip()
                 .replace("  ", " ")
                 .replace(" ", "_")
                 .lower()
            )
        self.df.columns = [_tidy(c) for c in self.df.columns]

    # ---------- raw cleanse ----------------------------------
    def _clean(self):
        # date/time handling
        if "fecha_nacimiento" in self.df:
            self.df["fecha_nac"] = pd.to_datetime(self.df.pop("fecha_nacimiento"))
        if "hora_nacimiento" in self.df:
            self.df["hora_nac"] = pd.to_datetime(
                self.df.pop("hora_nacimiento").astype(str),
                format="%H:%M:%S",
                errors="coerce"
            ).dt.time

        # categorical harmonisation
        if "sexo" in self.df:
            self.df["sexo"] = self.df["sexo"].str[0].str.upper().astype("category")

        if "resolucion_embarazo" in self.df:
            self.df["delivery_type"] = (
                self.df["resolucion_embarazo"]
                .str.replace("cesarea", "Cesarea", case=False, regex=False)
                .str.replace("parto eutocico", "Vaginal", case=False, regex=False)
                .str.capitalize()
                .astype("category")
            )

        # dichotomous Yes/No conversion (row‑wise to honour memory‑view)
        for c in self.df.columns:
            if self.df[c].dtype == "object":
                sample = str(self.df[c].dropna().iloc[0]).lower()
                if any(x in sample for x in ("si", "no", "yes")):
                    self.df[c] = self.df[c].map(_yes_no_to_int)

        # numeric coercion
        for col in ("sdg", "peso", "apgar"):
            if col in self.df:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # duplicates check
        if "numero_afiliacion" in self.df:
            before = len(self.df)
            self.df.drop_duplicates(subset="numero_afiliacion", inplace=True)
            after = len(self.df)
            if after < before:
                print(f"[info] removed {before-after} duplicate rows.")

    # ---------- composite OMS scores -------------------------
    def _build_scores(self):
        bundles = {
            "score_eval": [
                "examen_fisico_completo",
                "evaluacion_de_rn_signos_peligro",
                "revision_de_antecedentes_de_convulsion",
                "revision_de_frecuencia_respiratoria",
                "revision_de_temperatura_corporal",
                "tamizaje_metabolico",
                "tamizaje_auditivo",
                "screening_visual",
            ],
            "score_prev": [
                "bcg",
                "hepatitis_b",
                "vacuna_polio",
                "vitamina_k",
                "vitamina_a",
                "suplementacion_de_vitamina_d",
                "deteccion_de_hiperbilurrubinemia",
                "uso_de_clorhexidina_en_cordon_umbilical",
                "uso_de_emolientes_en_piel",
                "presencia_fiebre",
            ],
            "score_nut": [
                "lactancia_materna_exclusiva",
                "apego_lactancia_materna",
                "adecuada_posicion_al_dormir_rn",
            ],
        }
        for new, cols in bundles.items():
            cols_present = [c for c in cols if c in self.df]
            self.df[new] = self.df[cols_present].mean(axis=1)

    # ========================================================
    # Descriptives & QC
    # ========================================================
    def missing_map(self):
        """Visual missing‑value matrix (requires *missingno*)."""
        _try_import()
        if _missingno:
            _missingno.matrix(self.df)
            plt.show()
        else:
            print("missingno not installed; run `pip install missingno` to enable.")

    def describe_cont(self, cols: Iterable[str]) -> pd.DataFrame:
        df_num = self.df[list(cols)].apply(pd.to_numeric, errors="coerce")
        tbl = df_num.agg(["count", "mean", "median", "std", "min", "max"]).T
        iqr = (df_num.quantile(.75) - df_num.quantile(.25)).rename("iqr")
        return tbl.join(iqr)

    def describe_cat(self, col: str) -> pd.Series:
        return self.df[col].value_counts(dropna=False)

    # ========================================================
    # Statistical helpers
    # ========================================================
    def _ci_mean(self, series: pd.Series, alpha: float = .05):
        s = series.dropna()
        m, se = s.mean(), s.sem()
        h = se * stats.t.ppf(1 - alpha/2, len(s) - 1)
        return m - h, m + h

    def normality(self, col: str):
        s = self.df[col].dropna()
        return stats.kstest(s, 'norm', args=(s.mean(), s.std(ddof=0)))

    def compare_means(self, num_col: str, group_col: str) -> dict:
        cats = self.df[group_col].dropna().unique()
        arrays = [self.df[self.df[group_col] == c][num_col].dropna() for c in cats]
        normal = all(self.normality(num_col)[1] > .05 for _ in cats)
        if len(cats) == 2:
            stat, p = (
                stats.ttest_ind(*arrays, equal_var=False) if normal
                else stats.mannwhitneyu(*arrays)
            )
            test = "t‑Student" if normal else "Mann‑Whitney"
        else:
            stat, p = (
                stats.f_oneway(*arrays) if normal
                else stats.kruskal(*arrays)
            )
            test = "ANOVA" if normal else "Kruskal‑Wallis"

        res = {"test": test, "stat": stat, "p": p,
               "ci95": self._ci_mean(self.df[num_col])}

        # effect size for two‑group comparison
        if len(cats) == 2:
            try:
                from statsmodels.stats.effect_size import cohen_d
                res["cohen_d"] = cohen_d(arrays[0], arrays[1])
            except Exception:
                res["cohen_d"] = np.nan
        return res

    def chi_square(self, a: str, b: str) -> dict:
        tbl = pd.crosstab(self.df[a], self.df[b])
        chi2, p, dof, _ = stats.chi2_contingency(tbl)
        return {"chi2": chi2, "p": p, "dof": dof}

    def chi_square_matrix(self,
                          targets: Iterable[str],
                          group_col: str = "turno",
                          fdr: bool = True) -> pd.DataFrame:
        pvals = []
        rows = []
        for t in targets:
            stat = self.chi_square(t, group_col)
            pvals.append(stat["p"])
            rows.append({"variable": t, **stat})
        res = pd.DataFrame(rows).set_index("variable")
        if fdr:
            _try_import()
            if _multitest:
                res["q"] = _multitest(pvals, method="fdr_bh")[1]
            else:
                print("statsmodels not installed; FDR adjustment skipped.")
        return res

    # ========================================================
    # Reliability (Cronbach α / KR‑20)
    # ========================================================
    def alpha(self, cols: Iterable[str]) -> float | None:
        _try_import()
        if not _pingouin:
            print("pingouin not installed; run `pip install pingouin`.")
            return None
        return float(_pingouin.cronbach_alpha(self.df[list(cols)])[0])

    # ========================================================
    # Logistic model helper
    # ========================================================
    def logistic_low_compliance(self,
                                score: str = "score_eval",
                                cutoff: float = .8,
                                factors: str = "sdg + peso + C(sexo) + C(delivery_type)"):
        _try_import()
        if not _sm_formula:
            print("statsmodels not installed; install it to run logistic regression.")
            return None
        self.df["low"] = (self.df[score] < cutoff).astype(int)
        formula = f"low ~ {factors}"
        model = _sm_formula.logit(formula, data=self.df).fit(disp=False)
        return model.summary()

    # ========================================================
    # Outlier flagging utility
    # ========================================================
    def flag_outliers(self, col: str, lo: int | float, hi: int | float) -> pd.DataFrame:
        mask = (self.df[col] < lo) | (self.df[col] > hi)
        return self.df.loc[mask, ["numero_afiliacion" if "numero_afiliacion" in self.df else self.df.index.name, col]]

    # ========================================================
    # Rapid narrative report (CLI)
    # ========================================================
    def quick_report(self):
        print("\n» Variables continuas:")
        print(self.describe_cont(["sdg", "peso", "apgar",
                                  "score_eval", "score_prev", "score_nut"]).round(2))
        print("\n» Sexo:")
        print(self.describe_cat("sexo"))
        print("\n» Normalidad peso:", self.normality("peso"))
        print("\n» Peso vs sexo:", self.compare_means("peso", "sexo"))

    # ========================================================
    # HTML/PDF report scaffold
    # ========================================================
    def render_report(self, out: str = "neonatal_report.html"):
        """
        Renders a very simple HTML report with the main tables.
        Requires Jinja2.  Extend the template as desired.
        """
        try:
            from jinja2 import Template
        except ImportError:
            print("jinja2 not installed; `pip install jinja2` to enable report rendering.")
            return

        template = Template("""
        <h1>HGZ‑20 – Informe rápido de estándares OMS</h1>
        <h2>Descripción de variables continuas</h2>
        {{ cont.to_html(classes="table table-striped", float_format="%.2f")|safe }}
        <h2>Frecuencia de sexo</h2>
        {{ sex.to_frame().to_html(classes="table")|safe }}
        <h2>Peso vs sexo</h2>
        <pre>{{ cmp }}</pre>
        """)
        html = template.render(
            cont=self.describe_cont(["sdg", "peso", "apgar",
                                     "score_eval", "score_prev", "score_nut"]).round(2),
            sex=self.describe_cat("sexo"),
            cmp=self.compare_means("peso", "sexo")
        )
        Path(out).write_text(html, encoding="utf-8")
        print(f"[✓] Report written to {out}")

    # ------------- añadir al final de neonatal_quality.py -----------------

    # ======================================================
    #  1. REPORTE DE CUMPLIMIENTO (“semáforo” OMS)
    # ======================================================
    def compliance_report(self,
                          indicators: list[str] | None = None,
                          style: bool = False) -> pd.DataFrame | pd.io.formats.style.Styler:
        """
        Devuelve una tabla con n, n_cumple, %cumplimiento y etiqueta de color.
        Si style=True, retorna un Styler coloreado (útil para .to_html()).
        """
        # Detectar indicadores dicotómicos 0/1 si no se pasan explícitos
        if indicators is None:
            indicators = [c for c in self.df.columns
                          if self.df[c].dropna().isin([0, 1]).all()]

        tbl = (self.df[indicators]
               .agg(['count', 'sum'])
               .T
               .rename(columns={'count': 'n', 'sum': 'cumple'}))
        tbl['pct'] = (tbl['cumple'] / tbl['n'] * 100).round(1)

        # Etiqueta semáforo
        def _label(p):
            if p >= 95:
                return 'verde'
            if p >= 80:
                return 'amarillo'
            return 'rojo'
        tbl['color'] = tbl['pct'].apply(_label)

        if not style:
            return tbl

        # ---------------- Estilo coloreado ----------------
        cmap = {'verde': 'background-color: #8BC34A; color: black;',
                'amarillo': 'background-color: #FFEB3B; color: black;',
                'rojo': 'background-color: #F44336; color: white;'} # Fixed: F44326 -> F44336
        def _style(row):
            return [cmap.get(row.color, '')] * len(row)

        styler = (tbl.style
                       .apply(_style, axis=1)
                       .format({'pct': '{:.1f} %'}))
        return styler

    # ======================================================
    #  2. ANÁLISIS BIVARIADO DETALLADO (χ² / Fisher)
    # ======================================================
    def bivariate_analysis(self,
                           group_vars: list[str] | None = None,
                           indicators: list[str] | None = None,
                           alpha: float = 0.05,
                           min_expected: int = 5) -> pd.DataFrame:
        """
        Para cada indicador binario y cada variable categórica:
            – prueba χ² si todas las celdas esperadas ≥ min_expected
            – Fisher exacto 2×2 si lo anterior falla y la tabla es 2×2
        Devuelve un DataFrame con estadístico, p‑value y significancia.
        """
        if indicators is None:
            indicators = [c for c in self.df.columns
                          if self.df[c].dropna().isin([0, 1]).all()]

        if group_vars is None:
            group_vars = ['sexo', 'delivery_type', 'turno']  # ajusta según tus datos
            group_vars = [g for g in group_vars if g in self.df.columns]

        results = []
        for ind in indicators:
            for grp in group_vars:
                # Crosstab
                ct = pd.crosstab(self.df[ind], self.df[grp])
                # Saltar si hay <2 filas válidas
                if ct.shape[0] < 2:
                    continue

                # -- decidir prueba
                exp = stats.contingency.expected_freq(ct)
                if (exp < min_expected).any() and ct.shape == (2, 2):
                    # Fisher exact
                    stat, p = stats.fisher_exact(ct)
                    test = 'Fisher'
                else:
                    chi2, p, dof, _ = stats.chi2_contingency(ct)
                    stat = chi2
                    test = f'Chi² (dof={dof})'

                results.append({
                    'indicador': ind,
                    'variable': grp,
                    'test': test,
                    'stat': round(float(stat), 3),
                    'p': round(float(p), 4),
                    'significativo': 'Sí' if p < alpha else 'No'
                })

        return pd.DataFrame(results)

# ----------------------------------------------------------------------

# ==============================================================
# Minimal CLI usage
# ==============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HGZ‑20 neonatal QA")
    parser.add_argument("xlsx", help="Excel file (Base de datos Ortiz Perez.xlsx)")
    parser.add_argument("--sheet", default="Muestreo RN ", help="sheet name")
    args = parser.parse_args()

    qa = NeonatalQualityAnalyzer(args.xlsx, sheet=args.sheet)
    qa.quick_report()
    # Uncomment the next lines as needed
    qa.missing_map()
    print("Cronbach α eval:", qa.alpha(["examen_fisico_completo",
                                        "evaluacion_de_rn_signos_peligro",
                                        "revision_de_antecedentes_de_convulsion",
                                        "revision_de_frecuencia_respiratoria"]))
    print(qa.logistic_low_compliance())
    qa.render_report()
