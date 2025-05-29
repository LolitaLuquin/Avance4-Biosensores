
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from scipy.signal import find_peaks
from collections import Counter
from io import StringIO

st.set_page_config(page_title="Análisis Biosensores iMotions", layout="wide")
st.title("🔬 Análisis Integrado de Biosensores - AVANCE3")

# Funciones auxiliares
def make_unique(headers):
    counts = Counter()
    new_headers = []
    for h in headers:
        counts[h] += 1
        if counts[h] > 1:
            new_headers.append(f"{h}_{counts[h]-1}")
        else:
            new_headers.append(h)
    return new_headers

def read_imotions_csv(file, header_index, tipo):
    lines = file.read().decode('utf-8').splitlines()
    headers = make_unique(lines[header_index].strip().split(","))
    data = "\n".join(lines[header_index + 1:])
    df = pd.read_csv(StringIO(data), names=headers)
    df["Participant"] = file.name.replace(".csv", "")
    df["Tipo"] = tipo
    return df

# Carga de archivos
st.sidebar.header("📤 Cargar archivos CSV")
uploaded_et = st.sidebar.file_uploader("Eyetracking", type="csv", accept_multiple_files=True, key="ET")
uploaded_fea = st.sidebar.file_uploader("FEA", type="csv", accept_multiple_files=True, key="FEA")
uploaded_gsr = st.sidebar.file_uploader("GSR", type="csv", accept_multiple_files=True, key="GSR")

df_et = pd.concat([read_imotions_csv(f, 25, "Eyetracking") for f in uploaded_et], ignore_index=True) if uploaded_et else pd.DataFrame()
df_fea = pd.concat([read_imotions_csv(f, 25, "FEA") for f in uploaded_fea], ignore_index=True) if uploaded_fea else pd.DataFrame()
df_gsr = pd.concat([read_imotions_csv(f, 27, "GSR") for f in uploaded_gsr], ignore_index=True) if uploaded_gsr else pd.DataFrame()

# Submenús por biosensor
if not df_et.empty:
    with st.expander("👁️ Eyetracking"):
        df_et = df_et[df_et["EventSource_1"] == 1].copy()
        df_et["ET_TimeSignal"] = pd.to_numeric(df_et["ET_TimeSignal"], errors="coerce")
        df_et = df_et.dropna(subset=["ET_TimeSignal"])

        resultados_et = []
        for (stim, part), g in df_et.groupby(["SourceStimuliName", "Participant"]):
            tiempos = g["ET_TimeSignal"].sort_values()
            resultados_et.append({
                "Estímulo": stim, "Participante": part,
                "Tiempo_Permanencia": tiempos.max() - tiempos.min(),
                "TTFF": tiempos.min(), "N_Fijaciones": len(tiempos)
            })
        df_et_resumen = pd.DataFrame(resultados_et)
        st.dataframe(df_et_resumen)

        # Gráficos
        for var in ["Tiempo_Permanencia", "TTFF", "N_Fijaciones"]:
            fig, ax = plt.subplots()
            sns.barplot(data=df_et_resumen, x="Estímulo", y=var, errorbar="sd", ax=ax)
            ax.set_title(f"{var} por Estímulo")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # ANOVA y F²
        for var in ["Tiempo_Permanencia", "TTFF", "N_Fijaciones"]:
            grupos = [g[var].values for _, g in df_et_resumen.groupby("Estímulo")]
            anova = f_oneway(*grupos)
            mean = df_et_resumen[var].mean()
            ss_total = ((df_et_resumen[var] - mean) ** 2).sum()
            ss_between = sum([len(g)*(g[var].mean() - mean)**2 for _, g in df_et_resumen.groupby("Estímulo")])
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
            st.write(f"**{var}** → F: {anova.statistic:.2f}, p: {anova.pvalue:.4f}, η²: {eta_sq:.4f}")

if not df_gsr.empty:
    with st.expander("🫀 GSR"):
        df_gsr["GSR Conductance CAL"] = pd.to_numeric(df_gsr["GSR Conductance CAL"], errors="coerce")
        gsr_summary = []
        for stim, grupo in df_gsr.groupby("SourceStimuliName"):
            signal = grupo["GSR Conductance CAL"].dropna().values
            peaks, props = find_peaks(signal, height=0.02)
            amplitudes = props["peak_heights"]
            gsr_summary.append({
                "Estímulo": stim,
                "Núm_Picos": len(peaks),
                "Amp_Media": np.mean(amplitudes) if len(amplitudes) > 0 else 0,
                "Amp_SD": np.std(amplitudes) if len(amplitudes) > 0 else 0
            })
        df_gsr_summary = pd.DataFrame(gsr_summary)
        st.dataframe(df_gsr_summary)

        fig, ax = plt.subplots()
        sns.barplot(data=df_gsr_summary, x="Estímulo", y="Amp_Media", ax=ax)
        ax.set_title("Amplitud Media de Picos GSR por Estímulo")
        plt.xticks(rotation=45)
        st.pyplot(fig)

if not df_fea.empty:
    with st.expander("😀 FEA"):
        emociones = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
        df_fea["Engagement_Promedio"] = df_fea[emociones].mean(axis=1)
        df_fea["Valence_Class"] = df_fea["Valence"].apply(lambda x: "Positiva" if x > 0 else ("Negativa" if x < 0 else "Neutra"))

        tabla_fea = df_fea.groupby("SourceStimuliName").agg({
            "Valence": ["mean", "std"],
            "Engagement_Promedio": ["mean", "std"]
        }).reset_index()
        tabla_fea.columns = ["Estímulo", "Valencia_Media", "Valencia_SD", "Engagement_Media", "Engagement_SD"]
        st.dataframe(tabla_fea)

        for var in ["Valence", "Engagement_Promedio"]:
            fig, ax = plt.subplots()
            sns.boxplot(data=df_fea, x="SourceStimuliName", y=var, ax=ax)
            ax.set_title(f"{var} por Estímulo")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        grupos_val = [g["Valence"].dropna() for _, g in df_fea.groupby("SourceStimuliName") if len(g) > 1]
        grupos_eng = [g["Engagement_Promedio"].dropna() for _, g in df_fea.groupby("SourceStimuliName") if len(g) > 1]
        if len(grupos_val) > 1:
            a1 = f_oneway(*grupos_val)
            st.write(f"**Valencia** → F: {a1.statistic:.2f}, p: {a1.pvalue:.4f}")
        if len(grupos_eng) > 1:
            a2 = f_oneway(*grupos_eng)
            st.write(f"**Engagement** → F: {a2.statistic:.2f}, p: {a2.pvalue:.4f}")
