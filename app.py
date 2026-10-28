"""A/B Testing Dashboard - Internship project (internee.pk).

Demo dataset: public landing-page A/B test (NOT real internee.pk analytics).
Run with:  streamlit run app.py        (ab_data.csv must be in the same folder)

Sections of this file
  1. Imports and constants
  2. Style sheet (spacing, cards, tabs, banner, animations)
  3. Helper functions (cards, banner, charts, statistics)
  4. Sidebar controls and data loading
  5. Header
  6. Tab 1 Overview | Tab 2 Data quality | Tab 3 Over time | Tab 4 Power & impact
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binomtest, chisquare
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

st.set_page_config(page_title="A/B Testing Dashboard", page_icon="📊", layout="wide")

# ============================================================== 1. CONSTANTS
G = ["control", "treatment"]
COLORS = {"control": "#1F4E96", "treatment": "#F47B00"}          # same colours in every chart
REQUIRED = {"user_id", "timestamp", "group", "landing_page", "converted"}
ALPHAS = [0.01, 0.05, 0.10]
CARD_H = 160                                                       # every KPI card has this height (equal heights)
# tone name -> (soft background, strong colour)
TONES = {"blue": ("#E3EEFF", "#1F4E96"), "orange": ("#FFEBD6", "#D96A00"), "green": ("#DDF5E4", "#16803A"),
         "red": ("#FFE3E0", "#C0281E"), "gray": ("#EBEEF2", "#3F4954"), "purple": ("#EEE6FF", "#5B34A6")}
# tone name -> (gradient start, gradient end) for the big recommendation banner
BANNER = {"orange": ("#C95F00", "#F58A0B"), "green": ("#12692F", "#2FA85A"), "red": ("#A81F17", "#E0483D")}

# ============================================================== 2. STYLE SHEET
# flash0/1/2 are identical keyframes with different names: every alpha value uses its own animation name,
# so moving the slider changes the name and the highlight replays.
FLASH = "".join(
    "@keyframes flash%d {0%% {box-shadow:0 0 0 0 rgba(244,123,0,.85); transform:scale(1.03);} "
    "60%% {box-shadow:0 0 0 14px rgba(244,123,0,0); transform:scale(1.0);} 100%% {box-shadow:0 2px 8px rgba(0,0,0,.08);}}\n" % i
    for i in range(3))

CSS = """
<style>
/* ---- spacing: Streamlit adds a hidden -17px margin to text blocks; reset it so cards never overlap ---- */
[data-testid="stMarkdownContainer"] {margin-bottom: 0 !important;}
.block-container {padding: 3rem 1.6rem 0.8rem 1.6rem !important; max-width: 100% !important;}   /* full width */
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stVerticalBlock"] {gap: 0.5rem !important;}                                         /* tight vertical gaps */
[data-testid="stHorizontalBlock"] {gap: 0.6rem !important; align-items: stretch !important;}      /* equal-height columns */
html, body, [class*="css"] {font-size: 18px;}
h3 {font-size: clamp(1.4rem, 1.2vw + .5rem, 1.7rem) !important; font-weight: 800 !important; margin: 0 0 .05rem 0 !important; padding: 0 !important;}
p {margin-bottom: .3rem;}
[data-testid="stCaptionContainer"] {margin-top: -2px;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {font-size: 17px !important; color: #3F4954 !important;}

/* ---- page + sidebar backgrounds ---- */
[data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #F4F8FF 0%, #FFFFFF 380px);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #EAF1FF 0%, #F7F3FF 100%); border-right: 1px solid #D5E0F5;}
[data-testid="stSidebar"] h2 {font-weight: 900;}

/* ---- dark blue hero header ---- */
.hero {display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px;
       background: linear-gradient(120deg, #0F2A5C 0%, #1F4E96 55%, #3C7BE0 100%); color:#fff;
       border-radius: 16px; padding: 12px 24px; box-shadow: 0 8px 24px rgba(15,42,92,.35);}
.hero-title {font-size: 36px; font-weight: 900; line-height: 1.1;}
.hero-sub {font-size: 18px; opacity: .92; margin-top: 2px;}
.chip {border-radius: 999px; padding: 6px 16px; font-weight: 800; font-size: 16.5px; box-shadow: 0 1px 5px rgba(0,0,0,.12);
       transition: transform .2s ease;}
.chip:hover {transform: scale(1.06);}
.hero .chip {background: rgba(255,255,255,.18); color:#fff; border:1px solid rgba(255,255,255,.35); box-shadow:none;}

/* ---- section cards: rounded, soft shadow, compact padding (containers created with key="sec_...") ---- */
[class*="st-key-sec_"] {background: #FFFFFF; border: 1px solid #DCE6F7; border-radius: 16px;
  padding: 11px 14px 10px 14px !important; box-shadow: 0 2px 10px rgba(31,78,150,.08);
  transition: box-shadow .25s ease, transform .25s ease, border-color .25s ease;}
[class*="st-key-sec_"]:hover {box-shadow: 0 8px 22px rgba(31,78,150,.16); transform: translateY(-2px); border-color: #A9C0EA;}
.st-key-sec_conv    {background: linear-gradient(135deg, #EAF2FF 0%, #FFFFFF 70%);}
.st-key-sec_bounce  {background: linear-gradient(135deg, #F1EAFF 0%, #FFFFFF 70%);}
.st-key-sec_gauge   {background: linear-gradient(135deg, #E6F0FF 0%, #FFFFFF 75%); height: 100%;}
.st-key-sec_meaning {background: linear-gradient(135deg, #E4F7EA 0%, #FFFFFF 70%); border-color:#9ED8B1;}
.st-key-sec_power   {background: linear-gradient(135deg, #FFF3E4 0%, #FFFFFF 70%);}
.st-key-sec_impact  {background: linear-gradient(135deg, #FFEDEA 0%, #FFFFFF 70%);}
.st-key-sec_insight {background: linear-gradient(135deg, #E4F7EA 0%, #FFFFFF 70%); border-color:#9ED8B1;}
.st-key-sec_dq1     {background: linear-gradient(135deg, #EAF2FF 0%, #FFFFFF 70%);}
.st-key-sec_dq2     {background: linear-gradient(135deg, #F1EAFF 0%, #FFFFFF 70%);}
[data-testid="stColumn"] > div, [data-testid="stColumn"] > div > [data-testid="stVerticalBlock"] {height: 100%;}

/* ---- KPI cards: all the same height, hover lift ---- */
.kpi {border-radius: 14px; padding: 11px 15px; height: __H__px; box-sizing: border-box; overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,.08); transition: transform .2s ease, box-shadow .2s ease; cursor: pointer; container-type: inline-size;}
.kpi .lab {font-size: 16px; font-weight: 800; color: #26303A; letter-spacing: .2px; text-transform: uppercase; line-height: 1.2;}
.kpi .val {font-weight: 900; line-height: 1.12; margin: 3px 0 2px 0; white-space: nowrap;}
.kpi .sub {font-size: 17px; font-weight: 500; color: #3F4954; line-height: 1.25;}
.kpi:hover {transform: translateY(-4px) scale(1.015); box-shadow: 0 10px 24px rgba(0,0,0,.18);}

/* ---- recommendation banner: solid colour, white bold text ---- */
.banner {border-radius: 16px; padding: 16px 24px; color: #fff; box-shadow: 0 8px 22px rgba(0,0,0,.25);
         transition: transform .2s ease;}
.banner .t {font-size: 31px; font-weight: 900; line-height: 1.2; letter-spacing: .1px;}
.banner .b {font-size: 18.5px; font-weight: 600; margin-top: 5px; opacity: .97;}
.banner:hover {transform: translateY(-2px);}

/* ---- insight boxes (Over time tab) ---- */
.insight {background:#fff; border:1px solid #BFE3CA; border-left:7px solid #16803A; border-radius:12px; padding:11px 15px;
          min-height: 170px; box-shadow: 0 2px 8px rgba(0,0,0,.06); transition: transform .2s ease;}
.insight:hover {transform: translateY(-3px);}
.insight .it {font-size: 23px; font-weight: 900; color:#16803A; margin-bottom:3px;}
.insight .ib {font-size: 19px; color:#16202B; line-height:1.45;}

/* ---- tabs: pill buttons, ACTIVE tab = strong dark gradient, bigger bold white text ---- */
div[role="tablist"], div[data-baseweb="tab-list"] {gap: 8px; background: #DCE8FB; padding: 6px; border-radius: 14px;
      border-bottom: none !important; box-shadow: inset 0 1px 3px rgba(31,78,150,.15);}
[data-testid="stTab"], button[data-baseweb="tab"] {border-radius: 10px !important; padding: 9px 24px !important;
      background: transparent; transition: all .2s ease; cursor: pointer; border-bottom: none !important;}
[data-testid="stTab"] p, button[data-baseweb="tab"] p {font-size: 1.25rem !important; font-weight: 700 !important; margin: 0; color:#33425B;}
[data-testid="stTab"]:hover, button[data-baseweb="tab"]:hover {background: #C4D8F8 !important; transform: translateY(-1px);}
[data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] {
      background: linear-gradient(135deg, #0F2A5C 0%, #2F6FDB 100%) !important;
      box-shadow: 0 6px 16px rgba(15,42,92,.45); transform: translateY(-2px) scale(1.04);}
[data-testid="stTab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] p {
      color: #FFFFFF !important; font-weight: 900 !important; font-size: 1.3rem !important;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display: none !important;}
div[role="tablist"]::after, div[role="tablist"]::before {display: none !important;}
.react-aria-SelectionIndicator, [class*="SelectionIndicator"] {display: none !important;}
[data-testid="stTabPanel"], [role="tabpanel"] {animation: tabIn .35s ease;}                       /* fade/slide when switching */
@keyframes tabIn {from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: translateY(0);}}

/* ---- radio buttons as pills, sliders and inputs with hover feedback ---- */
[data-testid="stRadioGroup"], div[role="radiogroup"] {gap: 8px !important;}
[data-testid="stRadioOption"], div[role="radiogroup"] > label {background: #FFFFFF; border: 2px solid #C9D8F2;
      border-radius: 999px; padding: 6px 18px; transition: all .2s ease; cursor: pointer;}
[data-testid="stRadioOption"]:hover {border-color: #1F4E96; background: #EAF2FF; transform: translateY(-1px);}
[data-testid="stRadioOption"][data-selected="true"] {background: #1F4E96; border-color: #1F4E96; box-shadow: 0 3px 10px rgba(31,78,150,.35);}
[data-testid="stRadioOption"][data-selected="true"] * {color: #FFFFFF !important; font-weight: 800;}
[data-testid="stSlider"] {padding: 4px 6px; border-radius: 12px; transition: background .2s ease;}
[data-testid="stSlider"]:hover {background: rgba(31,78,150,.07);}
[data-testid="stNumberInput"] input {font-weight: 800; font-size: 1.05rem;}


/* ---- READABILITY: final decision sentence (prominent, dark text on a light tinted background) ---- */
.final {border-radius: 14px; padding: 13px 22px; box-shadow: 0 4px 14px rgba(0,0,0,.12); transition: transform .2s ease;}
.final:hover {transform: translateY(-2px);}
.final .ft {font-size: 29px; font-weight: 900; line-height: 1.3; color: #101A24;}
.final .ft span {text-transform: uppercase; letter-spacing: .4px;}

/* ---- READABILITY: larger text in the "What this means for Internee.pk" section and in sidebar help text ---- */
.st-key-sec_meaning p, .st-key-sec_meaning li {font-size: 21px !important; line-height: 1.5 !important; color: #16202B !important;}
.st-key-sec_meaning strong {font-weight: 900;}
/* sidebar (left panel): all text larger and darker */
[data-testid="stSidebar"] h2 {font-size: 2rem !important;}
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {font-size: 19px !important; font-weight: 700 !important; color: #16202B !important; line-height: 1.3 !important;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {font-size: 18px !important; font-weight: 600 !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {font-size: 17px !important; line-height: 1.45 !important; color: #2E3A47 !important;}
[data-testid="stSidebar"] input {font-size: 19px !important; font-weight: 800 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span, [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small, [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {font-size: 16px !important;}
[data-testid="stSidebar"] [data-testid="stSlider"] * {font-size: 17px;}
/* wider sidebar so the larger text fits (only when the sidebar is open; collapsing still works) */
[data-testid="stSidebar"][data-collapsed="false"] {width: clamp(310px, 20vw, 380px) !important; min-width: clamp(310px, 20vw, 380px) !important;}

/* ---- FINAL POLISH: even tighter spacing ---- */
[data-testid="stVerticalBlock"] {gap: 0.42rem !important;}
[data-testid="stHorizontalBlock"] {gap: 0.5rem !important;}
[class*="st-key-sec_"] {padding: 9px 13px 8px 13px !important;}

/* ---- FINAL POLISH: paired cards in the same row stretch to equal height ---- */
.st-key-sec_dq_water, .st-key-sec_dq2, .st-key-sec_dq_traffic, .st-key-sec_dq_dict, .st-key-sec_plan, .st-key-sec_curve,
.st-key-sec_impact, .st-key-sec_range, .st-key-sec_chart_a, .st-key-sec_chart_b, .st-key-sec_time1, .st-key-sec_time2 {height: 100%;}

/* the wrapper element between a column and a paired card must also be full height */
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_gauge),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_dq_water),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_dq2),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_dq_traffic),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_dq_dict),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_plan),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_curve),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_impact),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_range),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_chart_a),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_chart_b),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_time1),
[data-testid="stLayoutWrapper"]:has(> .st-key-sec_time2)
  {height: 100%;}

/* the gauge sits in the middle of its (stretched) card now that the verdict box is gone */
.st-key-sec_gauge [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {margin: auto 0;}

/* ---- FINAL POLISH: charts grow with the window height so tall/zoomed-out screens are filled (never smaller than min) ---- */
.st-key-sec_chart_a [data-testid="stPlotlyChart"], .st-key-sec_chart_b [data-testid="stPlotlyChart"]
  {height: clamp(370px, calc(100vh - 880px), 640px) !important;}
.st-key-sec_time1 [data-testid="stPlotlyChart"], .st-key-sec_time2 [data-testid="stPlotlyChart"]
  {height: clamp(400px, calc(100vh - 650px), 860px) !important;}
.st-key-sec_dq_water [data-testid="stPlotlyChart"] {height: clamp(392px, 36vh, 640px) !important;}
.st-key-sec_dq_traffic [data-testid="stPlotlyChart"] {height: clamp(330px, 31vh, 560px) !important;}
.st-key-sec_curve [data-testid="stPlotlyChart"] {height: clamp(330px, 36vh, 450px) !important;}
.st-key-sec_range [data-testid="stPlotlyChart"] {height: clamp(205px, 22vh, 270px) !important;}
/* Streamlit fixes the chart wrapper at 450px; let it follow the chart height instead so cards grow with their charts */
[class*="st-key-sec_"] [data-testid="stFullScreenFrame"], [class*="st-key-sec_"] [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"])
  {height: auto !important; flex: 0 0 auto !important; overflow: visible !important;}
[class*="st-key-sec_chart"] [data-testid="stPlotlyChart"] > div, [class*="st-key-sec_time"] [data-testid="stPlotlyChart"] > div,
[class*="st-key-sec_dq_"] [data-testid="stPlotlyChart"] > div, .st-key-sec_curve [data-testid="stPlotlyChart"] > div,
.st-key-sec_range [data-testid="stPlotlyChart"] > div, [class*="st-key-sec_chart"] .js-plotly-plot, [class*="st-key-sec_time"] .js-plotly-plot,
[class*="st-key-sec_dq_"] .js-plotly-plot, .st-key-sec_curve .js-plotly-plot, .st-key-sec_range .js-plotly-plot
  {height: 100% !important;}

/* ---- FINAL POLISH: ACTIVE TAB much more obvious: darker gradient, glow ring, bold white text, accent bar ---- */
[data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] {
      background: linear-gradient(135deg, #0A1F4A 0%, #1F5FD0 100%) !important;
      box-shadow: 0 0 0 3px rgba(60,123,224,.35), 0 8px 18px rgba(10,31,74,.5) !important;
      transform: translateY(-2px) scale(1.06); position: relative;}
[data-testid="stTab"][aria-selected="true"]::after {content: ""; position: absolute; left: 22%; right: 22%; bottom: 3px;
      height: 3px; border-radius: 3px; background: #FFC857;}
[data-testid="stTab"] p, button[data-baseweb="tab"] p {color: #4A5A75;}

/* ---- FINAL POLISH: stronger, more eye-catching banner ---- */
.banner {border: 3px solid rgba(255,255,255,.55); box-shadow: 0 10px 28px rgba(0,0,0,.30), 0 0 0 4px rgba(244,123,0,.18); padding: 15px 26px;}
.banner .tag {display: inline-block; background: rgba(255,255,255,.22); border: 1px solid rgba(255,255,255,.6); border-radius: 999px;
      padding: 2px 12px; font-size: 15px; font-weight: 900; letter-spacing: 1.4px; margin-bottom: 5px;}
.banner .t {font-size: 38px; text-shadow: 0 2px 4px rgba(0,0,0,.25);}
.banner .b {font-size: 22px; font-weight: 700;}
</style>
""".replace("__H__", str(CARD_H))
st.markdown(CSS + "<style>" + FLASH + "</style>", unsafe_allow_html=True)


# ============================================================== 3. HELPERS
def card(label, value, sub="", tone="blue", flash=None, size=46, height=True):
    """KPI card. Every card has the same fixed height so rows look even. flash = animation index (alpha change)."""
    bg, fg = TONES[tone]
    anim = f"animation: flash{flash} .9s ease;" if flash is not None else ""
    h = "" if height else "height:auto;"
    return (f'<div class="kpi" style="background:linear-gradient(135deg,{bg},#fff 140%);border-left:7px solid {fg};{anim}{h}">'
            f'<div class="lab">{label}</div><div class="val" style="color:{fg};font-size:{int(size*0.7)}px;font-size:clamp({int(size*0.56)}px,23cqw,{size}px);">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def banner(title, tone, flash=None):
    """Large recommendation banner: solid colour, white bold title only. Flashes when alpha changes."""
    c1, c2 = BANNER[tone]
    anim = f"animation: flash{flash} .9s ease;" if flash is not None else ""
    st.markdown(f'<div class="banner" style="background:linear-gradient(120deg,{c1},{c2});{anim}">'
                f'<div class="tag">RECOMMENDATION</div><div class="t">{title}</div></div>', unsafe_allow_html=True)


def style_fig(fig, height=370):
    """Shared chart style: larger fonts, transparent background, compact margins."""
    fig.update_layout(template="plotly_white", height=height, autosize=True, font=dict(size=17), title_font=dict(size=22),
                      margin=dict(t=55, b=35, l=40, r=15), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(font=dict(size=17)), hoverlabel=dict(font_size=16))
    return fig


def show(fig):
    """Render a Plotly chart without the toolbar."""
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def top_legend(fig):
    """Horizontal legend above the plot so it never sits on the title."""
    return fig.update_layout(margin=dict(t=100, b=40, l=40, r=15),
                             legend=dict(orientation="h", y=1.03, yanchor="bottom", x=0, font=dict(size=17)))


@st.cache_data
def load(file):
    return pd.read_csv(file)


def clean(df):
    """Remove group/page mismatches, then duplicate users. Returns clean data + summary table."""
    ok = ((df["group"] == "control") & (df["landing_page"] == "old_page")) | \
         ((df["group"] == "treatment") & (df["landing_page"] == "new_page"))
    step1 = df[ok]
    out = step1.drop_duplicates("user_id").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["date"] = out["timestamp"].dt.date
    table = pd.DataFrame({
        "Step": ["Raw rows", "Mismatched group/page rows removed", "Duplicate users removed", "Final clean rows"],
        "Rows": [len(df), int((~ok).sum()), len(step1) - len(out), len(out)]})
    return out, table


def ab_test(d, metric):
    """Two-proportion z-test + 95% CI of the difference (treatment - control)."""
    n = d.groupby("group")[metric].count().reindex(G)
    x = d.groupby("group")[metric].sum().reindex(G)
    p = x / n
    z, pval = proportions_ztest([x["treatment"], x["control"]], [n["treatment"], n["control"]])
    se_each = np.sqrt(p * (1 - p) / n)
    se = np.sqrt((se_each ** 2).sum())
    diff = p["treatment"] - p["control"]
    return dict(n=n, x=x, p=p, pval=pval, diff=diff, lo=diff - 1.96 * se, hi=diff + 1.96 * se, se_each=se_each)


def metric_cards(r, alpha, flash, lower_better=False, tag=""):
    """Row of 4 equal cards: control, treatment, difference (+95% CI), p-value."""
    sig = r["pval"] < alpha
    good = (r["diff"] < 0) if lower_better else (r["diff"] > 0)
    diff_tone = ("green" if good else "red") if sig else "orange"
    c = st.columns(4)
    c[0].markdown(card("Control" + tag, f"{r['p']['control']*100:.2f}%", f"{int(r['n']['control']):,} users", "blue"), unsafe_allow_html=True)
    c[1].markdown(card("Treatment" + tag, f"{r['p']['treatment']*100:.2f}%", f"{int(r['n']['treatment']):,} users", "orange"), unsafe_allow_html=True)
    c[2].markdown(card("Difference", f"{r['diff']*100:+.2f} pp", f"95% CI [{r['lo']*100:+.2f}, {r['hi']*100:+.2f}] pp", diff_tone, flash), unsafe_allow_html=True)
    c[3].markdown(card("p-value", f"{r['pval']:.4f}", "", "green" if sig else "gray", flash), unsafe_allow_html=True)


def placeholder_cards(why):
    """Same 4-card layout as the real metrics, shown when a metric cannot be calculated."""
    c = st.columns(4)
    for col, lab in zip(c, ["Control", "Treatment", "Difference", "p-value"]):
        col.markdown(card(lab, "N/A", why, "gray"), unsafe_allow_html=True)


def p_gauge(pval, alpha):
    """Gauge of the p-value against alpha: the red line moves when the slider moves."""
    axis_max = 0.5 if pval <= 0.5 else 1.0
    sig = pval < alpha
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pval, number=dict(valueformat=".4f", font=dict(size=34, color="#16803A" if sig else "#3F4954")),
        title=dict(text=f"p-value vs alpha ({alpha})", font=dict(size=20)),
        gauge=dict(axis=dict(range=[0, axis_max], tickfont=dict(size=15)),
                   bar=dict(color="#16803A" if sig else "#F47B00", thickness=0.35),
                   steps=[dict(range=[0, alpha], color="#BDEBCB"), dict(range=[alpha, axis_max], color="#E6EAF0")],
                   threshold=dict(line=dict(color="#C0281E", width=5), thickness=0.9, value=alpha))))
    # bottom margin leaves room for the 0 / 0.5 tick labels so the gauge is never clipped
    fig.update_layout(height=290, margin=dict(t=55, b=34, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font=dict(size=16))
    return fig


def settle_point(d, tol_pp=0.1):
    """Users per group after which each group's cumulative rate stays within tol_pp of its final value."""
    pts = []
    for g in G:
        s = d.loc[d["group"] == g, "converted"].to_numpy()
        cum = np.cumsum(s) / np.arange(1, len(s) + 1) * 100
        bad = np.where(np.abs(cum - cum[-1]) > tol_pp)[0]
        pts.append(int(bad[-1]) + 2 if len(bad) else 1)
    return max(pts)


# ============================================================== 4. SIDEBAR + DATA
st.sidebar.header("⚙️ Control panel")
upload = st.sidebar.file_uploader("Upload your own A/B CSV (optional)", type="csv")
alpha = st.sidebar.select_slider("Significance level (alpha)", ALPHAS, value=0.05,
                                 help="Results update live. A result is significant when p < alpha.")
visitors = st.sidebar.number_input("Visitors for impact estimate (assumption)", 1000, 10_000_000, 100_000, 1000)
demo_bounce = st.sidebar.checkbox("Demo only: add SIMULATED bounce data", value=False,
                                  help="Used only if your file has no 'bounced' column. Random data, identical for both groups. "
                                       "For layout demonstration only; never report it as a real result.")
st.sidebar.caption("Required columns: user_id, timestamp, group (control/treatment), landing_page "
                   "(old_page/new_page), converted (0/1). Optional: bounced (0/1).")

raw = load(upload if upload else "ab_data.csv")
missing = REQUIRED - set(raw.columns)
if missing:
    st.error(f"Your file is missing columns: {', '.join(sorted(missing))}")
    st.stop()
df, qtable = clean(raw)

bounce_real = "bounced" in df.columns
bounce_sim = False
if not bounce_real and demo_bounce:                       # simulated bounce: same probability in both groups
    df["bounced"] = (np.random.default_rng(7).random(len(df)) < 0.40).astype(int)
    bounce_sim = True
has_bounce = bounce_real or bounce_sim

res = ab_test(df, "converted")
sig = res["pval"] < alpha
better = sig and res["diff"] > 0
worse = sig and res["diff"] < 0
pp = res["diff"] * 100
flash = ALPHAS.index(alpha)                               # picks a different animation name per alpha value

if st.session_state.get("prev_alpha", alpha) != alpha:    # pop-up when the slider changes
    st.toast(f"alpha set to {alpha}: the result is now {'SIGNIFICANT' if sig else 'NOT significant'} "
             f"(p = {res['pval']:.4f})", icon="✅" if sig else "⚠️")
st.session_state["prev_alpha"] = alpha

# ============================================================== 5. HEADER
st.markdown(
    '<div class="hero"><div><div class="hero-title">📊 A/B Testing Dashboard</div>'
    '<div class="hero-sub">Does the new page convert better than the current page?</div></div>'
    '<div style="display:flex;flex-wrap:wrap;gap:8px;">'
    '<span class="chip">Internship project: internee.pk</span>'
    f'<span class="chip">alpha = {alpha}</span></div></div>',
    unsafe_allow_html=True)
st.caption("Demo dataset: public landing-page A/B test, **not** real internee.pk analytics. "
           "Upload your own file in the sidebar to analyze other data.")

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "🧹 Data quality", "📈 Over time", "⚡ Power & impact"])

# ============================================================== TAB 1: OVERVIEW
with tab1:
    top_l, top_r = st.columns([2.9, 1.25])
    with top_l:
        with st.container(key="sec_conv"):
            st.subheader("📌 Conversion rate (primary metric)")
            metric_cards(res, alpha, flash)

        with st.container(key="sec_bounce"):
            st.subheader("🚪 Bounce rate (secondary metric, lower is better)")
            if has_bounce:
                br = ab_test(df, "bounced")
                metric_cards(br, alpha, flash, lower_better=True, tag=" (simulated)" if bounce_sim else "")
                if bounce_sim:
                    st.caption("⚠️ **SIMULATED DATA:** random demo values (same probability for both groups). Not a real result.")
                else:
                    st.caption("Bounce rate calculated from the `bounced` column of your file.")
                if not bounce_sim and br["pval"] < alpha:
                    st.warning("Bounce rate differs significantly between the groups. Review it before deciding.")
            else:
                # clean placeholder cards in the same style as the conversion cards
                placeholder_cards("Not available in this dataset")
                st.caption("The public dataset has no `bounced` column. Upload a CSV with `bounced` (0/1), e.g. exported from "
                           "Google Analytics 4, or tick *Demo only: add SIMULATED bounce data* in the sidebar to preview.")

    with top_r, st.container(key="sec_gauge"):
        st.subheader("🎚️ Significance meter")
        show(p_gauge(res["pval"], alpha))

    # ---- FINAL DECISION: one clear bold sentence directly under the banner (colour follows the result)
    if better:
        final_txt, final_tone = "Roll out the new design.", "green"
    elif worse:
        final_txt, final_tone = "Keep the current homepage.", "red"
    else:
        final_txt, final_tone = "Keep the current homepage.", "orange"
    f_bg, f_fg = TONES[final_tone]
    st.markdown(f'<div class="final" style="background:{f_bg};border:2px solid {f_fg};border-left:12px solid {f_fg};">'
                f'<div class="ft"><span style="color:{f_fg};">Final decision:</span> {final_txt}</div></div>', unsafe_allow_html=True)

    # ---- two charts
    left, right = st.columns([1, 1.25])
    with left, st.container(key="sec_chart_a"):
        fig = go.Figure(go.Bar(x=["Control (current)", "Treatment (new)"], y=res["p"].values * 100,
                               error_y=dict(type="data", array=1.96 * res["se_each"].values * 100,
                                            thickness=3, width=10, color="#1B232C"),
                               marker_color=[COLORS[g] for g in G],
                               text=[f"<b>{v*100:.2f}%</b>" for v in res["p"]], textposition="inside",
                               textfont=dict(size=22, color="white"), width=0.55))
        fig.update_layout(title="Conversion rate (error bars = 95% CI)", yaxis_title="Conversion rate (%)")
        fig.update_yaxes(range=[0, res["p"].max() * 100 * 1.3])
        show(style_fig(fig, None))
    with right, st.container(key="sec_chart_b"):
        fig = go.Figure(go.Scatter(x=[pp], y=[0], mode="markers", marker=dict(size=16, color="#1B232C"),
                                   error_x=dict(type="data", symmetric=False, thickness=3, width=12,
                                                array=[(res["hi"] - res["diff"]) * 100],
                                                arrayminus=[(res["diff"] - res["lo"]) * 100], color="#1B232C")))
        fig.add_vline(x=0, line_dash="dash", line_color="#C0281E", line_width=3,
                      annotation_text="No difference", annotation_font_size=15)
        fig.add_annotation(x=pp, y=0, text=f"<b>{pp:+.2f} pp</b>", yshift=28, showarrow=False, font=dict(size=18))
        fig.update_layout(title=f"Difference with 95% CI: [{res['lo']*100:+.2f}, {res['hi']*100:+.2f}] pp",
                          xaxis_title="Treatment minus control (percentage points)", yaxis=dict(visible=False))
        show(style_fig(fig, None))

    # ---- what this means for internee.pk
    with st.container(key="sec_meaning"):
        st.subheader("💡 What this means for Internee.pk")
        m1, m2 = st.columns(2)
        if better:
            msg = ("**The new page is better.** It improved sign-ups and the difference is statistically significant. "
                   "**Recommendation: roll out the new design**, but first confirm that bounce rate and page speed did not get worse.")
        elif worse:
            msg = ("**The new page is not better. It is significantly worse.** It reduced sign-ups in this data. "
                   "**Recommendation: keep the current homepage** and review what made the new design less effective before trying again.")
        else:
            msg = (f"**The new page is not better than the current homepage.** It converted {abs(pp):.2f} pp "
                   f"{'lower' if pp < 0 else 'higher'} and the difference is not statistically significant, so there is no reason to switch.\n\n"
                   "**Recommendation: keep the current homepage.**\n\n"
                   "**Next test:** try a bolder change, for example a clearer main call-to-action or a very different layout, "
                   "instead of small tweaks such as going from two buttons to one. Small changes are hard to detect, so a bolder change gives a clearer answer.")
        m1.markdown(f"**Decision for the homepage**\n\n{msg}")
        m2.markdown("**Engagement and caution**\n\n"
                    "- Track bounce rate, button clicks and time on page in Google Analytics 4, because conversion alone "
                    "does not show how visitors engage with the homepage.\n"
                    "- These results come from a public dataset, so they illustrate the method and are not internee.pk's "
                    "real performance. Confirm with a live test on internee.pk traffic.")

# ============================================================== TAB 2: DATA QUALITY
with tab2:
    gn = df["group"].value_counts().reindex(G)
    srm_p = chisquare(gn).pvalue

    with st.container(key="sec_dq1"):
        st.subheader("🧹 Cleaning summary")
        c = st.columns(4)
        for col, (_, row), tone in zip(c, qtable.iterrows(), ["gray", "orange", "orange", "green"]):
            col.markdown(card(row["Step"], f"{int(row['Rows']):,}", "", tone), unsafe_allow_html=True)
        st.caption("Almost all duplicate IDs were inside the mismatched rows, so removing mismatches first also removed them.")

    a1, a2 = st.columns([1.15, 1])
    with a1, st.container(key="sec_dq_water"):
        steps = qtable["Rows"].tolist()
        fig = go.Figure(go.Waterfall(x=["Raw rows", "Mismatched removed", "Duplicates removed", "Clean rows"],
                                     measure=["absolute", "relative", "relative", "total"],
                                     y=[steps[0], -steps[1], -steps[2], 0],
                                     text=[f"{steps[0]:,}", f"-{steps[1]:,}", f"-{steps[2]:,}", f"{steps[3]:,}"],
                                     textposition="outside", textfont=dict(size=15),
                                     decreasing=dict(marker=dict(color="#D96A00")),
                                     increasing=dict(marker=dict(color="#16803A")),
                                     totals=dict(marker=dict(color="#1F4E96")),
                                     connector=dict(line=dict(color="#9AA5B1"))))
        fig.update_layout(title="From raw rows to clean rows", yaxis_title="Rows (axis does not start at 0)", showlegend=False)
        fig.update_yaxes(range=[steps[3] * 0.97, steps[0] * 1.01])
        show(style_fig(fig, None))
    with a2, st.container(key="sec_dq2"):
        st.subheader("⚖️ Sample ratio check")
        c = st.columns(3)
        c[0].markdown(card("Control users", f"{gn['control']:,}", "", "blue", size=34), unsafe_allow_html=True)
        c[1].markdown(card("Treatment users", f"{gn['treatment']:,}", "", "orange", size=34), unsafe_allow_html=True)
        c[2].markdown(card("SRM p-value", f"{srm_p:.3f}", "Balanced split" if srm_p > 0.01 else "Possible problem",
                           "green" if srm_p > 0.01 else "red", size=34), unsafe_allow_html=True)
        st.markdown("**Group vs landing page before cleaning**")
        st.dataframe(pd.crosstab(raw["group"], raw["landing_page"], margins=True), width="stretch")
        st.caption("Control on new_page and treatment on old_page are the mismatches that were removed.")

    b1, b2 = st.columns([1.6, 1])
    with b1, st.container(key="sec_dq_traffic"):
        traffic = df.groupby(["date", "group"]).size().unstack()[G]
        fig = go.Figure([go.Bar(x=traffic.index, y=traffic[g], name=g.capitalize(), marker_color=COLORS[g]) for g in G])
        fig.update_layout(title="Daily users per group", yaxis_title="Users", barmode="group")
        show(top_legend(style_fig(fig, None)))
        st.caption("Both groups receive similar traffic every day. The first and last days are partial days.")
    with b2, st.container(key="sec_dq_dict"):
        st.subheader("📖 Data dictionary")
        st.dataframe(pd.DataFrame({
            "Column": ["user_id", "timestamp", "group", "landing_page", "converted", "bounced (optional)"],
            "Meaning": ["Unique visitor ID", "Time of the visit", "control or treatment", "old_page or new_page",
                        "1 = converted, 0 = not", "1 = bounced, 0 = stayed"]}), hide_index=True, width="stretch")
        st.caption(f"Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

# ============================================================== TAB 3: OVER TIME
with tab3:
    daily = df.groupby(["date", "group"])["converted"].mean().unstack()[G] * 100
    n_days = len(daily)
    t_ahead = int((daily["treatment"] > daily["control"]).sum())
    c_ahead = n_days - t_ahead
    max_gap = float((daily["treatment"] - daily["control"]).abs().max())
    sp = settle_point(df.sort_values("timestamp"))
    day_p = binomtest(t_ahead, n_days, 0.5).pvalue          # sign test: is one group ahead more often than chance?

    with st.container(key="sec_time_kpi"):
        c = st.columns(4)
        c[0].markdown(card("Days in test", f"{n_days}", f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}", "blue"), unsafe_allow_html=True)
        c[1].markdown(card("Days control ahead", f"{c_ahead}", f"of {n_days} days", "blue"), unsafe_allow_html=True)
        c[2].markdown(card("Days treatment ahead", f"{t_ahead}", f"of {n_days} days", "orange"), unsafe_allow_html=True)
        c[3].markdown(card("Largest daily gap", f"{max_gap:.2f} pp", "between the two groups", "purple"), unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1, st.container(key="sec_time1"):
        fig = go.Figure([go.Scatter(x=daily.index, y=daily[g], name=g.capitalize(), mode="lines+markers",
                                    line=dict(color=COLORS[g], width=3), marker=dict(size=8)) for g in G])
        fig.update_layout(title="Daily conversion rate", yaxis_title="Conversion rate (%)")
        show(top_legend(style_fig(fig, None)))
    with t2, st.container(key="sec_time2"):
        d = df.sort_values("timestamp")
        fig = go.Figure()
        for g in G:
            s = d.loc[d["group"] == g, "converted"].reset_index(drop=True)
            fig.add_trace(go.Scatter(y=(s.expanding().mean() * 100).iloc[200:], x=np.arange(200, len(s)),
                                     name=g.capitalize(), line=dict(color=COLORS[g], width=3)))
        fig.update_layout(title="Cumulative conversion rate (after the first 200 users)",
                          xaxis_title="Users in time order (per group)", yaxis_title="Conversion rate (%)")
        show(top_legend(style_fig(fig, None)))

    # ---- insight summary (numbers are calculated from the data, not typed in)
    with st.container(key="sec_insight"):
        st.subheader("🔎 What the time view tells us")
        i = st.columns(3)
        consistent = day_p < 0.05
        title0 = "One group led most days" if consistent else "No consistent daily winner"
        i[0].markdown(
            f'<div class="insight"><div class="it">{title0}</div>'
            f'<div class="ib">Treatment was ahead on {t_ahead} of {n_days} days and control on {c_ahead}. '
            f'{"This split is unlikely to be chance" if consistent else "This split is within chance"} (sign test p = {day_p:.2f}). '
            f'Daily gaps reach {max_gap:.2f} pp but change direction.</div></div>', unsafe_allow_html=True)
        i[1].markdown(
            f'<div class="insight"><div class="it">Do not stop the test early</div>'
            f'<div class="ib">Both lines settle within 0.1 pp of their final value only after about {sp:,} users per group. '
            f'Early results are unstable and can be misleading.</div></div>', unsafe_allow_html=True)
        i[2].markdown(
            f'<div class="insight"><div class="it">Test duration</div>'
            f'<div class="ib">The test ran for {n_days} calendar days (first and last are partial) and covered {df["timestamp"].dt.dayofweek.nunique()} of 7 weekdays, '
            f'so weekday effects are included in the result.</div></div>', unsafe_allow_html=True)

# ============================================================== TAB 4: POWER & IMPACT
with tab4:
    p_c, n_c, n_t = res["p"]["control"], res["n"]["control"], res["n"]["treatment"]
    pw = NormalIndPower()
    ratio = n_t / n_c
    h = pw.solve_power(effect_size=None, nobs1=n_c, alpha=alpha, power=0.8, ratio=ratio)
    mde_now = (np.sin(np.arcsin(np.sqrt(p_c)) + h / 2) ** 2 - p_c) * 100
    obs_power = pw.power(abs(proportion_effectsize(res["p"]["treatment"], p_c)), nobs1=n_c, alpha=alpha, ratio=ratio)

    with st.container(key="sec_power"):
        st.subheader("⚡ Was the test big enough?")
        c = st.columns(3)
        c[0].markdown(card("Observed effect", f"{pp:+.2f} pp", "treatment - control", "orange"), unsafe_allow_html=True)
        c[1].markdown(card("Power for the observed effect", f"{obs_power*100:.0f}%", "chance of detecting an effect this small", "purple", flash), unsafe_allow_html=True)
        c[2].markdown(card("Smallest detectable effect", f"{mde_now:.2f} pp", "with 80% power at the current sample size", "green", flash), unsafe_allow_html=True)
        st.caption("Power for the observed effect only restates the p-value, so do not use it to judge the result. "
                   "The smallest detectable effect is the better measure: effects of that size or larger would very likely have been detected.")

    # ---- row 2: planner (left) + users-needed curve (right)
    pl, pr = st.columns([1.25, 1])
    with pl, st.container(key="sec_plan"):
        st.subheader("🎯 Plan a future test: choose a minimum detectable effect (MDE)")
        a, b, c3 = st.columns([1.8, 1, 1.3])
        choice = a.radio("MDE (percentage points)", ["0.5", "1.0", "Custom"], horizontal=True)
        mde = float(choice) if choice != "Custom" else b.number_input("Custom MDE (pp)", 0.05, 20.0, 0.75, 0.05)
        target_power = c3.select_slider("Desired power", [0.7, 0.8, 0.9], value=0.8)
        base = st.number_input("Baseline conversion rate (%)", 0.1, 99.0, round(p_c * 100, 2), 0.1)
        need = int(np.ceil(pw.solve_power(effect_size=proportion_effectsize(base / 100 + mde / 100, base / 100),
                                          alpha=alpha, power=target_power, ratio=1)))
        c = st.columns(3)
        c[0].markdown(card("Users needed per group", f"{need:,}", f"to detect {mde:.2f} pp at {int(target_power*100)}% power", "blue"), unsafe_allow_html=True)
        c[1].markdown(card("Total users needed", f"{need*2:,}", "both groups combined", "purple"), unsafe_allow_html=True)
        enough = n_c >= need
        c[2].markdown(card("Current test size (per group)", f"{int(n_c):,}", "large enough for this MDE" if enough else "too small for this MDE",
                           "green" if enough else "red"), unsafe_allow_html=True)
    with pr, st.container(key="sec_curve"):
        # same calculation as the planner, repeated over a range of MDE values
        grid = np.arange(0.25, 2.01, 0.05)
        curve = [pw.solve_power(effect_size=proportion_effectsize(base / 100 + m / 100, base / 100),
                                alpha=alpha, power=target_power, ratio=1) for m in grid]
        fig = go.Figure(go.Scatter(x=grid, y=curve, mode="lines", name="Users needed per group",
                                   line=dict(color="#1F4E96", width=3)))
        fig.add_trace(go.Scatter(x=[mde], y=[need], mode="markers+text", name="Your MDE", text=[f"{need:,}"],
                                 textposition="top right", textfont=dict(size=15),
                                 marker=dict(size=14, color="#F47B00", line=dict(width=2, color="white"))))
        fig.add_hline(y=n_c, line_dash="dash", line_color="#16803A", line_width=2,
                      annotation_text="Current sample per group", annotation_font_size=14, annotation_position="top right")
        fig.update_layout(title="Users needed per group vs MDE", xaxis_title="MDE (percentage points)",
                          yaxis_title="Users per group", showlegend=False)
        show(style_fig(fig, None))

    # ---- row 3: impact cards (left) + impact range chart (right)
    il, ir = st.columns([1.25, 1])
    with il, st.container(key="sec_impact"):
        st.subheader(f"💰 Impact estimate per {visitors:,} visitors (assumption, not real traffic)")
        c = st.columns(3)
        c[0].markdown(card("Expected change in conversions", f"{res['diff']*visitors:+,.0f}", "best estimate", "orange"), unsafe_allow_html=True)
        c[1].markdown(card("Worst case (lower CI)", f"{res['lo']*visitors:+,.0f}", "conversions", "red"), unsafe_allow_html=True)
        c[2].markdown(card("Best case (upper CI)", f"{res['hi']*visitors:+,.0f}", "conversions", "green"), unsafe_allow_html=True)
        if res["lo"] < 0 < res["hi"]:
            st.caption("The range includes both losses and gains, so this result cannot prove the new page is worse or better.")
    with ir, st.container(key="sec_range"):
        fig = go.Figure(go.Scatter(x=[res["diff"] * visitors], y=[0], mode="markers", marker=dict(size=14, color="#1B232C"),
                                   error_x=dict(type="data", symmetric=False, thickness=3, width=12, color="#1B232C",
                                                array=[(res["hi"] - res["diff"]) * visitors],
                                                arrayminus=[(res["diff"] - res["lo"]) * visitors])))
        fig.add_vline(x=0, line_dash="dash", line_color="#C0281E", line_width=3,
                      annotation_text="No change", annotation_font_size=14)
        fig.update_layout(title="Change in conversions with 95% range", xaxis_title=f"Conversions per {visitors:,} visitors",
                          yaxis=dict(visible=False))
        show(style_fig(fig, None))
