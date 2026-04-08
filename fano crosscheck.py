import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Equation Cross-Checker', layout='wide')
st.title('Equation Cross-Checker')
st.markdown('Compare Eq1 vs Eq2, Eq2 vs Eq1, and test substitution logic across Eq1, Eq2, and Eq3.')

with st.sidebar:
    st.header('Parameters')
    omega0 = st.slider('ω₀', -0.5, 0.5, 0.0, 0.01)
    gamma = st.slider('γ', 0.02, 0.5, 0.10, 0.01)
    q = st.slider('q', -5.0, 5.0, 2.0, 0.1)
    l = st.slider('l', 0.1, 3.0, 1.0, 0.1)
    L = st.slider('L', 0.1, 3.0, 1.0, 0.1)
    a = st.slider('a', 0.1, 2.0, 0.5, 0.05)
    npts = st.slider('ω grid points', 100, 800, 300, 50)
    nsteps = st.slider('Integration steps', 20, 400, 120, 20)

def I_F(omega, omega0, gamma, q):
    eps = (omega - omega0) / (gamma / 2)
    return (q + eps) ** 2 / (1 + eps ** 2)

def I_L(omega, omega0, gamma, l, a, nsteps):
    dk = 1.0 / nsteps
    s = 0.0
    for i in range(nsteps):
        k = (i + 0.5) * dk
        lor = 1.0 / ((omega - omega0) ** 2 + (gamma / 2) ** 2)
        gauss = np.exp(-(k ** 2) * (l ** 2) / (4 * a ** 2))
        s += lor * gauss * dk
    return s

def I3_from_IF(omega, omega0, gamma, q, L, a, nsteps):
    dk = 1.0 / nsteps
    s = 0.0
    fano = I_F(omega, omega0, gamma, q)
    for i in range(nsteps):
        k = (i + 0.5) * dk
        gauss = np.exp(-(k ** 2) * (L ** 2) / (4 * a ** 2))
        s += fano * gauss * dk
    return s

def I3_from_IL(omega, omega0, gamma, l, L, a, nsteps):
    dk = 1.0 / nsteps
    s = 0.0
    il = I_L(omega, omega0, gamma, l, a, nsteps)
    for i in range(nsteps):
        k = (i + 0.5) * dk
        gauss = np.exp(-(k ** 2) * (L ** 2) / (4 * a ** 2))
        s += il * gauss * dk
    return s

def normalize(arr):
    arr = np.asarray(arr)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

omega = np.linspace(-1.0, 1.0, npts)
IF_vals = np.array([I_F(w, omega0, gamma, q) for w in omega])
IL_vals = np.array([I_L(w, omega0, gamma, l, a, nsteps) for w in omega])
I3_IF_vals = np.array([I3_from_IF(w, omega0, gamma, q, L, a, nsteps) for w in omega])
I3_IL_vals = np.array([I3_from_IL(w, omega0, gamma, l, L, a, nsteps) for w in omega])

IF_n = normalize(IF_vals)
IL_n = normalize(IL_vals)
I3_IF_n = normalize(I3_IF_vals)
I3_IL_n = normalize(I3_IL_vals)

st.subheader('Symmetric comparison')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Eq1 vs Eq2 corr', f'{corr(IL_n, IF_n):.4f}')
c2.metric('Eq2 vs Eq1 corr', f'{corr(IF_n, IL_n):.4f}')
c3.metric('Eq1 vs Eq2 RMSE', f'{rmse(IL_n, IF_n):.4f}')
c4.metric('Eq2 vs Eq1 RMSE', f'{rmse(IF_n, IL_n):.4f}')

st.info('Eq1 vs Eq2 and Eq2 vs Eq1 are numerically identical for symmetric metrics like correlation and RMSE.')

st.subheader('Substitution checks')
d1, d2 = st.columns(2)
d1.metric('Eq3(I_F) vs Eq2 corr', f'{corr(I3_IF_n, IF_n):.4f}')
d2.metric('Eq3(I_L) vs Eq1 corr', f'{corr(I3_IL_n, IL_n):.4f}')

fig = go.Figure()
fig.add_trace(go.Scatter(x=omega, y=IL_n, mode='lines', name='Eq1: I_L'))
fig.add_trace(go.Scatter(x=omega, y=IF_n, mode='lines', name='Eq2: I_F'))
fig.add_trace(go.Scatter(x=omega, y=I3_IF_n, mode='lines', name='Eq3 from Eq2'))
fig.add_trace(go.Scatter(x=omega, y=I3_IL_n, mode='lines', name='Eq3 from Eq1'))
fig.update_layout(template='plotly_white', height=500, title='Normalized comparison')
st.plotly_chart(fig, use_container_width=True)

df = pd.DataFrame({
    'omega': omega,
    'Eq1_IL': IL_vals,
    'Eq2_IF': IF_vals,
    'Eq3_from_IF': I3_IF_vals,
    'Eq3_from_IL': I3_IL_vals,
    'Eq1_IL_norm': IL_n,
    'Eq2_IF_norm': IF_n,
    'Eq3_from_IF_norm': I3_IF_n,
    'Eq3_from_IL_norm': I3_IL_n,
})
st.dataframe(df, use_container_width=True)
