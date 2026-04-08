import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Equation Cross-Checker', layout='wide')
st.title('Equation Cross-Checker')
st.caption('Select any equation pair or substitution path using checkboxes, then compare with one shared parameter panel.')

st.markdown('### Equation forms')
col_eq1, col_eq2, col_eq3 = st.columns(3)
with col_eq1:
    st.latex(r"I_L(\omega)=\int_0^1 \frac{\exp\left(-k^2 l^2 / 4a^2\right)}{[\omega-\omega_0]^2+(\gamma/2)^2}\,dk")
with col_eq2:
    st.latex(r"I_F(\omega)=\frac{(q+\epsilon)^2}{1+\epsilon^2},\quad \epsilon=\frac{\omega-\omega_0}{\gamma/2}")
with col_eq3:
    st.latex(r"I_3(\omega)=\int_0^1 W(\omega)\exp\left(-k^2 L^2 / 4a^2\right)\,dk")

with st.sidebar:
    st.header('Choose comparison')
    compare_eq1_eq2 = st.checkbox('Eq1 vs Eq2', value=True)
    compare_eq2_eq1 = st.checkbox('Eq2 vs Eq1', value=False)
    compare_eq3_if = st.checkbox('Eq3 using Eq2 result', value=True)
    compare_eq3_il = st.checkbox('Eq3 using Eq1 result', value=True)
    compare_all = st.checkbox('Show all selected together', value=True)

    st.header('Shared sliders')
    omega0 = st.slider('ω₀ (center frequency)', -0.5, 0.5, 0.0, 0.01)
    gamma = st.slider('γ (linewidth)', 0.02, 0.5, 0.10, 0.01)
    q = st.slider('q (Fano asymmetry)', -5.0, 5.0, 2.0, 0.1)
    l = st.slider('l (Eq1 broadening)', 0.1, 3.0, 1.0, 0.1)
    L = st.slider('L (Eq3 broadening)', 0.1, 3.0, 1.0, 0.1)
    a = st.slider('a (scaling)', 0.1, 2.0, 0.5, 0.05)
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

def peak_shift(a, b, omega):
    return float(abs(omega[np.argmax(a)] - omega[np.argmax(b)]))

def compare_block(title, x, y, label_x, label_y, omega):
    c1, c2, c3 = st.columns(3)
    c1.metric('Correlation', f'{corr(x, y):.4f}')
    c2.metric('RMSE', f'{rmse(x, y):.4f}')
    c3.metric('Peak shift', f'{peak_shift(x, y, omega):.4f}')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=omega, y=x, mode='lines', name=label_x, line=dict(width=3, color='#01696f')))
    fig.add_trace(go.Scatter(x=omega, y=y, mode='lines', name=label_y, line=dict(width=3, color='#da7101')))
    fig.update_layout(title=title, template='plotly_white', height=420)
    st.plotly_chart(fig, use_container_width=True)

omega = np.linspace(-1.0, 1.0, npts)
IF_vals = np.array([I_F(w, omega0, gamma, q) for w in omega])
IL_vals = np.array([I_L(w, omega0, gamma, l, a, nsteps) for w in omega])
I3_IF_vals = np.array([I3_from_IF(w, omega0, gamma, q, L, a, nsteps) for w in omega])
I3_IL_vals = np.array([I3_from_IL(w, omega0, gamma, l, L, a, nsteps) for w in omega])

IF_n = normalize(IF_vals)
IL_n = normalize(IL_vals)
I3_IF_n = normalize(I3_IF_vals)
I3_IL_n = normalize(I3_IL_vals)

selected = []
if compare_eq1_eq2:
    selected.append(('Eq1 vs Eq2', IL_n, IF_n, 'Eq1: I_L', 'Eq2: I_F'))
if compare_eq2_eq1:
    selected.append(('Eq2 vs Eq1', IF_n, IL_n, 'Eq2: I_F', 'Eq1: I_L'))
if compare_eq3_if:
    selected.append(('Eq3 using Eq2 result', I3_IF_n, IF_n, 'Eq3 from Eq2', 'Eq2: I_F'))
if compare_eq3_il:
    selected.append(('Eq3 using Eq1 result', I3_IL_n, IL_n, 'Eq3 from Eq1', 'Eq1: I_L'))

if not selected:
    st.warning('Please select at least one comparison from the sidebar.')
else:
    st.subheader('Selected comparisons')
    for title, x, y, lx, ly in selected:
        st.markdown(f'#### {title}')
        compare_block(title, x, y, lx, ly, omega)

if compare_all and selected:
    st.subheader('All selected curves together')
    fig_all = go.Figure()
    colors = ['#01696f', '#da7101', '#7a39bb', '#d19900', '#a12c7b', '#006494']
    data_map = {
        'Eq1: I_L': IL_n,
        'Eq2: I_F': IF_n,
        'Eq3 from Eq2': I3_IF_n,
        'Eq3 from Eq1': I3_IL_n,
    }
    added = []
    for _, _, _, lx, ly in selected:
        for name in [lx, ly]:
            if name not in added and name in data_map:
                fig_all.add_trace(go.Scatter(x=omega, y=data_map[name], mode='lines', name=name, line=dict(width=3, color=colors[len(added) % len(colors)])))
                added.append(name)
    fig_all.update_layout(template='plotly_white', height=480, title='Combined selected equations')
    st.plotly_chart(fig_all, use_container_width=True)

st.subheader('Interpretation notes')
st.markdown('- **Eq1 vs Eq2** and **Eq2 vs Eq1** will give the same correlation, RMSE, and peak shift because these are symmetric comparison metrics.')
st.markdown('- **Eq3 using Eq2** is mathematically the more natural substitution if Eq3 is built from the Fano term.')
st.markdown('- **Eq3 using Eq1** can still be checked numerically, but it should be interpreted as a test case, not always a strict analytical identity.')

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

st.subheader('Computed data table')
st.dataframe(df, use_container_width=True)
csv = df.to_csv(index=False).encode('utf-8')
st.download_button('Download CSV', data=csv, file_name='equation_crosscheck_data.csv', mime='text/csv')
