import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Equation Comparison Tool', layout='wide')
st.title('Equation Comparison Tool')
st.caption('Select Eq.1, Eq.2, and/or Eq.3 using checkboxes, adjust shared sliders, and compare the resulting curves.')

st.markdown("## Equations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Eq. 1")
    st.latex(r"I_1(\omega)=\int_0^1 \frac{\exp\left(-k^2L^2/4a^2\right)}{[\omega-\omega_0]^2+(\Gamma/2)^2}\,dk")

with col2:
    st.markdown("### Eq. 2")
    st.latex(r"I_2(\omega)=\frac{(q+\epsilon)^2}{1+\epsilon^2}")
    st.latex(r"\epsilon=\frac{\omega-\omega_0}{\Gamma/2}")

st.markdown("### Eq. 3")
st.latex(r"I_3(\omega)=\int_0^1 \left[\frac{(\epsilon+q)^2}{1+\epsilon^2}\right]\exp\left(-\frac{k^2L^2}{4a^2}\right)\,dk")
st.latex(r"\epsilon=\frac{\omega-\omega(k)}{\Gamma/2},\qquad \omega(k)=\left[A+B\cos\left(\frac{\pi k}{2}\right)\right]^{1/2}")

with st.sidebar:
    st.header('Select equations to compare')
    show_eq1 = st.checkbox('Show Eq. 1', value=True)
    show_eq2 = st.checkbox('Show Eq. 2', value=True)
    show_eq3 = st.checkbox('Show Eq. 3', value=True)

    st.header('Shared parameters')
    omega0 = st.slider('ω₀', 480.0, 560.0, 520.0, 1.0)
    Gamma = st.slider('Γ', 1.0, 50.0, 10.0, 0.5)
    q = st.slider('q', -10.0, 10.0, 2.0, 0.1)
    L = st.slider('L (used in Eq.1 and Eq.3)', 0.1, 10.0, 3.0, 0.1)
    a = st.slider('a', 0.1, 2.0, 0.543, 0.001)

    st.header('Dispersion constants for Eq. 3')
    A = st.number_input('A (cm^-2)', value=171400.0, step=100.0)
    B = st.number_input('B (cm^-2)', value=100000.0, step=100.0)

    st.header('Numerical resolution')
    omega_min = st.slider('ω min', 400.0, 600.0, 450.0, 1.0)
    omega_max = st.slider('ω max', 450.0, 700.0, 600.0, 1.0)
    npts = st.slider('ω points', 100, 1000, 400, 50)
    nsteps = st.slider('Integration steps', 50, 1000, 300, 50)
    normalize_curves = st.checkbox('Normalize curves for shape comparison', value=True)


def omega_k(k, A, B):
    return np.sqrt(A + B * np.cos(np.pi * k / 2.0))


def eq1(omega, omega0, Gamma, L, a, nsteps):
    dk = 1.0 / nsteps
    s = 0.0
    for i in range(nsteps):
        k = (i + 0.5) * dk
        confinement = np.exp(-(k ** 2) * (L ** 2) / (4 * a ** 2))
        lorentz = 1.0 / (((omega - omega0) ** 2) + (Gamma / 2.0) ** 2)
        s += confinement * lorentz * dk
    return s


def eq2(omega, omega0, Gamma, q):
    eps = (omega - omega0) / (Gamma / 2.0)
    return ((eps + q) ** 2) / (1.0 + eps ** 2)


def eq3(omega, Gamma, q, L, a, A, B, nsteps):
    dk = 1.0 / nsteps
    s = 0.0
    for i in range(nsteps):
        k = (i + 0.5) * dk
        wk = omega_k(k, A, B)
        eps = (omega - wk) / (Gamma / 2.0)
        fano = ((eps + q) ** 2) / (1.0 + eps ** 2)
        confinement = np.exp(-(k ** 2) * (L ** 2) / (4 * a ** 2))
        s += fano * confinement * dk
    return s


def normalize(y):
    y = np.asarray(y)
    ymin, ymax = y.min(), y.max()
    if ymax - ymin == 0:
        return np.zeros_like(y)
    return (y - ymin) / (ymax - ymin)


def correlation(y1, y2):
    if np.std(y1) == 0 or np.std(y2) == 0:
        return 0.0
    return float(np.corrcoef(y1, y2)[0, 1])


def rmse(y1, y2):
    return float(np.sqrt(np.mean((np.asarray(y1) - np.asarray(y2)) ** 2)))


def peak_shift(y1, y2, omega_grid):
    return float(abs(omega_grid[np.argmax(y1)] - omega_grid[np.argmax(y2)]))


omega = np.linspace(omega_min, omega_max, npts)

y1 = np.array([eq1(w, omega0, Gamma, L, a, nsteps) for w in omega])
y2 = np.array([eq2(w, omega0, Gamma, q) for w in omega])
y3 = np.array([eq3(w, Gamma, q, L, a, A, B, nsteps) for w in omega])

plot_y1 = normalize(y1) if normalize_curves else y1
plot_y2 = normalize(y2) if normalize_curves else y2
plot_y3 = normalize(y3) if normalize_curves else y3

selected = []
if show_eq1:
    selected.append(('Eq. 1', plot_y1, '#01696f'))
if show_eq2:
    selected.append(('Eq. 2', plot_y2, '#da7101'))
if show_eq3:
    selected.append(('Eq. 3', plot_y3, '#7a39bb'))

if not selected:
    st.warning('Please select at least one equation.')
else:
    st.markdown('## Plot')
    fig = go.Figure()
    for name, y, color in selected:
        fig.add_trace(go.Scatter(x=omega, y=y, mode='lines', name=name, line=dict(width=3, color=color)))
    fig.update_layout(
        template='plotly_white',
        height=500,
        xaxis_title='ω',
        yaxis_title='Intensity'
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(selected) >= 2:
        st.markdown('## Pairwise comparison metrics')
        rows = []
        name_to_y = {name: y for name, y, _ in selected}
        names = list(name_to_y.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                yy1, yy2 = name_to_y[n1], name_to_y[n2]
                rows.append({
                    'Equation pair': f'{n1} vs {n2}',
                    'Correlation': round(correlation(yy1, yy2), 6),
                    'RMSE': round(rmse(yy1, yy2), 6),
                    'Peak shift': round(peak_shift(yy1, yy2, omega), 6),
                })

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True)

st.markdown('## Notes')
st.markdown('- Eq. 1 and Eq. 3 now use the same confinement parameter L.')
st.markdown('- Eq. 2 is the direct Fano profile.')
st.markdown('- Eq. 3 is displayed in full-width so the equation is not cut.')
st.markdown('- Curves can be normalized to compare shape only, or left unnormalized to compare raw magnitude.')

full_df = pd.DataFrame({
    'omega': omega,
    'Eq1': y1,
    'Eq2': y2,
    'Eq3': y3,
    'Eq1_plot': plot_y1,
    'Eq2_plot': plot_y2,
    'Eq3_plot': plot_y3,
})

st.markdown('## Computed data')
st.dataframe(full_df, use_container_width=True)
st.download_button(
    'Download computed CSV',
    full_df.to_csv(index=False).encode('utf-8'),
    'equation_comparison.csv',
    'text/csv'
)
