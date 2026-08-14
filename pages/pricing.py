import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import ncx2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
def GBM(sigma, S0, T, r, q):
    # Analytical Solution
    Z = np.random.normal(0,1)
    drift = r - q
    S = S0 * np.exp((drift - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return S

def CEV(sigma,S0, T, r, q, steps, constant):
    dt = T / steps
    S = S0
    drift = r - q
    # Euler-Maruyama Discretisation
    for _ in range(steps):
        Z = np.random.normal()
        S += drift*S*dt + sigma*(S**constant)*np.sqrt(dt)*Z
    return S

def MJD(sigma, S0, T, r, q, steps, jump, jump_rate):
    dt = T / steps
    S = S0
    drift = r - q
    # Euler-Maruyama Discretisation
    for _ in range(steps):
        S += drift*dt*S + sigma*S*np.sqrt(dt)*np.random.normal(0,1) + S*(jump-1)*np.random.poisson(jump_rate*dt)
    return S

def Heston(S0, T, r, q, theta, zeta, kappa, rho, v0, steps):
    dt = T/ steps
    v = v0
    S = S0
    drift = r - q
    # Euler-Maruyama Discretisation
    for _ in range(steps):
        Z1 = np.random.normal(0,1)
        Z2 = np.random.normal(0,1)
        Zv = Z1
        Zs = rho*Z1 + np.sqrt(1-rho**2)*Z2
        v +=  kappa*(theta - v)*dt + zeta*np.sqrt(v)*np.sqrt(dt)*Zv
        S +=  drift*S*dt + np.sqrt(v)*S*np.sqrt(dt)*Zs
    return S

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Black Scholes
def compute_black_scholes_price(S,K,T,r,sigma):
    # Intermediates
    d1 = (np.log(S/K)+(r+(sigma*sigma)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Call option price
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    # Put option price
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    return call_price, put_price

# Constant of Elasticity Variance Model (CEV)
def compute_cev_price(S,K,T,r,sigma,constant):
    beta = 2*constant
    x = (2*r)/((sigma**2)*(2-beta)*(np.exp(r*(2-beta)*T)-1)) * S**(2 - beta) * np.exp(r * (2 - beta) * T)
    y = (2*r)/((sigma**2)*(2-beta)*(1-np.exp(-r*(2-beta)*T))) * K**(2-beta)
    call_price = S*ncx2.sf(2*y,(2+2/(2-beta)),(2*x)) - K*np.exp(-r*T)*(1-ncx2.sf(2*x,(2/(2-beta)),(2*y)))
    put_price = K*np.exp(-r*T)*ncx2.sf(2*x,(2/(2-beta)),(2*y)) - S*(1-ncx2.sf(2*y,(2+2/(2-beta)),(2*x)))

    return call_price, put_price

# Payoff helper function
def compute_payoff(S,K,option="Call"):
    payoffs = []

    for price in S:
        if option == "Call":
            payoff = max(price-K,0)
        elif option == "Put":
            payoff = max(K-price,0)
        payoffs.append(payoff)

    return payoffs

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def monte_carlo(S,K,T,r,q,sigma,n_sims,steps,constant,jump,jump_rate,theta,zeta,kappa,rho,v0,motion):
    payoffs_call = []
    payoffs_put = []
    running_sum = 0
    running_sum_squared = 0
    running_mean = []
    running_std = []
    running_interval_lower = []
    running_interval_upper = []
    running_interval_width = []
    z = 1.96 # 95% confidence interval

    # Paths
    for i in tqdm(range(n_sims),desc='Monte Carlo Simulation'):

        if motion == "GBM":
            S_final = GBM(sigma,S,T,r,q)
        elif motion == "CEV":
            S_final = CEV(sigma,S,T,r,q,steps,constant)
        elif motion == "Heston":
            S_final = Heston(S,T,r,q,theta,zeta,kappa,rho,v0,steps)
        elif motion == "MJD":
            S_final = MJD(sigma,S,T,r,q,steps,jump,jump_rate)

        payoff_call = max(S_final - K,0)
        payoff_put = max(K-S_final,0)
        payoffs_call.append(payoff_call)
        payoffs_put.append(payoff_put)
        running_sum += S_final
        running_sum_squared += S_final**2
        mean = running_sum / (i + 1)
        running_mean.append(mean)
        std = np.sqrt(running_sum_squared/(i+1) - mean**2)
        running_std.append(std)
        lower = mean - z*std / np.sqrt(i+1)
        upper = mean + z*std / np.sqrt(i+1)
        width = 2*z*std / np.sqrt(i+1)
        running_interval_lower.append(lower)
        running_interval_upper.append(upper)
        running_interval_width.append(width)

    # Mean payoff
    mean_payoff_call = np.mean(payoffs_call).item()
    mean_payoff_put = np.mean(payoffs_put).item()

    # Discount
    call_price = np.exp(-r*T) * mean_payoff_call
    put_price = np.exp(-r*T) * mean_payoff_put

    convergence_data = {"Mean": running_mean, 
        "Standard Deviation": running_std,
        "Lower Value": running_interval_lower,
        "Upper Value": running_interval_upper,
        "Confidence Width": running_interval_width}

    convergence = pd.DataFrame(convergence_data)

    return call_price, put_price, convergence

# Finite-Differencing
def forward_euler(S0,K,T,r,constant,s_steps,t_steps,motion): # Explicit
    # Grid spacing
    S_max = 2000
    dt = T/t_steps
    dS = S_max/s_steps

    # Grid discretisation
    S = np.linspace(0,S_max,s_steps+1)
    t = np.linspace(0,T,t_steps+1)
    V_call = np.zeros((s_steps+1, t_steps+1))
    V_put = np.zeros((s_steps+1, t_steps+1))

    # Terminal payoff condition
    V_call[:,-1] = np.maximum(S - K, 0)
    V_put[:,-1] = np.maximum(K - S, 0)

    # Coefficients for explicit scheme
    if motion == "GBM":
        a = 0.5*dt*(sigma**2*S**2/dS**2 - r*S/dS)
        b = 1 - dt*(sigma**2*S**2/dS**2 + r)
        c = 0.5*dt*(sigma**2*S**2/dS**2 + r*S/dS)

    elif motion == "CEV":
        a = 0.5*dt*(sigma**2*S**(2*constant)/dS**2 - r*S/dS)
        b = 1 - dt*(sigma**2*S**(2*constant)/dS**2 + r)
        c = 0.5*dt*(sigma**2*S**(2*constant)/dS**2 + r*S/dS)

    elif motion == "Heston":
        a = 0.5*dt*(sigma**2*S**(2*constant)/dS**2 - r*S/dS)
        b = 1 - dt*(sigma**2*S**(2*constant)/dS**2 + r)
        c = 0.5*dt*(sigma**2*S**(2*constant)/dS**2 + r*S/dS)

    elif motion == "MJD":
        a = 0.5*dt*(sigma**2*S**(2*constant)/dS**2 - r*S/dS)
        b = 1 - dt*(sigma**2*S**(2*constant)/dS**2 + r)
        c = 0.5*dt*(sigma**2*S**(2*constant)/dS**2 + r*S/dS)

    # Solve backwards in time
    for j in range(t_steps-1, -1, -1):
        # Boundary conditions
        tau = T - t[j]
        V_call[0,j] = 0
        V_call[-1,j] = S_max - K*np.exp(-r*tau)
        V_put[0,j] = K*np.exp(-r*tau)
        V_put[-1,j] = 0

        for i in range(1, s_steps):
            V_call[i,j] = a[i]*V_call[i-1,j+1] + b[i]*V_call[i,j+1] + c[i]*V_call[i+1,j+1]
            V_put[i,j] = a[i]*V_put[i-1,j+1] + b[i]*V_put[i,j+1] + c[i]*V_put[i+1,j+1]

    # Find id of spot price closest to desired spot
    idx = np.argmin(np.abs(S - S0))
    call_price = V_call[idx,0]
    put_price = V_put[idx,0]

    return call_price, put_price

def backward_euler(S0,K,T,r,constant,s_steps,t_steps,motion): # Implicit
    # Grid spacing
    S_max = 2000
    dt = T/t_steps
    dS = S_max/s_steps

    # Grid discretisation
    S = np.linspace(0,S_max,s_steps+1)
    t = np.linspace(0,T,t_steps+1)
    V_call = np.zeros((s_steps+1, t_steps+1))
    V_put = np.zeros((s_steps+1, t_steps+1))

    # Terminal payoff condition
    V_call[:,-1] = np.maximum(S - K, 0)
    V_put[:,-1] = np.maximum(K - S, 0)

    # Coefficients
    A = np.zeros(s_steps + 1)
    B = np.zeros(s_steps + 1)
    C = np.zeros(s_steps + 1)
    D = np.zeros(s_steps + 1)
    E = np.zeros(s_steps + 1)
    F = np.zeros(s_steps + 1)

    # Coefficients for implicit scheme
    if motion == "GBM":
        for i in range(1, s_steps):
            A[i] = -dt / 2 * (sigma**2 * S[i]**2 / dS**2 - r * S[i] / dS)
            B[i] = 1 + dt * (sigma**2 * S[i]**2 / dS**2 + r)
            C[i] = -dt / 2 * (sigma**2 * S[i]**2 / dS**2+ r * S[i] / dS)

    elif motion == "CEV":
        for i in range(1, s_steps):
            A[i] = -dt / 2 * (sigma**2 * S[i]**(2*constant) / dS**2- r * S[i] / dS)
            B[i] = 1 + dt / 2 * (sigma**2 * S[i]**(2*constant) / dS**2+ r)
            C[i] = -dt / 2 * (sigma**2 * S[i]**(2*constant) / dS**2+ r * S[i] / dS)

    # Solve backwards in time
    for j in range(t_steps - 1, -1, -1):
        tau = T - t[j]

        # Boundary conditions
        V_call[0, j] = 0
        V_call[-1, j] = S_max - K * np.exp(-r * tau)
        V_put[0, j] = K * np.exp(-r * tau)
        V_put[-1, j] = 0

        # RHS
        rhs_call = np.zeros(s_steps - 1)
        rhs_put = np.zeros(s_steps - 1)

        for i in range(1, s_steps):
            rhs_call[i - 1] = V_call[i, j + 1]
            rhs_put[i - 1] = V_put[i, j + 1]

        rhs_call[0] -= A[1] * V_call[0, j]
        rhs_call[-1] -= C[s_steps - 1] * V_call[-1, j]

        rhs_put[0] -= A[1] * V_put[0, j]
        rhs_put[-1] -= C[s_steps - 1] * V_put[-1, j]

        # Thomas algorithm solving tridiagonal matrix for call
        lower = A[1:s_steps].copy()
        diag = B[1:s_steps].copy()
        upper = C[1:s_steps].copy()

        # Forward elimination
        for i in range(1, s_steps - 1):
            m = lower[i] / diag[i - 1]
            diag[i] = diag[i] - m * upper[i - 1]
            rhs_call[i] = rhs_call[i] - m * rhs_call[i - 1]

        # Back substitution
        x_call = np.zeros(s_steps - 1)
        x_call[-1] = rhs_call[-1] / diag[-1]

        for i in range(s_steps - 3, -1, -1):
            x_call[i] = (rhs_call[i]- upper[i] * x_call[i + 1]) / diag[i]

        V_call[1:s_steps, j] = x_call

        # Thomas algorithm for put
        lower = A[1:s_steps].copy()
        diag = B[1:s_steps].copy()
        upper = C[1:s_steps].copy()

        # Forward elimination
        for i in range(1, s_steps - 1):
            m = lower[i] / diag[i - 1]
            diag[i] = diag[i] - m * upper[i - 1]
            rhs_put[i] = rhs_put[i] - m * rhs_put[i - 1]

        # Back substitution
        x_put = np.zeros(s_steps - 1)
        x_put[-1] = rhs_put[-1] / diag[-1]

        for i in range(s_steps - 3, -1, -1):
            x_put[i] = (rhs_put[i]- upper[i] * x_put[i + 1]) / diag[i]

        V_put[1:s_steps, j] = x_put

    # Find id of spot price closest to desired spot
    idx = np.argmin(np.abs(S - S0))
    call_price = V_call[idx, 0]
    put_price = V_put[idx, 0]

    return call_price, put_price

def crank_nicolson(S0,K,T,r,constant,s_steps,t_steps,motion):
    # Grid spacing
    S_max = 2000
    dt = T/t_steps
    dS = S_max/s_steps

    # Grid discretisation
    S = np.linspace(0,S_max,s_steps+1)
    t = np.linspace(0,T,t_steps+1)
    V_call = np.zeros((s_steps+1, t_steps+1))
    V_put = np.zeros((s_steps+1, t_steps+1))

    # Terminal payoff condition
    V_call[:,-1] = np.maximum(S - K, 0)
    V_put[:,-1] = np.maximum(K - S, 0)

    # Coefficients
    A = np.zeros(s_steps + 1)
    B = np.zeros(s_steps + 1)
    C = np.zeros(s_steps + 1)
    D = np.zeros(s_steps + 1)
    E = np.zeros(s_steps + 1)
    F = np.zeros(s_steps + 1)

    # Coefficients for Carnk-Nicolson scheme
    if motion == "GBM":
        for i in range(1, s_steps):
            A[i] = -dt / 4 * (sigma**2 * S[i]**2 / dS**2- r * S[i] / dS)
            B[i] = 1 + dt / 2 * (sigma**2 * S[i]**2 / dS**2+ r)
            C[i] = -dt / 4 * (sigma**2 * S[i]**2 / dS**2+ r * S[i] / dS)
            D[i] = dt / 4 * (sigma**2 * S[i]**2 / dS**2- r * S[i] / dS)
            E[i] = 1 - dt / 2 * (sigma**2 * S[i]**2 / dS**2+ r)
            F[i] = dt / 4 * (sigma**2 * S[i]**2 / dS**2+ r * S[i] / dS)

    elif motion == "CEV":
        for i in range(1, s_steps):
            A[i] = -dt / 4 * (sigma**2 * S[i]**(2*constant) / dS**2- r * S[i] / dS)
            B[i] = 1 + dt / 2 * (sigma**2 * S[i]**(2*constant) / dS**2+ r)
            C[i] = -dt / 4 * (sigma**2 * S[i]**(2*constant) / dS**2+ r * S[i] / dS)
            D[i] = dt / 4 * (sigma**2 * S[i]**(2*constant) / dS**2- r * S[i] / dS)
            E[i] = 1 - dt / 2 * (sigma**2 * S[i]**(2*constant) / dS**2+ r)
            F[i] = dt / 4 * (sigma**2 * S[i]**(2*constant) / dS**2+ r * S[i] / dS)

    # Solve backwards in time
    for j in range(t_steps - 1, -1, -1):
        tau = T - t[j]

        # Boundary conditions
        V_call[0, j] = 0
        V_call[-1, j] = S_max - K * np.exp(-r * tau)
        V_put[0, j] = K * np.exp(-r * tau)
        V_put[-1, j] = 0

        # RHS
        rhs_call = np.zeros(s_steps - 1)
        rhs_put = np.zeros(s_steps - 1)

        for i in range(1, s_steps):
            rhs_call[i - 1] = (D[i] * V_call[i - 1, j + 1]+ E[i] * V_call[i, j + 1]+ F[i] * V_call[i + 1, j + 1])
            rhs_put[i - 1] = (D[i] * V_put[i - 1, j + 1]+ E[i] * V_put[i, j + 1]+ F[i] * V_put[i + 1, j + 1])

        rhs_call[0] -= A[1] * V_call[0, j]
        rhs_call[-1] -= C[s_steps - 1] * V_call[-1, j]

        rhs_put[0] -= A[1] * V_put[0, j]
        rhs_put[-1] -= C[s_steps - 1] * V_put[-1, j]

        # Thomas algorithm solving tridiagonal matrix for call
        lower = A[1:s_steps].copy()
        diag = B[1:s_steps].copy()
        upper = C[1:s_steps].copy()

        # Forward elimination
        for i in range(1, s_steps - 1):
            m = lower[i] / diag[i - 1]
            diag[i] = diag[i] - m * upper[i - 1]
            rhs_call[i] = rhs_call[i] - m * rhs_call[i - 1]

        # Back substitution
        x_call = np.zeros(s_steps - 1)
        x_call[-1] = rhs_call[-1] / diag[-1]

        for i in range(s_steps - 3, -1, -1):
            x_call[i] = (rhs_call[i]- upper[i] * x_call[i + 1]) / diag[i]

        V_call[1:s_steps, j] = x_call

        # Thomas algorithm for put
        lower = A[1:s_steps].copy()
        diag = B[1:s_steps].copy()
        upper = C[1:s_steps].copy()

        # Forward elimination
        for i in range(1, s_steps - 1):
            m = lower[i] / diag[i - 1]
            diag[i] = diag[i] - m * upper[i - 1]
            rhs_put[i] = rhs_put[i] - m * rhs_put[i - 1]

        # Back substitution
        x_put = np.zeros(s_steps - 1)
        x_put[-1] = rhs_put[-1] / diag[-1]

        for i in range(s_steps - 3, -1, -1):
            x_put[i] = (rhs_put[i]- upper[i] * x_put[i + 1]) / diag[i]

        V_put[1:s_steps, j] = x_put

    # Find id of spot price closest to desired spot
    idx = np.argmin(np.abs(S - S0))
    call_price = V_call[idx, 0]
    put_price = V_put[idx, 0]

    return call_price, put_price

# Mariana
# ---------------------------------------------------------------------------
# Default Parameters
# ---------------------------------------------------------------------------
S = 100 
K = 120 
T = 0.10 
r = 0.03 
q = 0.00
sigma = 0.40 
constant = 0.75
jump = 0.50
jump_rate = 0.20
theta = 0.04
zeta = 0.5
kappa = 2
rho = -0.7 # Typically between -0.9 to -0.3
v0 = 0.04
n_sims = 100000
s_steps = 200
t_steps = 850
n_sims = 100000
steps = 252
run = False

# Options
if "call_price" not in st.session_state:
    st.session_state.call_price = 0

if "put_price" not in st.session_state:
    st.session_state.put_price = 0

if "call_price_monte" not in st.session_state:
    st.session_state.call_price_monte = 0

if "put_price_monte" not in st.session_state:
    st.session_state.put_price_monte = 0

if "call_price_forward_euler" not in st.session_state:
    st.session_state.call_price_forward_euler = 0

if "put_price_forward_euler" not in st.session_state:
    st.session_state.put_price_forward_euler = 0

if "call_price_backward_euler" not in st.session_state:
    st.session_state.call_price_backward_euler = 0

if "put_price_backward_euler" not in st.session_state:
    st.session_state.put_price_backward_euler = 0

if "call_price_crank_nicolson" not in st.session_state:
    st.session_state.call_price_crank_nicolson = 0

if "put_price_crank_nicolson" not in st.session_state:
    st.session_state.put_price_crank_nicolson = 0

if "call_price_finite_difference" not in st.session_state:
    st.session_state.call_price_finite_difference = 0

if "put_price_finite_difference" not in st.session_state:
    st.session_state.put_price_finite_difference = 0

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Pricing PDE Solver (European Options)",layout="wide")

st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

custom_css = """
<style>
    /* Custom style for the active tab */
    .stTabs > .tablist > .react-tabs__tab--selected {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
    }
    /* Custom style for all tabs */
    .stTabs > .tablist > .react-tabs__tab {
        background-color: #e8e8e8;
        color: #4f4f4f;
        font-family: 'Courier New', Courier, monospace;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Solver (European Options)",text_alignment="center")

    # Setup
    framework = st.selectbox("Framework",("Black-Scholes (GBM)","Constant of Elasticity Variance (CEV)", "Heston","Jump Diffusion"))

    option_type = st.selectbox("Option Type",("Call", "Put"))

    # Market and Contract
    st.write("Market and Contract")

    col1,col2 = st.columns(2)

    with col1:
        S = st.text_input("Spot (S₀)", f"{S}")
        r = st.text_input("Rate (r)", f"{r}")
        sigma = st.text_input("Implied Vol (σ)", f"{sigma}")

    with col2:
        K = st.text_input("Strike (K)", f"{K}")
        T = st.text_input("Maturity (T)", f"{T}")
        q = st.text_input("Dividend Yield (q)",f"{q}")
    
    # Model Parameters
    if framework != "Black-Scholes (GBM)":
        st.write("Model Parameters")

    col1,col2 = st.columns(2)

    with col1:
        if framework == "Constant of Elasticity Variance (CEV)":
            constant = st.text_input("Constant (λ)",f"{constant}")
        if framework == "Heston":
            kappa = st.text_input("Mean Reversion Speed (κ)",f"{kappa}")
            zeta = st.text_input("Volatility of Volatility (ξ)",f"{zeta}")
            v0 = st.text_input("Initial Variance (v0​)",f"{v0}")
        if framework == "Jump Diffusion":
            jump_rate = st.text_input("Jump Rate (λ)",f"{jump_rate}")
    
    with col2:
        if framework == "Jump Diffusion":
            jump = st.text_input("Jump Multplier (J)",f"{jump}")
        if framework == "Heston":
            theta = st.text_input("Long Term Variance (θ)",f"{theta}")
            rho = st.text_input("Correlation (ρ)",f"{rho}")

    # Solvers
    st.write("Finite Differencing")

    scheme = st.selectbox("Scheme",("Crank-Nicolson (2nd Order)","Forward Euler","Backward Euler"))

    col1,col2 = st.columns(2)

    with col1:
        s_steps = st.text_input("S steps",f"{s_steps}")

    with col2:
        t_steps = st.text_input("t steps",f"{t_steps}")

    # Parameters conversion
    S,K,T,r,q,sigma,constant,jump,jump_rate,theta,zeta,kappa,rho,v0 = map(float, [S,K,T,r,q,sigma,constant,jump,jump_rate,theta,zeta,kappa,rho,v0,])
    n_sims,steps,s_steps,t_steps = map(int, [n_sims,steps,s_steps,t_steps])

    # Stability
    S_max = 2000
    dt = T/t_steps
    dS = S_max/s_steps
    factor = (sigma**2*S_max**2*dt)/(dS**2)

    # if (factor <= 0.2) and (scheme == "Forward Euler"):
    #     st.markdown(f"**Discretisation Stable:** "rf"$\frac{{\sigma^2 S_{{\max}}^2 \Delta t}}{{\Delta S^2}} <= 0.20$ "f"({factor:.2f})")
    # elif (factor > 0.2) and (scheme == "Forward Euler"):
    #     st.markdown(f"**Discretisation Unstable:** "rf"$\frac{{\sigma^2 S_{{\max}}^2 \Delta t}}{{\Delta S^2}} > 0.20$ "f"({factor:.2f})")

    # Monte Carlo
    st.write("Monte Carlo")

    col1,col2 = st.columns(2)

    with col1:
        n_sims = st.text_input("Paths",f"{n_sims}")

    with col2:
        steps = st.text_input("Steps",f"{steps}")

    if st.button("Compute"):
        run = True
        S,K,T,r,q,sigma,constant,jump,jump_rate,theta,zeta,kappa,rho,v0 = map(float, [S,K,T,r,q,sigma,constant,jump,jump_rate,theta,zeta,kappa,rho,v0,])
        n_sims,steps,s_steps,t_steps = map(int, [n_sims,steps,s_steps,t_steps])

        if framework == "Black-Scholes (GBM)":
            st.session_state.call_price, st.session_state.put_price = compute_black_scholes_price(S,K,T,r,sigma)
            st.session_state.call_price_monte, st.session_state.put_price_monte, convergence = monte_carlo(S,K,T,r,q,sigma,n_sims,steps,constant,jump,jump_rate,theta,zeta,kappa,rho,v0,motion="GBM")
            st.session_state.call_price_forward_euler, st.session_state.put_price_forward_euler = forward_euler(S,K,T,r,constant,s_steps,t_steps,motion="GBM")
            st.session_state.call_price_backward_euler, st.session_state.put_price_backward_euler = backward_euler(S,K,T,r,constant,s_steps,t_steps,motion="GBM")
            st.session_state.call_price_crank_nicolson, st.session_state.put_price_crank_nicolson = crank_nicolson(S,K,T,r,constant,s_steps,t_steps,motion="GBM")

        elif framework == "Constant of Elasticity Variance (CEV)":
            st.session_state.call_price, st.session_state.put_price = compute_cev_price(S,K,T,r,sigma,constant)
            st.session_state.call_price_monte, st.session_state.put_price_monte, convergence = monte_carlo(S,K,T,r,q,sigma,n_sims,steps,constant,jump,jump_rate,theta,zeta,kappa,rho,v0,motion="CEV")
            st.session_state.call_price_forward_euler, st.session_state.put_price_forward_euler = forward_euler(S,K,T,r,constant,s_steps,t_steps,motion="CEV")
            st.session_state.call_price_backward_euler, st.session_state.put_price_backward_euler = backward_euler(S,K,T,r,constant,s_steps,t_steps,motion="CEV")
            st.session_state.call_price_crank_nicolson, st.session_state.put_price_crank_nicolson = crank_nicolson(S,K,T,r,constant,s_steps,t_steps,motion="CEV")

        elif framework == "Heston":
            st.session_state.call_price_monte, st.session_state.put_price_monte, convergence = monte_carlo(S,K,T,r,q,sigma,n_sims,steps,constant,jump,jump_rate,theta,zeta,kappa,rho,v0,motion="Heston")
        
        elif framework == "Jump Diffusion":
            st.session_state.call_price_monte, st.session_state.put_price_monte, convergence = monte_carlo(S,K,T,r,q,sigma,n_sims,steps,constant,jump,jump_rate,theta,zeta,kappa,rho,v0,motion="MJD")

st.subheader(f"Pricing PDE Solver: {framework}",text_alignment="left")
st.write(f"European {option_type} - FD ({scheme}, {s_steps}x{t_steps}) - MC ({int(n_sims):,} paths)")

with st.container(border=True):
    st.write(f"Governing Equation and Conditions")

    if framework == "Black-Scholes (GBM)":
        st.latex(r"""
            dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
            """)

        st.latex(r"""\frac{\partial V}{\partial t}
            + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}
            + rS\frac{\partial V}{\partial S}
            - rV = 0
            """)
    elif framework == "Constant of Elasticity Variance (CEV)":
        st.latex(r"""
            dS_t
            =
            rS_t\,dt
            +
            \sigma S_t^\lambda\,dW_t
            """)

        st.latex(r"""
            \frac{\partial V}{\partial t}
            +\frac{1}{2}\sigma^2S^{2\lambda}
            \frac{\partial^2V}{\partial S^2}
            +rS\frac{\partial V}{\partial S}-rV=0
            """)
    elif framework == "Heston":
        st.latex(r"""
            dS_t = rS_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{(1)}
            """)

        st.latex(r"""
            dv_t = \kappa(\theta-v_t)\,dt
            +\xi\sqrt{v_t}\,dW_t^{(2)}
            """)
        st.latex(r"""
            dW_t^{(1)}dW_t^{(2)}=\rho\,dt
            """)

        st.latex(r"""
            \frac{\partial V}{\partial t}
            +\frac{1}{2}vS^2\frac{\partial^2V}{\partial S^2}
            +\rho\xi vS\frac{\partial^2V}{\partial S\partial v}
            +\frac{1}{2}\xi^2v\frac{\partial^2V}{\partial v^2}
            +rS\frac{\partial V}{\partial S}
            +\kappa(\theta-v)\frac{\partial V}{\partial v}
            -rV=0
            """)
    elif framework == "Jump Diffusion":
        st.latex(r"""dS_t=
            (r-\lambda k)S_t\,dt
            +\sigma S_t\,dW_t
            +(J-1)S_t\,dN_t
            """)
        
        st.latex(r"""
            \frac{\partial V}{\partial t}
            +\frac12\sigma^2S^2\frac{\partial^2V}{\partial S^2}
            +(r-\lambda k)S\frac{\partial V}{\partial S}
            -rV
            +\lambda
            \int_0^\infty
            \left[
            V(Sy,t)-V(S,t)
            \right]
            f(y)\,dy
            =0
            """)

        st.markdown(r"""
            ### Black–Scholes PDE

            The Black–Scholes PDE is

            $$
            \frac{\partial V}{\partial t}
            +
            \frac{1}{2}\sigma^2 S^2
            \frac{\partial^2 V}{\partial S^2}
            +
            rS\frac{\partial V}{\partial S}
            -rV=0
            $$

            For a **European call option**, the terminal condition at maturity
            $T$ is

            $$
            \boxed{
            V(S,T)=\max(S-K,0)
            }
            $$

            or equivalently,

            $$
            V(S,T)=(S-K)^+
            $$

            ### Boundary Conditions

            At $S=0$, the value of a European call is zero:

            $$
            \boxed{
            V(0,t)=0
            }
            $$

            At a sufficiently large stock price $S=S_{\max}$, the call behaves approximately like the stock minus the discounted strike:

            $$
            \boxed{
            V(S_{\max},t)
            \approx
            S_{\max}-Ke^{-r(T-t)}
            }
            $$

            where:

            - $S$ = underlying asset price
            - $K$ = strike price
            - $T$ = time to maturity
            - $t$ = current time
            - $r$ = risk-free interest rate
            - $\sigma$ = volatility
            - $V(S,t)$ = option value
            - $S_{\max}$ = maximum stock price in the finite-difference grid
            """)

col1, col2, col3, col4 = st.columns(4)

if scheme == "Forward Euler":
    st.session_state.call_price_finite_difference = st.session_state.call_price_forward_euler
    st.session_state.put_price_finite_difference = st.session_state.put_price_forward_euler

if scheme == "Backward Euler":
    st.session_state.call_price_finite_difference = st.session_state.call_price_backward_euler
    st.session_state.put_price_finite_difference = st.session_state.put_price_backward_euler

if scheme == "Crank-Nicolson (2nd Order)":
    st.session_state.call_price_finite_difference = st.session_state.call_price_crank_nicolson
    st.session_state.put_price_finite_difference = st.session_state.put_price_crank_nicolson

if option_type == "Call":
    price = st.session_state.call_price
    price_monte = st.session_state.call_price_monte
    price_finite_difference = st.session_state.call_price_finite_difference

elif option_type == "Put":
    price = st.session_state.put_price
    price_monte = st.session_state.put_price_monte
    price_finite_difference = st.session_state.put_price_finite_difference

col1.metric("Finite Difference", f"{price_finite_difference:.4f}",border=True)
col2.metric("Monte Carlo", f"{price_monte:.4f}",border=True)

if (framework == "Black-Scholes (GBM)") or (framework == "Constant of Elasticity Variance (CEV)"):
    col3.metric("Analytical Solution", f"{price:.4f}",border=True)

elif (framework == "Heston") or (framework == "Jump Diffusion"):
    col3.metric("No Analytical Solution","N/A",border=True)

diff = price_finite_difference - price_monte

col4.metric("Difference: (FF - MC)", f"{diff:.4f}",border=True)

if run == True:
    with st.container(border=True):
        surface_plot, heatmap_plot = st.tabs(['Surface','Heatmap'])

        S, K, T, r, sigma = map(float, [S, K, T, r, sigma])
        n_sims = int(n_sims)
        Sn = np.linspace(0,250,100)
        Tn = np.linspace(0.01,2,100)

        S_surface,T_surface = np.meshgrid(Sn,Tn)

        surface_call = np.zeros_like(S_surface)

        surface_put = np.zeros_like(S_surface)

        if framework == "Black-Scholes (GBM)":
            for i in range(S_surface.shape[0]):
                for j in range(S_surface.shape[1]):
                    call, put = compute_black_scholes_price(S_surface[i, j],K,T_surface[i, j],r,sigma)
                    surface_call[i, j] = call
                    surface_put[i, j] = put

        if framework == "Constant of Elasticity Variance (CEV)":
            for i in range(S_surface.shape[0]):
                for j in range(S_surface.shape[1]):
                    call, put = compute_cev_price(S_surface[i, j],K,T_surface[i, j],r,sigma,constant)
                    surface_call[i, j] = call
                    surface_put[i, j] = put

        if option_type == "Call":
            surface = surface_call
        elif option_type == "Put":
            surface = surface_put

        with surface_plot:
            fig = go.Figure(data=[go.Surface(x=Sn,y=Tn,z=surface,colorscale="Viridis")])

            fig.update_layout(
                template="plotly_dark",
                scene=dict(
                    xaxis_title="Underlying Price (S)",
                    yaxis_title="Time to Maturity (T)",
                    zaxis_title="Option Price",
                )
            )

            fig.update_layout(title=dict(text='Pricing Surface V(S,t)'), autosize=False,width=350, height=700)

            st.plotly_chart(
                fig,
                width="stretch",
                config={"scrollZoom":False}
            )

        with heatmap_plot:
            fig = go.Figure(data=[go.Heatmap(x=Sn,y=Tn,z=surface,colorscale="Viridis")])

            fig.update_layout(
                template="plotly_dark",
                xaxis=dict(
                    title="Underlying Price (S)"
                ),
                yaxis=dict(
                    title="Time to Maturity (T)"
                ),
                title=dict(
                    text="Pricing Heatmap V(S,t)"
                ),
                autosize=False,
                width=350,
                height=700
            )

            fig.update_layout(title=dict(text='Pricing Heatmap V(S,t)'), autosize=False,width=350, height=700)

            st.plotly_chart(
                fig,
                width="stretch",
                config={"scrollZoom":False}
            )

    with st.container(border=True):
        col1,col2 = st.columns(2)

        with col1:
            payoffs = compute_payoff(Sn,K,option=option_type)
            T_fixed = 0
            idx = np.argmin(np.abs(Tn - T_fixed)) # Find closest to chosen T

            price_slice = surface[idx,:]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(x=Sn, y=payoffs, name="Payoff ($)", mode="lines",
                    line=dict(color="#f87171", width=2.5, dash="dot")
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=Sn, 
                    y=price_slice, 
                    name=f"Pricing Function", 
                    mode="lines",
                    line=dict(color="#4ade80", width=2.5),
                    fill="tozeroy", 
                    fillcolor="rgba(74,222,128,0.08)"
                )
            )

            fig.update_yaxes(tickprefix="$")

            fig.update_layout(
                title="Payoff Diagram",
                xaxis_title="Spot Price ($)",
                yaxis_title="Payoff ($)",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                width="stretch",
                config={"scrollZoom":False}
            )

        with col2:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(x=convergence.index, y=convergence["Mean"], name="Mean", mode="lines",
                    line=dict(color="#f87171", width=2.5, dash="dot")
                )
            )

            fig.add_trace(
                go.Scatter(x=convergence.index, y=convergence["Standard Deviation"], name="Standard Deviation", mode="lines",
                    line=dict(color="#4ade80", width=2.5, dash="dot")
                )
            )

            fig.add_trace(
                go.Scatter(x=convergence.index, y=convergence["Confidence Width"], name="Confidence Interval Width", mode="lines",
                    line=dict(color="#60a5fa", width=2.5, dash="dot")
                )
            )

            fig.update_yaxes(tickprefix="$")

            fig.update_layout(
                title=f"Monte Carlo Convergence - 95% CI [{convergence["Lower Value"].iloc[-1]:.2f}, {convergence["Upper Value"].iloc[-1]:.2f}]",
                xaxis_title="Iteration",
                yaxis_title="Value",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                width="stretch",
                config={"scrollZoom":False}
            )

    with st.container(border=True):
        pricing = pd.DataFrame({'Prices':[price,price_monte,price_finite_difference]},index=['Analytical','Monte Carlo','Finite Difference'])

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=pricing.index,
                y=pricing['Prices'],
                showlegend=False,
                marker_color=["#636EFA", "#00CC96", "#EF553B"],
            )
        )

        fig.update_yaxes(tickprefix="$")

        fig.update_layout(
            title="Analytical vs FF vs MC Price",
            yaxis_title="Price ($)",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={"scrollZoom":False}
        )