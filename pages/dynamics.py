import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, t, lognorm, kurtosis, skew
from skewt_scipy.skewt import skewt

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def compute_returns(df: pd.Series):
    returns = df.pct_change().dropna()
    return returns

def compute_r_squared(data, simulated):
    x = data.values
    y = simulated.values
    ss_res = np.sum((x - y)**2)
    ss_tot = np.sum((x - np.mean(x))**2)
    r_squared = 1 - ss_res / ss_tot
    return r_squared

# Arithmetic Brownian Motion
def ABM(mu, sigma, S0, n_steps):
    dt = 1
    t = np.arange(dt,n_steps+1)
    S = np.zeros(len(t))
    S[0] = S0
    # Euler-Maruyama Discretisation (same as closed form)
    for i in range(1,len(S)):
        S[i] = S[i-1] + mu*dt + sigma*np.sqrt(dt)*np.random.normal(0,1)
    return S

# Geometric Brownian Motion - degenerate solution as S0 approaches 0 or sigma = 0
def GBM(mu, sigma, S0, n_steps):
    dt = 1
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), n_steps)) 
    t = np.arange(dt,n_steps+1)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W) 
    return S

def OU(mu, sigma, theta, S0, n_steps):
    dt = 1
    t = np.arange(dt,n_steps+1)
    X = np.zeros(len(t))
    X[0] = S0
    a = np.exp(-theta*dt)
    b = np.sqrt((sigma**2/(2*theta))*(1-np.exp(-2*theta*dt)))
    for i in range(1,len(t)):
        X[i] = mu + a*(X[i-1] - mu) + b*np.random.normal(0,1)
    return X

# Merton Jump Diffusion Model
def MJD(mu, sigma, jump, jump_rate, S0, n_steps):
    T = 1
    dt = T / n_steps
    t = np.linspace(dt,T,n_steps)
    S = np.zeros(len(t))
    S[0] = S0
    # Euler-Maruyama Discretisation
    for i in range(1,len(t)):
        S[i] = S[i-1] + mu*dt*S[i-1] + sigma*S[i-1]*np.sqrt(dt)*np.random.normal(0,1) + S[i-1]*(jump-1)*np.random.poisson(jump_rate*dt)
    return S

# Heston Stochastic Volatility Model
def Heston(mu, theta, zeta, kappa, rho, v0, S0, n_steps):
    T = 1
    dt = T/ n_steps
    t = np.linspace(dt,T,n_steps)
    S = np.zeros(len(t))
    v = np.zeros(len(t))
    v[0] = v0
    S[0] = S0
    # Euler-Maruyama Discretisation
    for i in range(1,len(t)):
        Z1 = np.random.normal(0,1)
        Z2 = np.random.normal(0,1)
        Zv = Z1
        Zs = rho*Z1 + np.sqrt(1-rho**2)*Z2
        v[i] = v[i-1] + kappa*(theta - v[i-1])*dt + zeta*np.sqrt(v[i-1])*np.sqrt(dt)*Zv
        S[i] = S[i-1] + mu*S[i-1]*dt + np.sqrt(v[i-1])*S[i-1]*np.sqrt(dt)*Zs
    return S

# Historical Calibration
def calibrate_arithmetic_brownian_motion(data: pd.Series):
    absolute_difference = data.diff().dropna()
    mu = absolute_difference.mean()
    sigma = absolute_difference.std()
    return mu, sigma

def calibrate_geometric_brownian_motion(data: pd.Series):
    dt = 1
    log_returns = np.log(data / data.shift(1)).dropna()
    sigma = log_returns.std() / np.sqrt(dt)
    mu = log_returns.mean() / dt + 0.5 * sigma**2
    return mu, sigma

def calibrate_ornstein_uhlenbeck_process(data: pd.Series):
    # Calibration of OU Process
    df = pd.DataFrame({'delta': data.diff(),'lag_close': data.shift(1)}).dropna()
    X = sm.add_constant(df['lag_close'])
    y = df['delta']

    model = sm.OLS(y, X).fit()

    alpha = model.params['const']
    beta = model.params['lag_close']
    var_eps = np.var(model.resid)

    dt = 1
    phi = 1 + beta
    theta = -np.log(phi)/dt
    mu = alpha/(1-phi)
    sigma = np.sqrt((2*theta*var_eps)/(1-np.exp(-2*theta*dt)))

    r_squared = model.rsquared

    return mu, theta, sigma, var_eps, r_squared

def calibrate_merton_jump_diffusion(data: pd.Series):
    pass

def calibrate_heston(data: pd.Series):
    pass

def fit_gaussian(data):
    mu, sigma = norm.fit(data)
    x = np.linspace(data.min(), data.max(), 500)
    y = norm.pdf(x, mu, sigma)
    return x, y

def fit_t_distribution(data):
    df_t, loc_t, scale_t = t.fit(data)
    x = np.linspace(data.min(), data.max(), 500)
    y = t.pdf(x,df=df_t,loc=loc_t,scale=scale_t)
    return x, y

def H3(x):
    h3 = x**3 - 3*x
    return h3

def H4(x):
    h4 = x**4 -6*x**2 + 3
    return h4

def H6(x):
    h6 = x**6 - 15*x**4 + 45*x**2 - 15
    return h6

# Type A approximation using H3,H4 and H6 - effects of kurtosis, skew and second-order effect of skew
def fit_gram_charlier_approximation(data): 
    mu, sigma = norm.fit(data)
    kurtosis_value = kurtosis(data)
    skew_value = skew(data)
    x = np.linspace(data.min(), data.max(), 500)
    z = (x - mu)/ sigma
    pdf = (norm.pdf(z)/sigma) * (1 + (skew_value * H3(z))/6 + (kurtosis_value * H4(z))/24 + (skew_value**2*H6(z)/72))
    y = pdf
    return x, y

def fit_skewed_t_distribution(data):
    mu, sigma, skew_value, kurtosis_value = skewt.fit(data)
    x = np.linspace(data.min(), data.max(), 500)
    y = skewt.pdf(x,mu,sigma,skew_value,kurtosis_value)
    return x, y

def fit_lognormal(data):
    loc, scale, size = lognorm.fit(data)
    x = np.linspace(data.min(), data.max(), 500)
    y = lognorm.pdf(x,loc,scale,size)
    return x, y

run = False
run_dist = False

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Market Modelling",layout="wide")

st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("Match Market Data To Chosen Model")

    st.text("Input metrics for data:",text_alignment='center')
    st.markdown("* Ticker e.g AAPL.",text_alignment='left')
    st.markdown("* Start date (date in Y-m-d).",text_alignment='left')
    st.markdown("* End date (date in Y-m-d).",text_alignment='left')
    st.markdown("* Interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo). *Note intraday data not available for period greater than 60 days).",text_alignment='left')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ticker_input = st.text_input("Ticker")

    with col2:
        start_input = st.text_input("Start Date")

    with col3:
        end_input = st.text_input("End Date")

    with col4:
        interval_input = st.text_input("Interval")

with st.sidebar:
    st.subheader("Data",text_alignment="center")

    data_type = st.selectbox("Data Type",("Open","Low",
    "High","Close","Volume"))

    selection = st.selectbox("Raw Price or Returns",("Raw","Returns"))

    st.subheader("Models",text_alignment="center")

    model = st.selectbox("Select Model",("Arithmetic Brownian Motion","Geometric Brownian Motion",
    "Ornstein-Uhlenbeck Process","Merton Jump Diffusion","Heston"))

    if st.button("Compute Dynamics"):
        run = True
        ticker = ticker_input.upper()
        start = dt.datetime.strptime(start_input, "%Y-%m-%d")
        end = dt.datetime.strptime(end_input, "%Y-%m-%d")
        interval = interval_input

        df = yf.download(tickers=ticker,start=start,end=end,interval=interval)

        data = df[data_type].squeeze()

        if selection == "Returns":
            data = compute_returns(data)

    st.subheader("Gaussians",text_alignment="center")

    selection2 = st.selectbox("Raw/Returns",("Raw","Returns"))

    options = ["Gaussian", "t-Distribution", "Skewed t-Distribution","Gram-Charlier Approximation","Lognormal Distribution"]
    gaussians = st.pills("Select Distribution", options, selection_mode="multi")

    if st.button("Compute Distributions"):
        run_dist= True
        ticker = ticker_input.upper()
        start = dt.datetime.strptime(start_input, "%Y-%m-%d")
        end = dt.datetime.strptime(end_input, "%Y-%m-%d")
        interval = interval_input

        df = yf.download(tickers=ticker,start=start,end=end,interval=interval)

        data = df[data_type].squeeze()

        if selection2 == "Returns":
            data = compute_returns(data)

with st.container(border=True):
    st.subheader("Data")

    if run == True:
        future_steps = 252
        S0_future = data.iloc[-1]
        future_index = [i for i in range(1,252+1)]

        if model == "Arithmetic Brownian Motion":
            mu, sigma = calibrate_arithmetic_brownian_motion(data)
            S0 = data.iloc[0]
            n_steps = data.shape[0]
            simulated = ABM(mu, sigma, S0, n_steps)
            simulated = pd.DataFrame({"Simulated": simulated},index = data.index)
            r_squared = compute_r_squared(data,simulated)
            future_paths = ABM(mu,sigma,S0_future,future_steps)
            future_paths = pd.DataFrame({"Future Paths": future_paths},index = future_index)

        if model == "Geometric Brownian Motion":
            mu, sigma = calibrate_geometric_brownian_motion(data)
            S0 = data.iloc[0]
            n_steps = data.shape[0]
            simulated = GBM(mu, sigma, S0, n_steps)
            simulated = pd.DataFrame({"Simulated": simulated},index = data.index)
            r_squared = compute_r_squared(data,simulated)
            future_paths = GBM(mu,sigma,S0_future,future_steps)
            future_paths = pd.DataFrame({"Future Paths": future_paths},index = future_index)

        if model == "Ornstein-Uhlenbeck Process":
            mu, theta, sigma, var_eps, r_squared = calibrate_ornstein_uhlenbeck_process(data)
            S0 = data.iloc[0]
            n_steps = data.shape[0]
            simulated = OU(mu, sigma, theta, S0, n_steps)
            simulated = pd.DataFrame({"Simulated": simulated},index = data.index)
            r = compute_r_squared(data,simulated)
            future_paths = OU(mu,sigma,theta,S0_future,future_steps)
            future_paths = pd.DataFrame({"Future Paths": future_paths},index = future_index)
        
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data.values, 
                name=f"{ticker}", 
                mode="lines",
                line=dict(color="#4ade80", width=2.2),
                fill="tozeroy", 
                fillcolor="rgba(74,222,128,0.08)"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=simulated.index, 
                y=simulated["Simulated"], 
                name=f"Tuned {model}, R² = {r_squared:.2f}", 
                mode="lines",
                line=dict(color="#f87171", width=2.2),
                fill="tozeroy", 
                fillcolor="rgba(74,222,128,0.08)"
            )
        )

        fig.update_yaxes(ticksuffix="%")

        fig.update_layout(
            title=f"{ticker} {selection} {data_type} Price ($)",
            xaxis_title="Date",
            yaxis_title=f"{data_type} Price ($)",
            template="plotly_dark"
        )

        st.plotly_chart(fig, width="stretch", config = {'scrollZoom': False})

with st.container(border=True):
    if run == True:
        n_sims = 10000
        finals = []
        st.subheader("Future Paths")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=future_paths.index, 
                y=future_paths["Future Paths"],  
                mode="lines",
                line=dict(color="#4ade80", width=2.2),
                fill="tozeroy", 
                fillcolor="rgba(74,222,128,0.08)"
            )
        )

        path_colours = px.colors.qualitative.Set2

        for i in range(n_sims):
            if model == "Arithmetic Brownian Motion":
                future_paths = ABM(mu,sigma,S0_future,future_steps)
            if model == "Geometric Brownian Motion":
                future_paths = GBM(mu,sigma,S0_future,future_steps)
            if model == "Ornstein-Uhlenbeck Process":
                future_paths = OU(mu,sigma,theta,S0_future,future_steps)
            if model == "Merton Jump Diffusion":
                pass
            if model == "Heston":
                pass
            
            finals.append(future_paths[-1])

            future_paths = pd.DataFrame({"Future Paths": future_paths},index = future_index)

            colour = path_colours[i % len(path_colours)]

            if i <= 100:
                fig.add_trace(
                    go.Scatter(
                        x=future_paths.index,
                        y=future_paths["Future Paths"],
                        mode="lines",
                        line=dict(
                            color=colour,
                            width=1,
                        ),
                        opacity=0.05,
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )

        expectation = np.mean(finals)

        fig.update_yaxes(ticksuffix="$")

        st.write(rf"Possible Future Paths: $E[S_{{\mathrm{{final}}}}] \approx$ ${expectation:.2f}")

        fig.update_layout(
            title = f"{model}: Possible Future Paths — {n_sims:,} calculated, 100 shown, 1 highlighted",
            xaxis_title="Day",
            yaxis_title=f"Price ($)",
            template="plotly_dark"
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, width="stretch", config = {'scrollZoom': False})

if (gaussians) and (run_dist==True):
    with st.container(border=True):
        x, y = fit_gaussian(data)
        xt, yt = fit_t_distribution(data)
        xgc, ygc = fit_gram_charlier_approximation(data)
        xst, yst = fit_skewed_t_distribution(data)
        xln, yln = fit_lognormal(data)

        st.subheader("Gaussian Fit")

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=data,
                histnorm="probability density",
                name="Observed Data",
                opacity=0.6,
                marker=dict(color="#60a5fa")
            )
        )

        if "Gaussian" in gaussians:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name="Gaussian Fit",
                    line=dict(color="#4ade80", width=2.2)
                )
            )

        if "t-Distribution" in gaussians:
            fig.add_trace(
                go.Scatter(
                    x=xt,
                    y=yt,
                    mode="lines",
                    name="t-Distribution",
                    line=dict(color="#f87171", width=2.2)
                )
            )

        if "Gram-Charlier Approximation" in gaussians:
            fig.add_trace(
                go.Scatter(
                    x=xgc,
                    y=ygc,
                    mode="lines",
                    name="Gram-Charlier Approximation",
                    line=dict(color="#ef4444", width=2.2)
                )
            )

        if "Skewed t-Distribution" in gaussians:
            fig.add_trace(
                go.Scatter(
                    x=xst,
                    y=yst,
                    mode="lines",
                    name="Skewed t-Distribution",
                    line=dict(color="#a855f7", width=2.2)
                )
            )

        if "Lognormal Distribution" in gaussians:
            fig.add_trace(
                go.Scatter(
                    x=xln,
                    y=yln,
                    mode="lines",
                    name="Lognormal Distribution",
                    line=dict(color="#2563EB", width=2.2)
                )
            )

        fig.update_layout(
            title=f"Distribution Fit — {ticker}",
            xaxis_title="Return",
            yaxis_title="Probability Density",
            template="plotly_dark"
        )

        st.plotly_chart(fig, width="stretch", config = {'scrollZoom': False})