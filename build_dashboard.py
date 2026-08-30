"""Build the static, interactive cattle-futures dashboard in docs/index.html.

The browser receives a published data snapshot. Python owns acquisition,
calculation, chart construction, and HTML generation; a small inline script
only coordinates read-only controls that a static GitHub Pages site needs.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "docs"
OUTPUT_FILE = OUTPUT_DIR / "index.html"
LIVE_FEED_FILE = OUTPUT_DIR / "live-market.json"

COLORS = {
    "blue": "#31688E",
    "blue_light": "#9CCAE3",
    "orange": "#E07A3F",
    "gold": "#D6A84B",
    "olive": "#738B3B",
    "pink": "#C66A8B",
    "ink": "#252A34",
    "muted": "#6F7782",
    "grid": "#D9DCE1",
}
MARKET_NAMES = {"GF=F": "Feeder cattle", "LE=F": "Live cattle", "ZC=F": "Corn"}
SOURCE_URLS = {
    "yahoo": "https://finance.yahoo.com/quote/GF=F/",
    "usda": "https://esmis.nal.usda.gov/publication/cattle-feed",
    "cftc": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
    "drought": "https://droughtmonitor.unl.edu/DmData/DataDownload.aspx",
    "cme": "https://www.cmegroup.com/markets/agriculture/livestock/feeder-cattle.html",
}
PLOT_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
RANGE_PRESETS = (
    ("1d", "1 day"),
    ("5d", "5 days"),
    ("1m", "1 month"),
    ("6m", "6 months"),
    ("1y", "1 year"),
    ("5y", "5 years"),
    ("all", "All history"),
)
RANGE_CHART_MIN_DAYS = {
    # These charts contain daily observations, so every preset is useful.
    "price-regime": 1,
    "market-drivers": 1,
    # The remaining charts intentionally retain their native observation
    # cadence instead of becoming empty for a 1D or 5D selection.
    "drought": 7,
    "positioning": 7,
    "risk": 7,
    "usda-history": 28,
}


def yahoo_closes(tickers: list[str] | str, **kwargs) -> pd.DataFrame:
    """Return clean Yahoo closing-price columns with a timezone-naive index."""
    raw = yf.download(
        tickers=tickers,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
        timeout=30,
        **kwargs,
    )
    if raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {tickers}")
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        name = tickers if isinstance(tickers, str) else tickers[0]
        closes = closes.rename(name).to_frame()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes.sort_index()


def find_usda_text_releases(pages: int = 4) -> list[str]:
    urls: list[str] = []
    for page in range(pages):
        response = requests.get(SOURCE_URLS["usda"], params={"page": page}, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        urls.extend(urljoin(SOURCE_URLS["usda"], link["href"]) for link in soup.select('a[href$=".txt"]'))
    return list(dict.fromkeys(urls))


def parse_usda_release(url: str) -> dict | None:
    text = requests.get(url, timeout=30).text
    release_match = re.search(r"Released\s+([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    headings = [
        match.start()
        for match in re.finditer(
            "Cattle on Feed Inventory, Placements, Marketings, and Other Disappearance on", text
        )
    ]
    if not release_match or len(headings) < 2:
        return None
    first_table = text[headings[0] : headings[1]]
    patterns = {
        "Inventory": r"On feed [A-Z][a-z]+ 1\s+\.*:\s*([\d,]+)\s+([\d,]+)\s+(\d+)",
        "Placements": r"Placed on feed during [A-Z][a-z]+\s+\.*:\s*([\d,]+)\s+([\d,]+)\s+(\d+)",
        "Marketings": r"Fed cattle marketed during [A-Z][a-z]+\s+\.*:\s*([\d,]+)\s+([\d,]+)\s+(\d+)",
    }
    row: dict = {"Release date": pd.to_datetime(release_match.group(1)), "Source": url}
    for metric, pattern in patterns.items():
        matches = re.findall(pattern, first_table)
        if not matches:
            return None
        selected = matches[-1] if metric == "Inventory" else matches[0]
        row[metric] = int(selected[1].replace(",", ""))
        row[f"{metric} YoY"] = int(selected[2]) - 100
    return row


def load_usda() -> tuple[pd.DataFrame, str]:
    try:
        urls = find_usda_text_releases()[:40]
        with ThreadPoolExecutor(max_workers=8) as executor:
            parsed = list(executor.map(parse_usda_release, urls))
        frame = (
            pd.DataFrame([row for row in parsed if row is not None])
            .drop_duplicates("Release date")
            .sort_values("Release date")
            .reset_index(drop=True)
        )
        if frame.empty:
            raise RuntimeError("no parsable text releases found")
        return frame, f"{len(frame)} releases through {frame['Release date'].max():%b %d, %Y}"
    except Exception as exc:  # Optional contextual source.
        return pd.DataFrame(), f"Unavailable: {exc}"


CATTLE_STATE_FIPS = {
    "Colorado": "08",
    "Kansas": "20",
    "Nebraska": "31",
    "Oklahoma": "40",
    "South Dakota": "46",
    "Texas": "48",
}
DROUGHT_API = "https://usdmdataservices.unl.edu/api/StateStatistics/GetDroughtSeverityStatisticsByAreaPercent"


def download_state_drought(item: tuple[str, str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    state, fips = item
    response = requests.get(
        DROUGHT_API,
        params={
            "aoi": fips,
            "startdate": f"{start.month}/{start.day}/{start.year}",
            "enddate": f"{end.month}/{end.day}/{end.year}",
            "statisticsType": 1,
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    frame = pd.DataFrame(response.json())
    raw_dates = frame["mapDate"].astype(str).str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(frame["mapDate"], errors="coerce")
    frame["Date"] = parsed
    frame["State"] = state
    return frame[["Date", "State", "d2", "d3", "d4"]].dropna(subset=["Date"])


def load_drought(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series, str]:
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            parts = list(executor.map(lambda item: download_state_drought(item, start, end), CATTLE_STATE_FIPS.items()))
        by_state = pd.concat(parts, ignore_index=True)
        by_state[["d2", "d3", "d4"]] = by_state[["d2", "d3", "d4"]].apply(pd.to_numeric)
        index = by_state.groupby("Date")["d2"].mean().sort_index()
        return by_state, index, f"Six-state D2+ coverage through {index.index.max():%b %d, %Y}"
    except Exception as exc:
        return pd.DataFrame(), pd.Series(dtype=float), f"Unavailable: {exc}"


def load_cftc(start: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    api = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
    params = {
        "$select": "report_date_as_yyyy_mm_dd,m_money_positions_long_all,m_money_positions_short_all,open_interest_all",
        "$where": f"cftc_contract_market_code='061641' AND report_date_as_yyyy_mm_dd >= '{start:%Y-%m-%d}T00:00:00.000'",
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": 5000,
    }
    try:
        response = requests.get(api, params=params, timeout=30)
        response.raise_for_status()
        frame = pd.DataFrame(response.json())
        frame["Date"] = pd.to_datetime(frame["report_date_as_yyyy_mm_dd"])
        numeric = ["m_money_positions_long_all", "m_money_positions_short_all", "open_interest_all"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric)
        frame["Managed money net"] = frame["m_money_positions_long_all"] - frame["m_money_positions_short_all"]
        frame["Managed money net % of OI"] = frame["Managed money net"] / frame["open_interest_all"]
        return frame, f"{len(frame)} weekly reports through {frame['Date'].max():%b %d, %Y}"
    except Exception as exc:
        return pd.DataFrame(), f"Unavailable: {exc}"


FEEDER_MONTH_CODES = {1: "F", 3: "H", 4: "J", 5: "K", 8: "Q", 9: "U", 10: "V", 11: "X"}


def feeder_contract_candidates(reference_date: pd.Timestamp, count: int = 8) -> list[tuple[str, pd.Timestamp]]:
    reference_period = reference_date.to_period("M")
    contracts: list[tuple[str, pd.Timestamp]] = []
    for period in pd.period_range(reference_period + 1, periods=24, freq="M"):
        if period.month in FEEDER_MONTH_CODES:
            symbol = f"GF{FEEDER_MONTH_CODES[period.month]}{str(period.year)[-2:]}.CME"
            contracts.append((symbol, period.to_timestamp()))
        if len(contracts) == count:
            break
    return contracts


def latest_contract_close(contract: tuple[str, pd.Timestamp]) -> dict | None:
    symbol, month = contract
    try:
        series = yahoo_closes(symbol, period="1mo", interval="1d").iloc[:, 0].dropna()
        if series.empty:
            return None
        return {"Symbol": symbol, "Contract month": month, "Latest date": series.index[-1], "Latest close": float(series.iloc[-1])}
    except Exception:
        return None


def load_curve(reference_date: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            rows = list(executor.map(latest_contract_close, feeder_contract_candidates(reference_date)))
        frame = pd.DataFrame([row for row in rows if row is not None]).sort_values("Contract month")
        if len(frame) < 2:
            raise RuntimeError("fewer than two listed contracts returned quotes")
        front = frame["Latest close"].iloc[0]
        frame["Difference from nearby"] = frame["Latest close"] - front
        frame["Approx. dollars vs nearby"] = frame["Difference from nearby"] * 500
        return frame, f"{len(frame)} listed contracts; closes through {frame['Latest date'].max():%b %d, %Y}"
    except Exception as exc:
        return pd.DataFrame(), f"Unavailable: {exc}"


def event_returns(price_series: pd.Series, event_dates: pd.Series) -> pd.DataFrame:
    prices = price_series.dropna().sort_index()
    rows = []
    for value in sorted(pd.to_datetime(event_dates.dropna()).unique()):
        event_date = pd.Timestamp(value)
        before, after = prices.loc[:event_date], prices.loc[prices.index > event_date]
        if before.empty or len(after) < 5:
            continue
        event_close = before.iloc[-1]
        rows.append(
            {
                "Release date": event_date,
                "Next-session return": after.iloc[0] / event_close - 1,
                "Five-session return": after.iloc[4] / event_close - 1,
            }
        )
    return pd.DataFrame(rows)


def drawdown_episodes(series: pd.Series) -> pd.DataFrame:
    underwater = series < 0
    groups = (underwater != underwater.shift()).cumsum()
    rows = []
    for _, segment in series[underwater].groupby(groups[underwater]):
        trough_date = segment.idxmin()
        prior = series.loc[: segment.index[0]].iloc[:-1]
        peak_date = prior.index[-1] if not prior.empty else segment.index[0]
        after = series.loc[segment.index[-1] :]
        recovered = after[after >= 0]
        recovery_date = recovered.index[0] if not recovered.empty else pd.NaT
        end_date = recovery_date if pd.notna(recovery_date) else segment.index[-1]
        rows.append(
            {
                "Peak": peak_date,
                "Trough": trough_date,
                "Recovery": recovery_date,
                "Max drawdown": segment.min(),
                "Weeks underwater": round((end_date - peak_date).days / 7),
                "Status": "Recovered" if pd.notna(recovery_date) else "Ongoing",
            }
        )
    return pd.DataFrame(rows).sort_values("Max drawdown").reset_index(drop=True)


def pct(value: float, decimals: int = 1, signed: bool = False) -> str:
    if pd.isna(value):
        return "—"
    sign = "+" if signed else ""
    return f"{value:{sign}.{decimals}%}"


def number(value: float, decimals: int = 1, signed: bool = False) -> str:
    if pd.isna(value):
        return "—"
    sign = "+" if signed else ""
    return f"{value:{sign},.{decimals}f}"


def vote_label(vote: float) -> str:
    if pd.isna(vote):
        return "Unavailable"
    if vote > 0.25:
        return "Bullish"
    if vote < -0.25:
        return "Bearish"
    return "Neutral / mixed"


def signed_vote(value: float, positive: float = 0.0, negative: float = 0.0) -> float:
    if pd.isna(value):
        return np.nan
    if value > positive:
        return 1.0
    if value < negative:
        return -1.0
    return 0.0


def style_figure(fig: go.Figure, title: str, subtitle: str, height: int = 500, hovermode: str | None = None) -> go.Figure:
    fig.update_layout(
        title={
            "text": f"<b>{html.escape(title)}</b><br><span style='font-size:12px;color:#6F7782'>{html.escape(subtitle)}</span>",
            "x": 0.02,
            "xanchor": "left",
        },
        height=height,
        # Keep the subtitle and horizontal legend in distinct rows above the
        # plotting area.  The extra top margin also gives wrapped legends
        # room on smaller screens.
        margin={"l": 64, "r": 28, "t": 112, "b": 52},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, ui-sans-serif, system-ui, sans-serif", "color": COLORS["ink"], "size": 12},
        colorway=[COLORS["blue"], COLORS["orange"], COLORS["olive"], COLORS["gold"], COLORS["pink"]],
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.96, "x": 0},
        hovermode=hovermode,
        uirevision="cattle-dashboard",
    )
    fig.update_xaxes(showgrid=False, linecolor="#AEB4BC", zerolinecolor="#AEB4BC")
    fig.update_yaxes(gridcolor=COLORS["grid"], linecolor="#AEB4BC", zerolinecolor="#89919B")
    return fig


def chart_html(fig: go.Figure, chart_id: str, panel_count: int = 1) -> str:
    chart = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=PLOT_CONFIG, div_id=chart_id)
    # Keep our responsive hook independent of Plotly's version-specific
    # inline wrapper markup.  Multi-panel figures need more vertical room on
    # phones; otherwise Plotly compresses each subplot into a thin strip.
    return f'<div class="chart-wrap" data-panel-count="{panel_count}">{chart}</div>'


def source_note(label: str, source_key: str, detail: str) -> str:
    return (
        f'<div class="source-note"><span>Source</span> '
        f'<a href="{SOURCE_URLS[source_key]}" target="_blank" rel="noopener">{html.escape(label)}</a>'
        f' · {html.escape(detail)}</div>'
    )


def table_html(frame: pd.DataFrame, formats: dict[str, callable] | None = None) -> str:
    display = frame.copy()
    for column, formatter in (formats or {}).items():
        if column in display:
            display[column] = display[column].map(formatter)
    table = display.to_html(index=False, border=0, classes="data-table", escape=True, na_rep="—")
    return f'<div class="table-scroll">{table}</div>'


def build() -> dict:
    built_at = datetime.now(timezone.utc)
    daily_prices = yahoo_closes(list(MARKET_NAMES), period="max", interval="1d").rename(columns=MARKET_NAMES)
    daily_feeder = daily_prices["Feeder cattle"].dropna()
    if daily_feeder.empty:
        raise RuntimeError("Core feeder-cattle history has no daily observations")
    weekly_prices = daily_prices.resample("W-FRI").last().dropna(how="all")
    feeder_all = weekly_prices["Feeder cattle"].dropna()
    if len(feeder_all) < 156:
        raise RuntimeError("Core feeder-cattle history has fewer than three years of weekly observations")

    sample_start = feeder_all.index.max() - pd.DateOffset(years=6)
    feeder = feeder_all.loc[feeder_all.index >= sample_start]
    returns = feeder.pct_change(fill_method=None).dropna()
    normalized = feeder / feeder.iloc[0]
    previous_peak = normalized.cummax()
    drawdown = normalized / previous_peak - 1
    elapsed_years = (feeder.index[-1] - feeder.index[0]).days / 365.2425
    annual_return = (feeder.iloc[-1] / feeder.iloc[0]) ** (1 / elapsed_years) - 1
    annual_volatility = returns.std() * math.sqrt(52)

    usda, usda_status = load_usda()
    drought_by_state, drought_index, drought_status = load_drought(sample_start, feeder.index.max())
    cftc, cftc_status = load_cftc(sample_start)
    curve, curve_status = load_curve(feeder.index.max())
    source_status = {
        "Yahoo prices": f"Daily history through {daily_prices['Feeder cattle'].last_valid_index():%b %d, %Y}",
        "USDA Cattle on Feed": usda_status,
        "U.S. Drought Monitor": drought_status,
        "CFTC positioning": cftc_status,
        "Listed futures curve": curve_status,
    }

    # Reusable monthly and seasonal calculations.
    feeder_monthly = daily_prices["Feeder cattle"].dropna().resample("ME").last()
    monthly_returns = feeder_monthly.pct_change(fill_method=None).dropna()
    seasonality = pd.DataFrame(
        {
            "Average return": monthly_returns.groupby(monthly_returns.index.month).mean(),
            "Median return": monthly_returns.groupby(monthly_returns.index.month).median(),
            "Positive share": monthly_returns.groupby(monthly_returns.index.month).apply(lambda s: (s > 0).mean()),
            "Observations": monthly_returns.groupby(monthly_returns.index.month).count(),
        }
    )
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    seasonality.index = month_labels

    # Five-factor recommendation copied from the notebook's transparent rules.
    horizon_weeks = 13
    trend_13 = feeder.pct_change(horizon_weeks).iloc[-1]
    moving_average_26 = feeder.rolling(26).mean().iloc[-1]
    distance_ma = feeder.iloc[-1] / moving_average_26 - 1
    trend_vote = np.mean([signed_vote(trend_13), signed_vote(distance_ma)])

    if not usda.empty:
        latest_usda = usda.iloc[-1]
        inventory_vote = 1.0 if latest_usda["Inventory YoY"] <= -1 else (-1.0 if latest_usda["Inventory YoY"] >= 1 else 0.0)
        placements_vote = 1.0 if latest_usda["Placements YoY"] <= -5 else (-1.0 if latest_usda["Placements YoY"] >= 5 else 0.0)
        supply_vote = np.mean([inventory_vote, placements_vote])
        supply_evidence = f"Inventory {latest_usda['Inventory YoY']:+.0f}% YoY; placements {latest_usda['Placements YoY']:+.0f}% YoY"
    else:
        latest_usda, supply_vote, supply_evidence = None, np.nan, "USDA data unavailable"

    horizon_contract = None
    if len(curve) >= 2:
        horizon_date = feeder.index[-1] + pd.Timedelta(weeks=horizon_weeks)
        horizon_contract = curve.loc[(curve["Contract month"] - horizon_date).abs().idxmin()]
        curve_difference = horizon_contract["Latest close"] / curve["Latest close"].iloc[0] - 1
        curve_vote = signed_vote(curve_difference, 0.02, -0.02)
        curve_evidence = f"{horizon_contract['Contract month']:%b %Y} is {curve_difference:+.1%} vs nearby"
    else:
        curve_difference, curve_vote, curve_evidence = np.nan, np.nan, "Contract curve unavailable"

    if not cftc.empty:
        latest_cftc = cftc.iloc[-1]
        net_percentile = (cftc["Managed money net"] <= latest_cftc["Managed money net"]).mean()
        positioning_vote = -1.0 if net_percentile >= 0.80 else (1.0 if net_percentile <= 0.20 else 0.0)
        positioning_evidence = f"Managed money net {latest_cftc['Managed money net']:,.0f}; {net_percentile:.0%} percentile"
    else:
        latest_cftc, net_percentile, positioning_vote = None, np.nan, np.nan
        positioning_evidence = "CFTC data unavailable"

    as_of_month = feeder.index[-1].month
    forward_month_numbers = [((as_of_month + offset - 1) % 12) + 1 for offset in range(1, 4)]
    forward_month_labels = [month_labels[value - 1] for value in forward_month_numbers]
    forward_returns = [seasonality.iloc[value - 1]["Average return"] for value in forward_month_numbers]
    forward_seasonal_return = np.prod([1 + value for value in forward_returns]) - 1
    seasonality_vote = signed_vote(forward_seasonal_return, 0.01, -0.01)
    seasonality_evidence = f"{', '.join(forward_month_labels)} historical average: {forward_seasonal_return:+.1%}"

    decision = pd.DataFrame(
        [
            {"Component": "Price trend", "Evidence": f"13-week return {trend_13:+.1%}; close vs 26-week average {distance_ma:+.1%}", "Vote": trend_vote, "Weight": 0.30},
            {"Component": "USDA supply", "Evidence": supply_evidence, "Vote": supply_vote, "Weight": 0.25},
            {"Component": "Futures curve", "Evidence": curve_evidence, "Vote": curve_vote, "Weight": 0.20},
            {"Component": "Positioning", "Evidence": positioning_evidence, "Vote": positioning_vote, "Weight": 0.15},
            {"Component": "Seasonality", "Evidence": seasonality_evidence, "Vote": seasonality_vote, "Weight": 0.10},
        ]
    )
    available = decision.dropna(subset=["Vote"])
    score = (available["Vote"] * available["Weight"]).sum() / available["Weight"].sum()
    recommendation = "BUY" if score >= 0.25 else ("SELL" if score <= -0.25 else "HOLD / WATCH")
    confidence = "high" if abs(score) >= 0.75 else ("medium" if abs(score) >= 0.45 else "low")
    decision["Signal"] = decision["Vote"].map(vote_label)
    decision["Contribution"] = decision["Vote"] * decision["Weight"]

    # Non-overlapping technical calibration.
    calibration = feeder_all.rename("Price").to_frame()
    calibration["13-week return"] = calibration["Price"].pct_change(13)
    calibration["Distance from 26-week average"] = calibration["Price"] / calibration["Price"].rolling(26).mean() - 1
    calibration["Trend vote"] = (np.sign(calibration["13-week return"]) + np.sign(calibration["Distance from 26-week average"])) / 2
    calibration["Forward 13-week return"] = calibration["Price"].shift(-13) / calibration["Price"] - 1
    calibration = calibration.dropna().iloc[::13].copy()
    calibration["Regime"] = calibration["Trend vote"].map({-1.0: "Bearish", 0.0: "Mixed", 1.0: "Bullish"})
    regime_order = ["Bearish", "Mixed", "Bullish"]
    calibration_summary = (
        calibration.groupby("Regime", observed=False)["Forward 13-week return"]
        .agg(Observations="count", Average="mean", Median="median", Worst="min", Best="max")
        .reindex(regime_order)
    )
    calibration_summary["Positive share"] = (
        calibration.assign(Positive=calibration["Forward 13-week return"] > 0)
        .groupby("Regime", observed=False)["Positive"]
        .mean()
        .reindex(regime_order)
    )

    charts: dict[str, str] = {}
    time_charts: list[str] = []

    # 1. Price and drawdown. Keep daily observations in the page so the
    # browser can switch between 1D, 5D, 1M, 6M, 1Y, 5Y, and all history
    # without another network request.
    price_drawdown = daily_feeder / daily_feeder.cummax() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.64, 0.36], vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=daily_feeder.index, y=daily_feeder, name="Feeder cattle", line={"color": COLORS["blue"], "width": 2.4}, hovertemplate="%{x|%b %d, %Y}<br>%{y:.3f} cents/lb<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily_feeder.index, y=daily_feeder.cummax(), name="Previous high", line={"color": COLORS["muted"], "width": 1.2, "dash": "dash"}, hovertemplate="%{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_drawdown.index, y=price_drawdown, name="Drawdown", fill="tozeroy", fillcolor="rgba(156,202,227,.35)", line={"color": COLORS["blue"], "width": 1.7}, hovertemplate="%{x|%b %d, %Y}<br>%{y:.1%}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="Cents/lb", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    style_figure(fig, "Price regime and drawdown", f"Daily closes; {daily_feeder.index.min():%b %Y}–{daily_feeder.index.max():%b %Y}", 660, "x unified")
    charts["price-regime"] = chart_html(fig, "price-regime", panel_count=2)
    time_charts.append("price-regime")

    # 2. Market drivers.
    driver_window = daily_prices.loc[daily_prices.index >= daily_prices.index.max() - pd.DateOffset(years=5)]
    indexed_drivers = driver_window.apply(lambda series: series / series.dropna().iloc[0] * 100)
    driver_returns = driver_window.pct_change(fill_method=None).dropna()
    correlations = driver_returns.corr()["Feeder cattle"].drop("Feeder cattle")
    premium = driver_window["Feeder cattle"] - driver_window["Live cattle"]
    fig = make_subplots(rows=3, cols=1, row_heights=[0.48, 0.22, 0.30], vertical_spacing=0.13)
    styles = {"Feeder cattle": (COLORS["blue"], "solid"), "Live cattle": (COLORS["orange"], "dash"), "Corn": (COLORS["olive"], "dot")}
    for name, (color, dash) in styles.items():
        fig.add_trace(go.Scatter(x=indexed_drivers.index, y=indexed_drivers[name], name=name, line={"color": color, "dash": dash, "width": 2}, hovertemplate=f"{name}<br>%{{x|%b %d, %Y}}<br>%{{y:.1f}}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=correlations.values, y=correlations.index, orientation="h", name="Correlation", marker={"color": [COLORS["orange"], COLORS["olive"]]}, text=[f"{v:.2f}" for v in correlations], textposition="outside", showlegend=False, hovertemplate="%{y}: %{x:.2f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=premium.index, y=premium, name="Feeder premium", line={"color": COLORS["blue"], "width": 1.8}, showlegend=False, hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f} cents/lb<extra></extra>"), row=3, col=1)
    fig.update_yaxes(title_text="Start = 100", row=1, col=1)
    fig.update_xaxes(range=[-1, 1], title_text="Correlation", row=2, col=1)
    fig.update_yaxes(title_text="Cents/lb", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    style_figure(fig, "Relationships with live cattle and corn", "Five-year indexed daily prices, weekly-return correlations, and feeder premium", 810, "x unified")
    charts["market-drivers"] = chart_html(fig, "market-drivers", panel_count=3)
    time_charts.append("market-drivers")

    # 3. USDA supply history and current year-over-year comparison.
    if not usda.empty:
        monthly_feeder = daily_prices["Feeder cattle"].resample("ME").last()
        chart_start = usda["Release date"].min() - pd.DateOffset(months=1)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        panels = [
            ("Feeder-cattle price", monthly_feeder.loc[monthly_feeder.index >= chart_start], COLORS["orange"], "Cents/lb"),
            ("Cattle on feed", usda.set_index("Release date")["Inventory"], COLORS["blue"], "1,000 head"),
            ("Placements", usda.set_index("Release date")["Placements"], COLORS["olive"], "1,000 head"),
            ("Marketings", usda.set_index("Release date")["Marketings"], COLORS["gold"], "1,000 head"),
        ]
        for row, (name, series, color, unit) in enumerate(panels, start=1):
            fig.add_trace(go.Scatter(x=series.index, y=series, name=name, line={"color": color, "width": 2}, marker={"size": 4}, mode="lines+markers", hovertemplate=f"{name}<br>%{{x|%b %d, %Y}}<br>%{{y:,.1f}} {unit}<extra></extra>"), row=row, col=1)
            fig.update_yaxes(title_text=unit, row=row, col=1)
        # Plotly suppresses tick labels on upper panels when x-axes are
        # shared. Keep the dates visible so each supply series can be read
        # without tracing down to the bottom subplot.
        fig.update_xaxes(showticklabels=True)
        fig.update_xaxes(title_text="Month / release date", row=4, col=1)
        style_figure(fig, "Feeder-cattle price and USDA supply measures", "Feedlots with capacity of 1,000+ head", 850, "x unified")
        charts["usda-history"] = chart_html(fig, "usda-history", panel_count=4)
        time_charts.append("usda-history")

        yoy = latest_usda[["Inventory YoY", "Placements YoY", "Marketings YoY"]].copy()
        yoy.index = ["Inventory", "Placements", "Marketings"]
        fig = go.Figure(go.Bar(x=yoy.values / 100, y=yoy.index, orientation="h", marker={"color": [COLORS["blue"] if value >= 0 else COLORS["blue_light"] for value in yoy], "line": {"color": COLORS["blue"], "width": 1}}, text=[f"{value:+.0f}%" for value in yoy], textposition="outside", hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))
        fig.add_vline(x=0, line_color=COLORS["ink"], line_width=1)
        fig.update_xaxes(tickformat="+.0%", title_text="Change from same month one year earlier")
        style_figure(fig, "Latest USDA measures versus prior year", f"Release dated {latest_usda['Release date']:%b %d, %Y}", 390)
        charts["usda-yoy"] = chart_html(fig, "usda-yoy")
    else:
        charts["usda-history"] = charts["usda-yoy"] = '<div class="unavailable">USDA charts unavailable for this build.</div>'

    # 4. Drought.
    if not drought_index.empty:
        drought_weekly = drought_index.resample("W-FRI").last()
        aligned = pd.concat([returns.rename("Feeder return"), drought_weekly.diff().rename("D2+ change")], axis=1).dropna()
        drought_corr = aligned.corr().iloc[0, 1]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=feeder.index, y=feeder, name="Feeder price", line={"color": COLORS["blue"], "width": 2}, hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f} cents/lb<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=drought_index.index, y=drought_index, name="D2+ area", fill="tozeroy", fillcolor="rgba(214,168,75,.28)", line={"color": COLORS["olive"], "width": 1.7}, hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}% of area<extra></extra>"), row=2, col=1)
        fig.update_yaxes(title_text="Cents/lb", row=1, col=1)
        fig.update_yaxes(title_text="Percent of area", rangemode="tozero", row=2, col=1)
        fig.update_xaxes(title_text="Week", row=2, col=1)
        style_figure(fig, "Feeder-cattle price and drought", f"Equal-weighted D2+ area across six cattle states; return/change correlation {drought_corr:.2f}", 660, "x unified")
        charts["drought"] = chart_html(fig, "drought", panel_count=2)
        time_charts.append("drought")
    else:
        drought_corr = np.nan
        charts["drought"] = '<div class="unavailable">Drought chart unavailable for this build.</div>'

    # 5. Futures curve.
    if len(curve) >= 2:
        labels = curve["Contract month"].dt.strftime("%b %Y")
        fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], vertical_spacing=0.18)
        fig.add_trace(go.Scatter(x=labels, y=curve["Latest close"], mode="lines+markers+text", text=[f"{value:.2f}" for value in curve["Latest close"]], textposition="top center", name="Close", line={"color": COLORS["blue"], "width": 2.2}, hovertemplate="%{x}<br>%{y:.3f} cents/lb<extra></extra>"), row=1, col=1)
        spread_colors = [COLORS["blue"] if value >= 0 else COLORS["blue_light"] for value in curve["Approx. dollars vs nearby"]]
        fig.add_trace(go.Bar(x=labels, y=curve["Approx. dollars vs nearby"], name="Value vs nearby", marker={"color": spread_colors, "line": {"color": COLORS["blue"], "width": 1}}, text=["" if abs(value) < 1 else f"${value:,.0f}" for value in curve["Approx. dollars vs nearby"]], textposition="auto", hovertemplate="%{x}<br>$%{y:,.0f}/contract<extra></extra>"), row=2, col=1)
        fig.update_yaxes(title_text="Cents/lb", row=1, col=1)
        fig.update_yaxes(title_text="$/contract", row=2, col=1)
        style_figure(fig, "Feeder-cattle futures term structure", "Latest listed-contract closes; 50,000 pounds per contract; focused price scale", 620)
        charts["curve"] = chart_html(fig, "curve", panel_count=2)
    else:
        charts["curve"] = '<div class="unavailable">Listed-contract curve unavailable for this build.</div>'

    # 6. Positioning.
    if not cftc.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=cftc["Date"], y=cftc["Managed money net"], name="Managed money net", fill="tozeroy", fillcolor="rgba(156,202,227,.35)", line={"color": COLORS["blue"], "width": 1.7}, hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f} contracts<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=cftc["Date"], y=cftc["open_interest_all"], name="Open interest", line={"color": COLORS["orange"], "width": 1.9}, hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f} contracts<extra></extra>"), row=2, col=1)
        fig.update_yaxes(title_text="Contracts", row=1, col=1)
        fig.update_yaxes(title_text="Contracts", row=2, col=1)
        fig.update_xaxes(title_text="CFTC report date", row=2, col=1)
        style_figure(fig, "CFTC positioning and participation", f"Latest managed-money net is at the {net_percentile:.0%} percentile of this sample", 650, "x unified")
        charts["positioning"] = chart_html(fig, "positioning", panel_count=2)
        time_charts.append("positioning")
    else:
        charts["positioning"] = '<div class="unavailable">CFTC positioning chart unavailable for this build.</div>'

    # 7. Seasonality.
    seasonal_colors = [COLORS["blue"] if value >= 0 else COLORS["blue_light"] for value in seasonality["Average return"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=month_labels, y=seasonality["Average return"], name="Average", marker={"color": seasonal_colors, "line": {"color": COLORS["blue"], "width": 1}}, customdata=np.stack([seasonality["Positive share"], seasonality["Observations"]], axis=-1), hovertemplate="%{x}<br>Average %{y:+.1%}<br>Positive %{customdata[0]:.0%}<br>n=%{customdata[1]:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=month_labels, y=seasonality["Median return"], name="Median", mode="markers", marker={"color": COLORS["orange"], "symbol": "diamond", "size": 9}, hovertemplate="%{x}<br>Median %{y:+.1%}<extra></extra>"))
    fig.add_hline(y=0, line_color=COLORS["ink"], line_width=1)
    fig.update_yaxes(title_text="Monthly return", tickformat="+.0%")
    style_figure(fig, "Feeder-cattle monthly seasonality", f"Full continuous-contract history; {feeder_monthly.index.min():%b %Y}–{feeder_monthly.index.max():%b %Y}", 480)
    charts["seasonality"] = chart_html(fig, "seasonality")

    # 8. USDA event study.
    report_events = event_returns(daily_prices["Feeder cattle"], usda["Release date"]) if not usda.empty else pd.DataFrame()
    if not report_events.empty:
        fig = go.Figure()
        fig.add_trace(go.Box(y=report_events["Next-session return"], name="Next session", marker_color=COLORS["blue"], boxmean=True, boxpoints="all", jitter=0.25, pointpos=0, hovertemplate="%{y:+.1%}<extra>Next session</extra>"))
        fig.add_trace(go.Box(y=report_events["Five-session return"], name="Five sessions", marker_color=COLORS["gold"], boxmean=True, boxpoints="all", jitter=0.25, pointpos=0, hovertemplate="%{y:+.1%}<extra>Five sessions</extra>"))
        fig.add_hline(y=0, line_color=COLORS["ink"], line_width=1)
        fig.update_yaxes(title_text="Return", tickformat="+.1%")
        style_figure(fig, "Returns following USDA Cattle on Feed releases", f"{len(report_events)} recent releases; box, mean, and individual events", 480)
        charts["usda-events"] = chart_html(fig, "usda-events")
    else:
        charts["usda-events"] = '<div class="unavailable">USDA event study unavailable for this build.</div>'

    # 9. Risk diagnostics.
    rolling_return = feeder.pct_change(52)
    rolling_vol = returns.rolling(52).std() * math.sqrt(52)
    extremes = pd.concat([returns.nsmallest(5).rename("Return").to_frame().assign(Group="Worst"), returns.nlargest(5).rename("Return").to_frame().assign(Group="Best")]).sort_values("Return")
    extreme_dates = [index.strftime("%Y-%m-%d") for index in extremes.index]
    fig = make_subplots(rows=4, cols=1, vertical_spacing=0.09, subplot_titles=("Rolling 52-week price return", "Rolling 52-week annualized volatility", "Distribution of weekly returns", "Five worst and five best weeks"))
    fig.add_trace(go.Scatter(x=rolling_return.index, y=rolling_return, line={"color": COLORS["blue"], "width": 1.8}, showlegend=False, hovertemplate="%{x|%b %d, %Y}<br>%{y:+.1%}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, line={"color": COLORS["orange"], "width": 1.8}, showlegend=False, hovertemplate="%{x|%b %d, %Y}<br>%{y:.1%}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Histogram(x=returns, nbinsx=30, marker={"color": COLORS["blue_light"], "line": {"color": COLORS["blue"], "width": 1}}, showlegend=False, hovertemplate="Return %{x:.1%}<br>Weeks %{y}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Bar(x=extremes["Return"], y=extreme_dates, orientation="h", marker={"color": [COLORS["blue_light"] if group == "Worst" else COLORS["blue"] for group in extremes["Group"]]}, text=[f"{value:+.1%}" for value in extremes["Return"]], textposition="outside", showlegend=False, hovertemplate="%{y}<br>%{x:+.1%}<extra></extra>"), row=4, col=1)
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(tickformat="+.0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(tickformat="+.0%", title_text="Weekly return", row=3, col=1)
    fig.update_xaxes(tickformat="+.0%", title_text="Weekly return", row=4, col=1)
    # ISO date strings otherwise become a continuous date axis. That makes
    # the bars nearly invisible because their default date width is tiny.
    # Treat each selected week as a category so the dates and bars are both
    # legible in the compact fourth panel.
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=extreme_dates, tickmode="array", tickvals=extreme_dates, ticktext=[index.strftime("%b %d '%y") for index in extremes.index], tickfont={"size": 10}, automargin=True, row=4, col=1)
    style_figure(fig, "Return and tail-risk diagnostics", f"Six-year weekly sample with {len(returns)} returns", 1000)
    charts["risk"] = chart_html(fig, "risk", panel_count=4)
    time_charts.append("risk")

    # 10. Recommendation contributions.
    plotted = decision.dropna(subset=["Contribution"]).sort_values("Contribution")
    fig = go.Figure(go.Bar(x=plotted["Contribution"], y=plotted["Component"], orientation="h", marker={"color": [COLORS["blue_light"] if value < 0 else COLORS["blue"] for value in plotted["Contribution"]], "line": {"color": COLORS["blue"], "width": 1}}, text=[f"{value:+.2f}" for value in plotted["Contribution"]], textposition="outside", customdata=plotted[["Evidence", "Signal"]], hovertemplate="%{y}<br>Contribution %{x:+.2f}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>"))
    fig.add_vline(x=0, line_color=COLORS["ink"], line_width=1)
    fig.update_xaxes(title_text="Weighted contribution to composite score")
    style_figure(fig, "Recommendation score contributions", "Negative = bearish; positive = bullish; fixed weights sum to 100%", 460)
    charts["recommendation"] = chart_html(fig, "recommendation")

    # 11. Technical calibration.
    medians = calibration_summary["Median"]
    fig = go.Figure(go.Bar(x=regime_order, y=medians, marker={"color": [COLORS["blue_light"], COLORS["gold"], COLORS["blue"]], "line": {"color": COLORS["blue"], "width": 1}}, text=["—" if pd.isna(value) else f"{value:+.1%}" for value in medians], textposition="outside", customdata=np.stack([calibration_summary["Positive share"], calibration_summary["Observations"]], axis=-1), hovertemplate="%{x}<br>Median %{y:+.1%}<br>Positive %{customdata[0]:.0%}<br>n=%{customdata[1]:.0f}<extra></extra>"))
    fig.add_hline(y=0, line_color=COLORS["ink"], line_width=1)
    fig.update_yaxes(title_text="Median forward 13-week return", tickformat="+.0%")
    style_figure(fig, "Forward returns by technical regime", f"Non-overlapping 13-week observations; {calibration.index.min():%Y}–{calibration.index.max():%Y}", 450)
    charts["calibration"] = chart_html(fig, "calibration")

    # Reader-facing tables.
    decision_table = decision[["Component", "Evidence", "Signal", "Weight", "Contribution"]].copy()
    decision_table_html = table_html(decision_table, {"Weight": lambda x: pct(x, 0), "Contribution": lambda x: number(x, 2, True)})
    episode_table = drawdown_episodes(drawdown).head(8)
    episode_table_html = table_html(episode_table, {"Peak": lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"), "Trough": lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"), "Recovery": lambda x: "Not yet" if pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d"), "Max drawdown": lambda x: pct(x, 1)})
    calibration_table = calibration_summary.reset_index()
    calibration_table_html = table_html(calibration_table, {name: (lambda x: pct(x, 1)) for name in ["Average", "Median", "Worst", "Best", "Positive share"]})

    latest_price = feeder.iloc[-1]
    week_change = returns.iloc[-1]
    contract_change = (feeder.iloc[-1] - feeder.iloc[-2]) * 500
    current_drawdown = drawdown.iloc[-1]
    latest_52 = feeder.pct_change(52).iloc[-1]
    recent_weekly_vol_points = returns.tail(13).std() * latest_price
    risk_band_points = 1.5 * recent_weekly_vol_points
    risk_band_dollars = risk_band_points * 500
    score_class = "positive" if recommendation == "BUY" else ("negative" if recommendation == "SELL" else "neutral")

    cards = [
        ("Latest close", f"{latest_price:.3f}", "cents/lb", f"Through {feeder.index[-1]:%b %d, %Y}"),
        ("Latest week", pct(week_change, 1, True), "Friday to Friday", f"≈ {contract_change:+,.0f} USD/contract"),
        ("Trailing 52 weeks", pct(latest_52, 1, True), "price change", f"Annual vol {annual_volatility:.1%}"),
        ("Current drawdown", pct(current_drawdown, 1), "from prior high", f"Worst {drawdown.min():.1%}"),
        ("13-week stance", recommendation, f"{confidence} confidence", f"Composite {score:+.2f}"),
    ]
    card_parts = []
    for label, value, unit, context in cards:
        is_live_price = label == "Latest close"
        card_parts.append(
            f'<article class="metric-card {score_class if label == "13-week stance" else ""}"'
            f'{" id=\"live-price-card\" aria-live=\"polite\"" if is_live_price else ""}>'
            f'<span{ " id=\"live-price-label\"" if is_live_price else ""}>{html.escape(label)}</span>'
            f'<strong{ " id=\"live-price-value\"" if is_live_price else ""}>{html.escape(value)}</strong>'
            f'<small{ " id=\"live-price-unit\"" if is_live_price else ""}>{html.escape(unit)}</small>'
            f'<p{ " id=\"live-price-context\"" if is_live_price else ""}>{html.escape(context)}</p></article>'
        )
    cards_html = "".join(card_parts)

    source_status_html = "".join(
        f'<li><span>{html.escape(name)}</span><strong class="{"warn" if status.startswith("Unavailable") else "ok"}">{html.escape(status)}</strong></li>'
        for name, status in source_status.items()
    )
    as_of = feeder.index.max()
    min_date = daily_feeder.index.min().date().isoformat()
    max_date = as_of.date().isoformat()

    content = f"""
    <section class="hero" data-topic="overview">
      <div class="eyebrow">FEEDER CATTLE · CONTINUOUS CONTRACT</div>
      <h1>Cattle Futures <em>Market Monitor</em></h1>
      <p class="lede">Price regime, physical supply, cross-market drivers, positioning, seasonality, and risk—assembled into one transparent 13-week tactical view.</p>
      <div class="hero-meta"><span>Analysis through <b>{as_of:%B %d, %Y}</b></span><span>Built <b>{built_at:%B %d, %Y · %H:%M UTC}</b></span><span id="live-status" class="live-status" role="status">Checking market feed…</span></div>
    </section>

    <section class="control-panel" aria-label="Dashboard controls">
      <div class="control-group wide"><label for="range-preset">Date range</label><div class="date-row"><select id="range-preset">{''.join(f'<option value="{value}"{" selected" if value == "all" else ""}>{label}</option>' for value, label in RANGE_PRESETS)}<option value="custom">Custom dates</option></select><input id="start-date" type="date" min="{min_date}" max="{max_date}" value="{min_date}" aria-label="Start date"><span>to</span><input id="end-date" type="date" min="{min_date}" max="{max_date}" value="{max_date}" aria-label="End date"><button id="apply-range" class="primary">Apply</button></div><small class="control-help">Changes the two Market Pulse charts first. Weekly/monthly charts only respond when the range is long enough for their data; summary charts do not change. Daily price history is preloaded, so changing the range does not fetch again.</small></div>
      <div class="control-group"><label for="topic-filter">Focus</label><select id="topic-filter"><option value="all">All analysis</option><option value="overview">Overview</option><option value="fundamentals">Supply & drought</option><option value="market">Market structure</option><option value="positioning">Positioning & seasonality</option><option value="risk">Risk & decision</option></select></div>
      <button id="theme-toggle" class="theme-toggle" aria-label="Toggle color theme"><span aria-hidden="true">◐</span><span class="theme-label">Dark</span></button>
    </section>

    <section class="metric-strip" data-topic="overview risk">{cards_html}</section>

    <section class="analysis-section" data-topic="overview">
      <div class="section-heading"><div><span class="section-number">01</span><h2>Market pulse</h2></div><p>Start with direction, distance from the high, and the markets that most directly frame feeder-cattle economics.</p></div>
      <article class="chart-card">{charts['price-regime']}{source_note('Yahoo Finance', 'yahoo', 'GF=F daily closes; full history is embedded for local range switching')}</article>
      <article class="chart-card">
        <div class="inline-controls" aria-label="Commodity visibility"><span>Visible series</span><label><input type="checkbox" data-trace-chart="market-drivers" data-trace-name="Feeder cattle" checked> Feeder</label><label><input type="checkbox" data-trace-chart="market-drivers" data-trace-name="Live cattle" checked> Live cattle</label><label><input type="checkbox" data-trace-chart="market-drivers" data-trace-name="Corn" checked> Corn</label></div>
        {charts['market-drivers']}{source_note('Yahoo Finance', 'yahoo', 'GF=F, LE=F, and ZC=F; five-year daily window; correlations use weekly returns')}
      </article>
    </section>

    <section class="analysis-section" data-topic="fundamentals">
      <div class="section-heading"><div><span class="section-number">02</span><h2>Physical supply</h2></div><p>USDA feedlot inventory, placements, and marketings provide the core supply read. Drought adds a six-state pasture-pressure context.</p></div>
      <div class="chart-grid two"><article class="chart-card">{charts['usda-history']}{source_note('USDA Cattle on Feed', 'usda', usda_status)}</article><article class="chart-card compact">{charts['usda-yoy']}{source_note('USDA Cattle on Feed', 'usda', 'Latest parsed release')}</article></div>
      <article class="chart-card">{charts['drought']}{source_note('U.S. Drought Monitor', 'drought', drought_status)}</article>
    </section>

    <section class="analysis-section" data-topic="market">
      <div class="section-heading"><div><span class="section-number">03</span><h2>Term structure</h2></div><p>The curve compares listed delivery months with the nearby contract. Dollar differences use the CME 50,000-pound contract size.</p></div>
      <article class="chart-card">{charts['curve']}{source_note('Yahoo Finance / CME', 'cme', curve_status)}</article>
    </section>

    <section class="analysis-section" data-topic="positioning">
      <div class="section-heading"><div><span class="section-number">04</span><h2>Positioning & seasonality</h2></div><p>Managed-money crowding is treated contrarian only at sample extremes. Calendar effects describe history, not a forecast.</p></div>
      <div class="chart-grid two"><article class="chart-card">{charts['positioning']}{source_note('CFTC Commitments of Traders', 'cftc', cftc_status)}</article><article class="chart-card compact">{charts['seasonality']}{source_note('Yahoo Finance', 'yahoo', 'Monthly GF=F closes; full available history')}</article></div>
    </section>

    <section class="analysis-section" data-topic="fundamentals risk">
      <div class="section-heading"><div><span class="section-number">05</span><h2>Events & risk</h2></div><p>Release-day behavior, rolling volatility, return tails, and drawdown duration turn headline moves into trade-sized context.</p></div>
      <div class="chart-grid two"><article class="chart-card compact">{charts['usda-events']}{source_note('USDA / Yahoo Finance', 'usda', 'Release dates aligned to next available closes')}</article><article class="chart-card">{charts['risk']}{source_note('Yahoo Finance', 'yahoo', 'Six-year weekly GF=F sample')}</article></div>
      <details class="data-detail"><summary>Largest drawdown episodes</summary>{episode_table_html}</details>
    </section>

    <section class="analysis-section decision-section" data-topic="risk overview">
      <div class="section-heading"><div><span class="section-number">06</span><h2>Tactical decision</h2></div><p>Five bounded votes combine into a {horizon_weeks}-week stance. Missing optional inputs are excluded and remaining weights are re-normalized.</p></div>
      <div class="decision-banner {score_class}"><div><span>Current stance</span><strong>{recommendation}</strong></div><div><span>Composite score</span><strong>{score:+.2f}</strong></div><div><span>Confidence</span><strong>{confidence.title()}</strong></div><p>Risk reference: 1.5× recent weekly volatility is <b>{risk_band_points:.2f} cents/lb</b>, about <b>${risk_band_dollars:,.0f} per contract</b>. Reassess when the factor alignment changes.</p></div>
      <div class="chart-grid two"><article class="chart-card compact">{charts['recommendation']}{source_note('Composite model', 'yahoo', 'Notebook rules; price plus available USDA, curve, CFTC, and seasonality inputs')}</article><article class="chart-card compact">{charts['calibration']}{source_note('Yahoo Finance', 'yahoo', 'Non-overlapping observations; technical component only')}</article></div>
      <details class="data-detail" open><summary>Recommendation inputs</summary>{decision_table_html}</details>
      <details class="data-detail"><summary>Technical calibration table</summary>{calibration_table_html}</details>
    </section>

    <section class="methodology" data-topic="methodology all">
      <div><span class="section-number">07</span><h2>Methods, sources & limitations</h2></div>
      <div class="method-grid">
        <article><h3>What the signal means</h3><p>The {horizon_weeks}-week recommendation combines trend (30%), USDA supply (25%), curve (20%), positioning (15%), and forward seasonality (10%). BUY begins at +0.25 and SELL at −0.25. It is a transparent decision aid, not an optimized system.</p></article>
        <article><h3>Contract math</h3><p>One CME feeder-cattle contract represents 50,000 pounds. A 1.00 move in the quoted price is approximately $500 per contract. Continuous-contract history does not include roll execution, fees, margin, or collateral interest.</p></article>
        <article><h3>Important limitations</h3><p>Correlations and event studies are descriptive. USDA releases may be revised. CFTC data are delayed. Drought is equal-weighted across six states. Seasonality and historical regimes can change. Futures can lose more than initial margin.</p></article>
      </div>
      <details class="source-status"><summary>Build-time source status</summary><ul>{source_status_html}</ul></details>
    </section>
    """

    live_feed_url = os.environ.get("CATTLE_LIVE_FEED_URL", "live-market.json").strip()
    page = render_page(
        content=content,
        plotly_js=get_plotlyjs(),
        time_charts=time_charts,
        min_date=min_date,
        max_date=max_date,
        range_chart_min_days=RANGE_CHART_MIN_DAYS,
        live_feed_url=live_feed_url,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(page, encoding="utf-8")
    (OUTPUT_DIR / ".nojekyll").write_text("", encoding="utf-8")
    metadata = {
        "built_at": built_at.isoformat(),
        "data_through": max_date,
        "chart_count": len(charts),
        "time_chart_ids": time_charts,
        "range_presets": [value for value, _ in RANGE_PRESETS],
        "range_chart_min_days": RANGE_CHART_MIN_DAYS,
        "range_data": {
            "source": "embedded Plotly chart data",
            "price_frequency": "daily",
            "price_start": min_date,
            "price_end": max_date,
        },
        "source_status": source_status,
        "recommendation": recommendation,
        "recommendation_score": round(float(score), 4),
    }
    (OUTPUT_DIR / "build-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    daily_feeder = daily_prices["Feeder cattle"].dropna()
    previous_daily_close = float(daily_feeder.iloc[-2]) if len(daily_feeder) > 1 else float(latest_price)
    seed_feed = {
        "schema_version": 1,
        "generated_at": built_at.isoformat(),
        "refresh_seconds": 60,
        "source": "Yahoo Finance",
        "feed_mode": "snapshot",
        "quotes": {
            "GF=F": {
                "symbol": "GF=F",
                "name": "Feeder cattle",
                "price": float(daily_feeder.iloc[-1]),
                "previous_close": previous_daily_close,
                "change": float(daily_feeder.iloc[-1] - previous_daily_close),
                "change_percent": float(daily_feeder.iloc[-1] / previous_daily_close - 1),
                "market_time": daily_feeder.index[-1].date().isoformat() + "T21:00:00Z",
                "market_state": "SNAPSHOT",
                "exchange_timezone": "America/Chicago",
            }
        },
    }
    LIVE_FEED_FILE.write_text(json.dumps(seed_feed, indent=2), encoding="utf-8")
    return metadata


def render_page(
    content: str,
    plotly_js: str,
    time_charts: list[str],
    min_date: str,
    max_date: str,
    range_chart_min_days: dict[str, int],
    live_feed_url: str,
) -> str:
    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Interactive feeder cattle futures market dashboard with price, supply, curve, positioning, seasonality, risk, and tactical recommendation views.">
  <meta name="color-scheme" content="light dark">
  <title>Cattle Futures Market Monitor</title>
  <script>{plotly_js}</script>
  <style>
    :root {{ --bg:#f3f1ec; --surface:#ffffff; --surface-2:#ebe7df; --ink:#20252d; --muted:#68707b; --line:#d8d3ca; --blue:#31688e; --blue-soft:#dcebf3; --orange:#e07a3f; --gold:#d6a84b; --olive:#738b3b; --shadow:0 12px 35px rgba(38,42,48,.08); --radius:18px; }}
    html[data-theme="dark"] {{ --bg:#11151a; --surface:#191f26; --surface-2:#222a33; --ink:#eef1f4; --muted:#a8b0ba; --line:#343e49; --blue:#74b4d3; --blue-soft:#243b48; --orange:#ef9a6a; --gold:#e0ba68; --olive:#9ebd60; --shadow:0 15px 42px rgba(0,0,0,.24); }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; background:var(--bg); }}
    body {{ margin:0; color:var(--ink); background:var(--bg); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.5; overflow-x:hidden; }}
    body::before {{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.23; background-image:radial-gradient(circle at 1px 1px,var(--line) 1px,transparent 0); background-size:24px 24px; mask-image:linear-gradient(to bottom,black,transparent 46%); z-index:-1; }}
    a {{ color:var(--blue); }}
    button,input,select {{ font:inherit; color:inherit; }}
    .shell {{ width:min(1500px,calc(100% - 40px)); margin:auto; }}
    .hero {{ padding:76px 2px 38px; max-width:980px; }}
    .eyebrow {{ font-size:.73rem; font-weight:800; letter-spacing:.17em; color:var(--blue); margin-bottom:14px; }}
    h1 {{ font-size:clamp(2.8rem,7vw,6.5rem); line-height:.92; letter-spacing:-.065em; margin:0; font-weight:830; }}
    h1 em {{ display:block; color:var(--muted); font-style:normal; font-weight:420; }}
    .lede {{ max-width:780px; color:var(--muted); font-size:clamp(1rem,2vw,1.25rem); margin:26px 0 18px; }}
    .hero-meta {{ display:flex; gap:22px; flex-wrap:wrap; font-size:.84rem; color:var(--muted); }}
    .hero-meta b {{ color:var(--ink); }}
    .live-status {{ display:inline-flex; align-items:center; gap:7px; }}
    .live-status::before {{ content:""; width:7px; height:7px; border-radius:50%; background:var(--muted); }}
    .live-status.is-live::before {{ background:var(--olive); box-shadow:0 0 0 4px color-mix(in srgb,var(--olive) 18%,transparent); }}
    .live-status.is-stale::before,.live-status.is-error::before {{ background:var(--orange); }}
    .control-panel {{ position:sticky; top:10px; z-index:30; display:grid; grid-template-columns:minmax(0,2fr) minmax(150px,1fr) auto; gap:12px; align-items:end; background:color-mix(in srgb,var(--surface) 92%,transparent); backdrop-filter:blur(16px); border:1px solid var(--line); box-shadow:var(--shadow); border-radius:var(--radius); padding:14px; margin-bottom:20px; }}
    .control-group {{ min-width:0; }}
    .control-group label,.inline-controls>span {{ display:block; text-transform:uppercase; letter-spacing:.09em; font-size:.65rem; font-weight:800; color:var(--muted); margin:0 0 6px 2px; }}
    .control-group.wide {{ min-width:0; }} .date-row {{ display:grid; grid-template-columns:minmax(105px,.85fr) minmax(115px,1fr) auto minmax(115px,1fr) auto; gap:7px; align-items:center; }} .control-help {{ display:block; color:var(--muted); font-size:.7rem; margin:6px 0 0 2px; }}
    select,input[type="date"],button {{ border:1px solid var(--line); background:var(--surface-2); border-radius:10px; min-height:44px; padding:8px 11px; }}
    select {{ width:100%; min-width:0; }}
    input[type="date"] {{ width:100%; min-width:0; color:var(--ink); }}
    button {{ cursor:pointer; font-weight:750; }}
    button.primary {{ background:var(--blue); color:#fff; border-color:transparent; }}
    .theme-toggle {{ display:flex; gap:7px; align-items:center; justify-content:center; background:var(--surface); min-width:86px; }}
    .metric-strip {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:18px 0 72px; }}
    .metric-card {{ background:var(--surface); border:1px solid var(--line); border-radius:15px; padding:18px; min-height:150px; box-shadow:var(--shadow); }}
    .metric-card>span,.decision-banner span {{ display:block; text-transform:uppercase; letter-spacing:.09em; font-size:.65rem; font-weight:800; color:var(--muted); }}
    .metric-card strong {{ display:block; font-size:clamp(1.55rem,3vw,2.3rem); letter-spacing:-.04em; margin:9px 0 0; font-variant-numeric:tabular-nums; }}
    .metric-card small {{ color:var(--muted); }} .metric-card p {{ margin:18px 0 0; font-size:.76rem; color:var(--muted); }}
    .metric-card.positive,.decision-banner.positive {{ border-top:4px solid var(--blue); }} .metric-card.negative,.decision-banner.negative {{ border-top:4px solid var(--orange); }} .metric-card.neutral,.decision-banner.neutral {{ border-top:4px solid var(--gold); }}
    .analysis-section {{ margin:0 0 88px; scroll-margin-top:110px; }}
    .section-heading {{ display:flex; align-items:end; justify-content:space-between; gap:30px; border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:18px; }}
    .section-heading>div,.methodology>div:first-child {{ display:flex; align-items:baseline; gap:14px; }}
    .section-number {{ font:700 .75rem/1 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--orange); }}
    h2 {{ font-size:clamp(1.6rem,3.5vw,2.7rem); letter-spacing:-.045em; line-height:1; margin:0; }}
    .section-heading p {{ color:var(--muted); max-width:620px; margin:0; font-size:.9rem; }}
    .chart-card {{ position:relative; background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow); margin-bottom:16px; min-width:0; }}
    .chart-grid {{ display:grid; gap:16px; align-items:start; }} .chart-grid.two {{ grid-template-columns:1fr; }}
    .source-note {{ border-top:1px solid var(--line); color:var(--muted); padding:10px 16px 12px; font-size:.71rem; overflow-wrap:anywhere; }} .source-note span {{ text-transform:uppercase; letter-spacing:.08em; font-weight:800; }}
    .inline-controls {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; padding:14px 18px 0; font-size:.78rem; }} .inline-controls>span {{ margin:0; }} .inline-controls label {{ display:flex; gap:5px; align-items:center; min-height:32px; }}
    .unavailable {{ margin:22px; min-height:180px; display:grid; place-items:center; color:var(--muted); background:var(--surface-2); border:1px dashed var(--line); border-radius:12px; }}
    .decision-banner {{ display:grid; grid-template-columns:repeat(3,1fr) 2fr; gap:20px; align-items:center; background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:24px; margin-bottom:16px; box-shadow:var(--shadow); }}
    .decision-banner strong {{ font-size:1.7rem; font-variant-numeric:tabular-nums; }} .decision-banner p {{ margin:0; color:var(--muted); font-size:.85rem; }}
    details {{ background:var(--surface); border:1px solid var(--line); border-radius:13px; margin-top:12px; overflow:hidden; }} summary {{ cursor:pointer; font-weight:750; padding:14px 16px; }}
    .table-scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }} .data-table {{ width:100%; border-collapse:collapse; font-size:.79rem; }} .data-table th {{ text-align:left; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-size:.63rem; }} .data-table th,.data-table td {{ border-top:1px solid var(--line); padding:10px 14px; vertical-align:top; }} .data-table td:not(:nth-child(2)) {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
    .methodology {{ border-top:1px solid var(--line); padding:32px 0 80px; }} .method-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:24px 0; }} .method-grid article {{ border-left:2px solid var(--line); padding-left:16px; }} .method-grid h3 {{ margin:0 0 8px; font-size:.9rem; }} .method-grid p {{ color:var(--muted); font-size:.8rem; margin:0; }}
    .source-status ul {{ list-style:none; margin:0; padding:0 16px 14px; }} .source-status li {{ display:flex; justify-content:space-between; gap:24px; padding:8px 0; border-top:1px solid var(--line); font-size:.78rem; }} .source-status strong {{ text-align:right; }} .source-status .ok {{ color:var(--olive); }} .source-status .warn {{ color:var(--orange); }}
    footer {{ border-top:1px solid var(--line); color:var(--muted); padding:22px 0 40px; font-size:.75rem; display:flex; justify-content:space-between; gap:20px; }}
    .chart-wrap {{ width:100%; min-width:0; }} .chart-wrap>div {{ width:100%!important; }}
    .chart-wrap[data-panel-count="2"]>div:first-child {{ min-height:560px; }}
    .chart-wrap[data-panel-count="3"]>div:first-child {{ min-height:700px; }}
    .chart-wrap[data-panel-count="4"]>div:first-child {{ min-height:820px; }}
    .js-plotly-plot,.plot-container,.svg-container {{ width:100%!important; }}
    [hidden] {{ display:none!important; }}
    @media (max-width:1050px) {{ .control-panel {{ grid-template-columns:1fr 1fr; }} .control-group.wide {{ grid-column:1/-1; }} .metric-strip {{ grid-template-columns:repeat(3,1fr); }} .chart-grid.two {{ grid-template-columns:1fr; }} .decision-banner {{ grid-template-columns:repeat(3,1fr); }} .decision-banner p {{ grid-column:1/-1; }} }}
    @media (max-width:680px) {{ .shell {{ width:calc(100% - 24px); }} .hero {{ padding:42px 0 28px; }} h1 {{ font-size:clamp(2.35rem,13vw,4.2rem); overflow-wrap:anywhere; }} .lede {{ margin:20px 0 16px; }} .hero-meta {{ gap:8px 16px; font-size:.76rem; }} .control-panel {{ position:relative; top:auto; grid-template-columns:1fr; padding:12px; gap:10px; }} .control-group.wide {{ grid-column:auto; }} .date-row {{ grid-template-columns:minmax(0,1fr) minmax(0,1fr); }} .date-row select {{ grid-column:1/-1; }} .date-row span {{ display:none; }} .date-row input:first-of-type {{ grid-column:1; }} .date-row input:last-of-type {{ grid-column:2; }} .date-row .primary {{ grid-column:1/-1; width:100%; }} .metric-strip {{ grid-template-columns:1fr 1fr; gap:8px; margin:14px 0 58px; }} .metric-card {{ min-height:135px; padding:14px; }} .metric-card strong {{ font-size:clamp(1.3rem,7vw,1.85rem); }} .metric-card p {{ margin-top:12px; }} .metric-card:last-child {{ grid-column:1/-1; min-height:0; }} .section-heading {{ align-items:start; flex-direction:column; gap:10px; }} .section-heading p {{ font-size:.82rem; }} .chart-wrap>div:first-child {{ height:clamp(340px,92vw,480px)!important; }} .chart-wrap[data-panel-count="2"]>div:first-child {{ height:clamp(540px,135vw,700px)!important; }} .chart-wrap[data-panel-count="3"]>div:first-child {{ height:clamp(680px,175vw,860px)!important; }} .chart-wrap[data-panel-count="4"]>div:first-child {{ height:clamp(1000px,220vw,1250px)!important; }} .chart-card.compact .chart-wrap>div:first-child {{ height:clamp(320px,82vw,430px)!important; }} .chart-wrap .js-plotly-plot {{ touch-action:pan-y; }} .inline-controls {{ gap:8px 14px; padding:12px 14px 0; }} .inline-controls>span {{ flex-basis:100%; }} .decision-banner {{ grid-template-columns:1fr 1fr; gap:14px; padding:18px; }} .decision-banner > div:last-of-type {{ grid-column:1/-1; }} .decision-banner p {{ grid-column:1/-1; }} .method-grid {{ grid-template-columns:1fr; }} .source-status li {{ align-items:flex-start; flex-direction:column; gap:3px; }} .source-status strong {{ text-align:left; overflow-wrap:anywhere; }} footer {{ flex-direction:column; gap:8px; }} .data-table {{ min-width:620px; font-size:.7rem; }} }}
    @media (max-width:380px) {{ .metric-strip {{ grid-template-columns:1fr; }} .metric-card:last-child {{ grid-column:auto; }} .date-row {{ grid-template-columns:1fr; }} .date-row select,.date-row input:first-of-type,.date-row input:last-of-type,.date-row .primary {{ grid-column:1; }} .decision-banner {{ grid-template-columns:1fr; }} .decision-banner > div:last-of-type {{ grid-column:auto; }} }}
    @media (prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} }}
    @media print {{ .control-panel,.inline-controls,.modebar-container {{ display:none!important; }} body {{ background:#fff; }} .shell {{ width:100%; }} .chart-card,.metric-card {{ box-shadow:none; break-inside:avoid; }} }}
  </style>
</head>
<body>
  <main class="shell">{content}</main>
  <footer class="shell"><span>General market research—not individualized financial advice.</span><span>Python-built · Static GitHub Pages snapshot · No cookies or analytics</span></footer>
  <script>
  (() => {{
    const timeCharts = {json.dumps(time_charts)};
    const minDate = {json.dumps(min_date)};
    let maxDate = {json.dumps(max_date)};
    const rangeChartMinDays = {json.dumps(range_chart_min_days)};
    const liveFeedUrl = {json.dumps(live_feed_url)};
    const root = document.documentElement;
    const startInput = document.getElementById('start-date');
    const endInput = document.getElementById('end-date');
    const rangePreset = document.getElementById('range-preset');
    const themeButton = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('cattle-theme');
    const initialTheme = savedTheme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

    function themePalette(theme) {{
      return theme === 'dark'
        ? {{paper:'#191f26', plot:'#191f26', ink:'#eef1f4', grid:'#343e49', line:'#677483'}}
        : {{paper:'#ffffff', plot:'#ffffff', ink:'#252a34', grid:'#d9dce1', line:'#89919b'}};
    }}
    function applyTheme(theme) {{
      root.dataset.theme = theme;
      localStorage.setItem('cattle-theme', theme);
      themeButton.querySelector('.theme-label').textContent = theme === 'dark' ? 'Light' : 'Dark';
      const p = themePalette(theme);
      document.querySelectorAll('.js-plotly-plot').forEach(gd => {{
        const update = {{paper_bgcolor:p.paper, plot_bgcolor:p.plot, 'font.color':p.ink}};
        Object.keys(gd._fullLayout || {{}}).filter(k => /^xaxis\\d*$|^yaxis\\d*$/.test(k)).forEach(k => {{
          update[k + '.gridcolor'] = p.grid; update[k + '.linecolor'] = p.line; update[k + '.zerolinecolor'] = p.line;
        }});
        Plotly.relayout(gd, update);
      }});
    }}
    function applyMobileChartLayout() {{
      const isMobile = matchMedia('(max-width:680px)').matches;
      document.querySelectorAll('.js-plotly-plot').forEach(gd => {{
        const wrapper = gd.closest('.chart-wrap');
        const panelCount = Number(wrapper && wrapper.dataset.panelCount) || 1;
        gd.style.touchAction = isMobile ? 'pan-y' : 'auto';
        const update = isMobile
          ? {{
              'margin.l': 52,
              'margin.r': 16,
              'margin.t': panelCount > 1 ? 106 : 92,
              'margin.b': 70,
              'title.font.size': 15,
              'legend.font.size': 10,
              'dragmode': false
            }}
          : {{'dragmode': 'zoom'}};
        if (isMobile) {{
          Object.keys(gd._fullLayout || {{}}).filter(key => /^xaxis\\d*$|^yaxis\\d*$/.test(key)).forEach(key => {{
            update[key + '.title.font.size'] = 10;
            update[key + '.tickfont.size'] = 10;
            update[key + '.automargin'] = true;
          }});
          if (panelCount > 1) update['legend.tracegroupgap'] = 0;
        }}
        Plotly.relayout(gd, update).then(() => Plotly.Plots.resize(gd));
      }});
    }}
    function applyDateRange() {{
      const start = startInput.value, end = endInput.value;
      if (!start || !end || start > end) return;
      const requestedDays = Math.max(1, Math.round((Date.parse(end + 'T12:00:00Z') - Date.parse(start + 'T12:00:00Z')) / 86400000));
      timeCharts.forEach(id => {{
        const gd = document.getElementById(id);
        if (!gd || !gd._fullLayout || requestedDays < (rangeChartMinDays[id] || 1)) return;
        const update = {{}};
        Object.keys(gd._fullLayout).filter(k => /^xaxis\\d*$/.test(k) && gd._fullLayout[k].type === 'date').forEach(k => update[k + '.range'] = [start, end]);
        Object.keys(gd._fullLayout).filter(k => /^yaxis\\d*$/.test(k)).forEach(k => update[k + '.autorange'] = true);
        if (Object.keys(update).length) Plotly.relayout(gd, update);
      }});
    }}
    function presetStart(preset) {{
      if (preset === 'all') return minDate;
      const date = new Date(maxDate + 'T12:00:00Z');
      if (preset === '1d') date.setUTCDate(date.getUTCDate() - 1);
      if (preset === '5d') date.setUTCDate(date.getUTCDate() - 5);
      const months = preset === '1m' ? 1 : preset === '6m' ? 6 : preset === '1y' ? 12 : preset === '5y' ? 60 : 0;
      if (months) {{
        const day = date.getUTCDate();
        date.setUTCDate(1);
        date.setUTCMonth(date.getUTCMonth() - months);
        const lastDay = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + 1, 0)).getUTCDate();
        date.setUTCDate(Math.min(day, lastDay));
      }}
      return date.toISOString().slice(0, 10) < minDate ? minDate : date.toISOString().slice(0, 10);
    }}
    function selectPreset(preset) {{
      if (preset === 'custom') return;
      startInput.value = presetStart(preset);
      endInput.value = maxDate;
      applyDateRange();
    }}
    function formatMarketTime(value) {{
      const date = new Date(value);
      if (Number.isNaN(date.valueOf())) return 'time unavailable';
      return new Intl.DateTimeFormat('en-US', {{
        month:'short', day:'numeric', hour:'numeric', minute:'2-digit', timeZone:'America/Chicago', timeZoneName:'short'
      }}).format(date);
    }}
    function updateLiveMarker(quote) {{
      const gd = document.getElementById('price-regime');
      if (!gd || !gd.data || !quote.market_time) return;
      const traceIndex = gd.data.findIndex(trace => trace.name === 'Live quote');
      const trace = {{
        x:[quote.market_time], y:[quote.price], name:'Live quote', mode:'markers', xaxis:'x', yaxis:'y',
        marker:{{color:'#e07a3f',size:11,line:{{color:'#ffffff',width:2}}}},
        hovertemplate:'Live / delayed quote<br>%{{x|%b %d, %Y %H:%M}}<br>%{{y:.3f}} cents/lb<extra></extra>'
      }};
      if (traceIndex >= 0) Plotly.restyle(gd, {{x:[trace.x], y:[trace.y]}}, [traceIndex]);
      else Plotly.addTraces(gd, trace);
    }}
    function applyLiveFeed(payload) {{
      const quote = payload && payload.quotes && payload.quotes['GF=F'];
      if (!quote || !Number.isFinite(Number(quote.price))) throw new Error('Feed has no valid GF=F quote');
      quote.price = Number(quote.price);
      const change = Number(quote.change);
      const changePercent = Number(quote.change_percent);
      const state = String(quote.market_state || 'UNKNOWN').toUpperCase();
      const isSnapshot = payload.feed_mode === 'snapshot' || state === 'SNAPSHOT';
      const generated = new Date(payload.generated_at);
      const feedAgeMinutes = Number.isNaN(generated.valueOf()) ? Infinity : (Date.now() - generated.valueOf()) / 60000;
      const isStale = !isSnapshot && feedAgeMinutes > 15;
      const status = document.getElementById('live-status');
      const stateLabel = state === 'REGULAR' || state === 'OPEN' ? 'Market open' : state === 'CLOSED' ? 'Market closed' : 'Market status unknown';
      status.className = 'live-status ' + (isStale || isSnapshot ? 'is-stale' : 'is-live');
      status.textContent = isSnapshot
        ? 'Quote snapshot · refreshes with dashboard'
        : (isStale ? 'Market feed stale' : 'Feed current') + ' · ' + stateLabel;
      document.getElementById('live-price-label').textContent = isSnapshot ? 'Latest close' : 'Latest quote';
      document.getElementById('live-price-value').textContent = quote.price.toFixed(3);
      document.getElementById('live-price-unit').textContent = 'cents/lb';
      const move = Number.isFinite(change) && Number.isFinite(changePercent)
        ? `${{change >= 0 ? '+' : ''}}${{change.toFixed(3)}} (${{changePercent >= 0 ? '+' : ''}}${{(changePercent * 100).toFixed(2)}}%) vs prior close · `
        : '';
      document.getElementById('live-price-context').textContent = move + 'As of ' + formatMarketTime(quote.market_time);
      updateLiveMarker(quote);
      const quoteDate = String(quote.market_time || '').slice(0, 10);
      if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}$/.test(quoteDate) && quoteDate > maxDate) {{
        maxDate = quoteDate;
        endInput.max = quoteDate;
      }}
    }}
    async function refreshLiveFeed() {{
      if (!liveFeedUrl) return;
      try {{
        const separator = liveFeedUrl.includes('?') ? '&' : '?';
        const response = await fetch(liveFeedUrl + separator + 't=' + Date.now(), {{cache:'no-store'}});
        if (!response.ok) throw new Error('Feed returned HTTP ' + response.status);
        applyLiveFeed(await response.json());
      }} catch (error) {{
        const status = document.getElementById('live-status');
        status.className = 'live-status is-error';
        status.textContent = 'Live feed unavailable · showing snapshot';
        console.warn('Unable to refresh market feed', error);
      }}
    }}
    themeButton.addEventListener('click', () => applyTheme(root.dataset.theme === 'dark' ? 'light' : 'dark'));
    document.getElementById('apply-range').addEventListener('click', applyDateRange);
    rangePreset.addEventListener('change', event => selectPreset(event.target.value));
    [startInput, endInput].forEach(input => input.addEventListener('input', () => {{ rangePreset.value = 'custom'; }}));
    document.getElementById('topic-filter').addEventListener('change', event => {{
      const topic=event.target.value;
      document.querySelectorAll('[data-topic]').forEach(section => section.hidden = topic !== 'all' && !section.dataset.topic.split(' ').includes(topic));
      setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
    }});
    document.querySelectorAll('[data-trace-chart]').forEach(input => input.addEventListener('change', event => {{
      const gd=document.getElementById(event.target.dataset.traceChart); if(!gd) return;
      const indices=[]; (gd.data || []).forEach((trace,i) => {{ if(trace.name === event.target.dataset.traceName) indices.push(i); }});
      if(indices.length) Plotly.restyle(gd, {{visible:event.target.checked ? true : 'legendonly'}}, indices);
    }}));
    window.addEventListener('resize', applyMobileChartLayout, {{passive:true}});
    requestAnimationFrame(applyMobileChartLayout);
    Promise.resolve().then(() => applyTheme(initialTheme));
    refreshLiveFeed();
    window.setInterval(refreshLiveFeed, 60_000);
  }})();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    result = build()
    print(json.dumps(result, indent=2))
