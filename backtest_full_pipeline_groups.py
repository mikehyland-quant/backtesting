import asyncio
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
script_directory = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Keep the anchor last. The anchor is the normalization reference, but any two
# members (including the anchor) can become the traded min/max legs.
sorted_symbols_list = ["EMB", "EMLC", "VWOB"]

filename = f"{'_'.join(sorted_symbols_list)}.csv"
prices_path = script_directory / "historical prices" / filename

start_date = pd.Timestamp("2024-01-02")
end_date = pd.Timestamp("2026-08-15")

moving_avg_days = 20
hurdle_step = 0.0002
number_of_entry_steps = 10
annual_borrow_cost = 0.01

ibkr_port = 7496
lookback_period = "5 Y"
length_of_each_period = "1 day"
use_regular_trading_hours = True
prices_to_use = "TRADES"

commission_per_share = 0.005
dollar_constant = 100_000
trading_days_per_year = 252
progress_interval = 10
output_directory = script_directory / "backtests"
summary_output_directory = script_directory / "summary_stats"


def report(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def enhance_prices(closes_df):
    df = closes_df.copy()
    df.reset_index(names="eop date", inplace=True)
    df.insert(0, "bop date", df["eop date"].shift(1))
    df["bop date"] = pd.to_datetime(df["bop date"], errors="coerce")
    df["eop date"] = pd.to_datetime(df["eop date"], errors="coerce")
    date_mask = (
        df["bop date"].between(start_date, end_date)
        & df["eop date"].between(start_date, end_date)
    )
    df = df.loc[date_mask].reset_index(drop=True)
    df.insert(2, "day count", (df["eop date"] - df["bop date"]).dt.days)

    for symbol in sorted_symbols_list:
        df[f"bop {symbol} price"] = df[symbol].shift(1)
        df[f"eop {symbol} price"] = df[symbol]
        df.drop(columns=symbol, inplace=True)
    return df


def calculate_unit_prices(enhanced_df):
    df = enhanced_df.copy()
    anchor = sorted_symbols_list[-1]

    for symbol in sorted_symbols_list:
        ratio_column = f"eop {anchor} / eop {symbol}"
        eop_average_column = f"eop {anchor} / {symbol} moving avg"
        bop_average_column = f"bop {anchor} / {symbol} moving avg"
        df[ratio_column] = df[f"eop {anchor} price"] / df[f"eop {symbol} price"]
        df[eop_average_column] = df[ratio_column].rolling(moving_avg_days).mean()
        df[bop_average_column] = df[eop_average_column].shift(1)
        df[f"eop {symbol} unit price"] = (
            df[bop_average_column] * df[f"eop {symbol} price"]
        )

    # Build every unit-price column before comparing any symbol with the
    # anchor, which is deliberately the last member of the input list.
    for symbol in sorted_symbols_list:
        df[f"{symbol} unit price pct diff"] = np.log(
            df[f"eop {symbol} unit price"] / df[f"eop {anchor} unit price"]
        )

    deviation_columns = [f"{s} unit price pct diff" for s in sorted_symbols_list]
    deviations = df[deviation_columns]
    valid = deviations.notna().all(axis=1)
    df["min symbol"] = None
    df["max symbol"] = None
    df.loc[valid, "min symbol"] = (
        deviations.loc[valid].idxmin(axis=1).str.removesuffix(" unit price pct diff")
    )
    df.loc[valid, "max symbol"] = (
        deviations.loc[valid].idxmax(axis=1).str.removesuffix(" unit price pct diff")
    )
    df["min unit price pct diff"] = deviations.min(axis=1)
    df["max unit price pct diff"] = deviations.max(axis=1)
    df["unit price pct diff"] = (
        df["max unit price pct diff"] - df["min unit price pct diff"]
    )
    df.loc[~valid, "unit price pct diff"] = np.nan
    return df


def build_entry_hurdles():
    return [0.0] + [
        round(hurdle_step * step, 4)
        for step in range(1, number_of_entry_steps + 1)
    ]


def build_exit_hurdles(entry_hurdle):
    number_of_steps = int(round(entry_hurdle / hurdle_step))
    return [0.0] + [
        round(hurdle_step * step, 4)
        for step in range(1, number_of_steps + 1)
    ]


def add_positions(base_df, entry_hurdle, exit_hurdle):
    df = base_df.copy()
    next_eop_date = df["eop date"].shift(-1)
    df["month end liquidation"] = (
        next_eop_date.notna()
        & (df["eop date"].dt.to_period("M") != next_eop_date.dt.to_period("M"))
    )
    df["go group"] = df["unit price pct diff"] > entry_hurdle
    df["exit group"] = df["unit price pct diff"] < exit_hurdle
    df["current long symbol"] = None
    df["current short symbol"] = None
    df["new long symbol"] = None
    df["new short symbol"] = None

    current_long = None
    current_short = None
    for row_index in range(moving_avg_days, len(df)):
        df.at[row_index, "current long symbol"] = current_long
        df.at[row_index, "current short symbol"] = current_short

        if df["month end liquidation"].iat[row_index] or df["exit group"].iat[row_index]:
            current_long = None
            current_short = None
        elif df["go group"].iat[row_index]:
            # Above entry: open or rotate to today's cheapest/richest members.
            current_long = df["min symbol"].iat[row_index]
            current_short = df["max symbol"].iat[row_index]
        # In the exit/entry band: retain the existing two names.

        df.at[row_index, "new long symbol"] = current_long
        df.at[row_index, "new short symbol"] = current_short

    df["current_position"] = df["current long symbol"].notna().astype(int)
    df["new_position"] = df["new long symbol"].notna().astype(int)
    return df


def add_shares(position_df):
    df = position_df.copy()
    anchor = sorted_symbols_list[-1]
    anchor_shares = (
        df[f"bop {anchor} / {anchor} moving avg"]
        * dollar_constant
        / df[f"eop {anchor} price"]
    ).round()

    for symbol in sorted_symbols_list:
        shares_per_unit = f"{symbol} shares per unit"
        df[shares_per_unit] = (
            anchor_shares * df[f"bop {anchor} / {symbol} moving avg"]
        ).round()
        bop_shares = f"bop {symbol} shares"
        eop_shares = f"eop {symbol} shares"
        leg_sign = np.select(
            [df["new long symbol"].eq(symbol), df["new short symbol"].eq(symbol)],
            [1, -1],
            default=0,
        )
        df[eop_shares] = leg_sign * df[shares_per_unit]
        df[bop_shares] = df[eop_shares].shift(1)
        df.at[moving_avg_days, bop_shares] = 0
        df[f"eop {symbol} shares traded"] = df[eop_shares] - df[bop_shares]
        df[f"eop {symbol} shares bought"] = df[f"eop {symbol} shares traded"].clip(lower=0)
        df[f"eop {symbol} shares sold"] = -df[f"eop {symbol} shares traded"].clip(upper=0)
    return df


def add_stats(shares_df):
    df = shares_df.copy()
    for symbol in sorted_symbols_list:
        df[f"{symbol} daily commission"] = (
            -df[f"eop {symbol} shares traded"].abs() * commission_per_share
        )
        df[f"{symbol} cop"] = df[f"eop {symbol} price"] - df[f"bop {symbol} price"]
        df[f"{symbol} pct cop"] = np.log(
            df[f"eop {symbol} price"] / df[f"bop {symbol} price"]
        )
        df[f"{symbol} daily position pnl"] = (
            df[f"bop {symbol} shares"] * df[f"{symbol} cop"]
        )
        investment = df[f"bop {symbol} shares"] * df[f"bop {symbol} price"]
        df[f"{symbol} investment amount"] = investment
        df[f"short {symbol} investment amount"] = investment.clip(upper=0)
        df[f"long {symbol} investment amount"] = investment.clip(lower=0)
        df[f"daily {symbol} borrow cost"] = (
            annual_borrow_cost
            * df[f"short {symbol} investment amount"]
            * df["day count"]
            / 360
        )

    df["daily total commissions"] = df[
        [f"{s} daily commission" for s in sorted_symbols_list]
    ].sum(axis=1)
    df["daily position pnl"] = df[
        [f"{s} daily position pnl" for s in sorted_symbols_list]
    ].sum(axis=1)
    investments = df[[f"{s} investment amount" for s in sorted_symbols_list]]
    df["net investment amount"] = investments.sum(axis=1)
    df["gross investment amount"] = investments.abs().sum(axis=1)
    df["short investment amount"] = df[
        [f"short {s} investment amount" for s in sorted_symbols_list]
    ].sum(axis=1)
    df["daily borrow cost"] = df[
        [f"daily {s} borrow cost" for s in sorted_symbols_list]
    ].sum(axis=1)
    df["daily net profit"] = (
        df["daily position pnl"] + df["daily total commissions"] + df["daily borrow cost"]
    )
    df["cumulative net profit"] = df["daily net profit"].cumsum()
    df["drawdown"] = df["cumulative net profit"] - df["cumulative net profit"].cummax()
    return df


def add_summary_stats(stats_df, entry_hurdle, exit_hurdle):
    df = stats_df
    valid_rows = df.index >= moving_avg_days
    daily_profit = df.loc[valid_rows, "daily net profit"]
    average_net = df.loc[valid_rows, "net investment amount"].mean()
    average_gross = df.loc[valid_rows, "gross investment amount"].mean()
    total_profit = daily_profit.sum()
    number_of_days = daily_profit.notna().sum()
    annualized_return = (
        total_profit / average_gross * trading_days_per_year / number_of_days
        if average_gross > 0 and number_of_days > 0 else np.nan
    )
    daily_std = daily_profit.std()
    annualized_sharpe = (
        daily_profit.mean() / daily_std * np.sqrt(trading_days_per_year)
        if daily_std != 0 and not pd.isna(daily_std) else np.nan
    )
    traded = [f"eop {s} shares traded" for s in sorted_symbols_list]
    leg_changes = (
        df["new long symbol"].fillna("").ne(df["current long symbol"].fillna(""))
        | df["new short symbol"].fillna("").ne(df["current short symbol"].fillna(""))
    )
    return pd.DataFrame([{
        "symbols": "_".join(sorted_symbols_list),
        "moving avg days": moving_avg_days,
        "entry hurdle": entry_hurdle,
        "exit hurdle": exit_hurdle,
        "total net profit": total_profit,
        "total borrow cost": df.loc[valid_rows, "daily borrow cost"].sum(),
        "average daily short investment": df.loc[valid_rows, "short investment amount"].mean(),
        "average daily net investment": average_net,
        "average daily gross investment": average_gross,
        "annualized return on avg gross investment": annualized_return,
        "annualized Sharpe": annualized_sharpe,
        "maximum drawdown": df["drawdown"].min(),
        "position changes": leg_changes.sum(),
        "total shares traded": df[traded].abs().sum().sum(),
    }])


async def main():
    if len(sorted_symbols_list) < 3:
        raise ValueError("This pipeline requires at least three symbols")
    if len(set(sorted_symbols_list)) != len(sorted_symbols_list):
        raise ValueError("sorted_symbols_list may not contain duplicates")
    if hurdle_step <= 0 or number_of_entry_steps < 1:
        raise ValueError("hurdle_step and number_of_entry_steps must be positive")

    report("Starting full group backtest pipeline")
    closes_df = pd.read_csv(prices_path, index_col="date", parse_dates=["date"])
    closes_df = closes_df[sorted_symbols_list]
    report(f"Downloaded {len(closes_df):,} completed price rows")
    base_df = calculate_unit_prices(enhance_prices(closes_df))
    strategy_count = sum(len(build_exit_hurdles(e)) for e in build_entry_hurdles())
    report(f"Calculating {strategy_count} position/statistics combinations")
    completed_results = []
    summary_dfs = []
    filename_root = f"{'_'.join(sorted_symbols_list)}_{moving_avg_days}_group"
    completed_count = 0
    for entry_hurdle in build_entry_hurdles():
        for exit_hurdle in build_exit_hurdles(entry_hurdle):
            final_df = add_stats(add_shares(add_positions(base_df, entry_hurdle, exit_hurdle)))
            summary_dfs.append(add_summary_stats(final_df, entry_hurdle, exit_hurdle))
            filename = f"{filename_root}_{entry_hurdle:.4f}_{exit_hurdle:.4f}.csv"
            completed_results.append((filename, final_df))
            completed_count += 1
            if completed_count % progress_interval == 0 or completed_count == strategy_count:
                report(f"Calculated {completed_count}/{strategy_count} strategies")

    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, final_df in completed_results:
        final_df.to_csv(output_directory / filename)
    summary_output_directory.mkdir(parents=True, exist_ok=True)
    summary_filename = f"{filename_root}_summary.csv"
    pd.concat(summary_dfs, ignore_index=True).to_csv(
        summary_output_directory / summary_filename, index=False
    )
    report(f"Detail files saved in: {output_directory.resolve()}")
    report(
        f"Summary saved as: "
        f"{(summary_output_directory / summary_filename).resolve()}"
    )
    report(f"Pipeline finished successfully: {strategy_count} detail files and 1 summary file saved")


if __name__ == "__main__":
    asyncio.run(main())
