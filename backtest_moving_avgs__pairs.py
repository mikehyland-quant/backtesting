# --- imports ---
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIRECTORY.parent))


# ============================================================
# SETTINGS - edit these values as needed
# Keep the anchor symbol last in the list.
# Use ["long"], ["short"], or ["long", "short"].
# ============================================================
prices_path         = SCRIPT_DIRECTORY / "historical prices" / "rsp_spy.csv"
sorted_symbols_list = ["RSP", "SPY"]

start_date = pd.Timestamp("2024-01-02")
end_date   = pd.Timestamp("2026-08-15")

moving_avg_days       = 10
annual_borrow_cost    = 0.01  # 1.00% annualized; change as needed
# trade_directions      = ["long", "short"]

commission_per_share     = 0.005
dollar_constant          = 100_000
trading_days_per_year    = 252
progress_interval        = 10   # for progress messaging
output_directory         = SCRIPT_DIRECTORY / "backtests"
summary_output_directory = SCRIPT_DIRECTORY / "summary_stats"

# ============================================================
# VARIABLE NAMES -
#    bop = beginning of period
#    eop = end of period
#    cop = change over period
# ============================================================


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

    # Actual calendar days represented by each row, including weekends/holidays.
    df.insert(2, "day count", (df["eop date"] - df["bop date"]).dt.days)

    for symbol in sorted_symbols_list:
        bop_price = f"bop {symbol} price"
        eop_price = f"eop {symbol} price"
        df[bop_price] = df[symbol].shift(1)
        df[eop_price] = df[symbol]
        df.drop(columns=symbol, inplace=True)

    return df


def calculate_moving_avg(enhanced_df):
    df = enhanced_df.copy()
    non_anchor = sorted_symbols_list[0]
    anchor = sorted_symbols_list[-1]

    ratio_column = f"eop {anchor} / eop {non_anchor}"
    eop_average_column = f"eop {anchor} / {non_anchor} moving avg"
    bop_average_column = eop_average_column.replace("eop", "bop")

    df[ratio_column] = (
        df[f"eop {anchor} price"] / df[f"eop {non_anchor} price"]
    )
    df[bop_average_column] = 0
    df[eop_average_column] = df[ratio_column].rolling(moving_avg_days).mean()
    df[bop_average_column] = df[eop_average_column].shift(1)

    return df

def add_positions(base_df):
    df = base_df.copy()

    df["current position"] = 0
    df["new position"] = 0

    non_anchor = sorted_symbols_list[0]
    anchor = sorted_symbols_list[-1]
    eop_average_column = f"bop {anchor} / {non_anchor} moving avg"
    df['new position'] = np.sign(
        df['eop SPY / eop RSP'] - df[eop_average_column]
    )

    df["current position"] = df["new position"].shift(1)

    return df


def add_shares(position_df):
    df = position_df.copy()
    non_anchor = sorted_symbols_list[0]
    anchor = sorted_symbols_list[-1]

    for symbol in sorted_symbols_list:
        leg_sign = -1 if symbol == non_anchor else 1

        bop_shares = f"bop {symbol} shares"
        eop_shares = f"eop {symbol} shares"

        df[bop_shares] = None
        df[eop_shares] = df["new position"] * leg_sign * dollar_constant / df[f"eop {symbol} price"]
        df[bop_shares] = df[eop_shares].shift(1)

        df.iat[moving_avg_days, df.columns.get_loc(bop_shares)] = 0

        df[f"eop {symbol} shares bought"] = df[bop_shares]
        df[f"eop {symbol} shares sold"] = df[bop_shares]

        df[f"eop {symbol} shares traded"] = df[eop_shares] - df[bop_shares]
        df[f"eop {symbol} shares bought"] = (
            df[f"eop {symbol} shares traded"].clip(lower=0)
        )
        df[f"eop {symbol} shares sold"] = (
            -df[f"eop {symbol} shares traded"].clip(upper=0)
        )

    return df


def add_stats(shares_df):
    df = shares_df.copy()

    for symbol in sorted_symbols_list:
        df[f"{symbol} daily commission"] = (
            -df[f"eop {symbol} shares traded"].abs() * commission_per_share
        )

    commission_columns = [f"{s} daily commission" for s in sorted_symbols_list]
    df["daily total commissions"] = df[commission_columns].sum(axis=1)

    for symbol in sorted_symbols_list:
        bop_price = f"bop {symbol} price"
        eop_price = f"eop {symbol} price"

        df[f"{symbol} cop"] = df[eop_price] - df[bop_price]
        df[f"{symbol} pct cop"] = np.log(df[eop_price] / df[bop_price])

        bop_shares = f"bop {symbol} shares"

        df[f"{symbol} daily position pnl"] = (
            df[bop_shares] * df[f"{symbol} cop"]
        )

    pnl_columns = [f"{s} daily position pnl" for s in sorted_symbols_list]
    df["daily position pnl"] = df[pnl_columns].sum(axis=1)

    for symbol in sorted_symbols_list:
        df[f"short {symbol} investment amount"] = 0
        df[f"long {symbol} investment amount"] = 0
        bop_shares = f"bop {symbol} shares"
        df[f"{symbol} investment amount"] = (
            df[bop_shares] * df[f"bop {symbol} price"]
        )
        df[f"short {symbol} investment amount"] = df[
            f"{symbol} investment amount"
        ].clip(upper=0)
        df[f"long {symbol} investment amount"] = df[
            f"{symbol} investment amount"
        ].clip(lower=0)
        df[f"daily {symbol} borrow cost"] = (
            annual_borrow_cost
            * df[f"short {symbol} investment amount"]
            * df["day count"]
            / 360
        )

    inv_columns = [f"{s} investment amount" for s in sorted_symbols_list]
    df["net investment amount"] = df[inv_columns].sum(axis=1)
    df["gross investment amount"] = df[inv_columns].abs().sum(axis=1)

    short_columns = [f"short {s} investment amount" for s in sorted_symbols_list]
    df["short investment amount"] = df[short_columns].sum(axis=1)

    borrow_columns = [f"daily {s} borrow cost" for s in sorted_symbols_list]
    df["daily borrow cost"] = df[borrow_columns].sum(axis=1)

    df["daily net profit"] = (
        df["daily position pnl"]
        + df["daily total commissions"]
        + df["daily borrow cost"]
    )
    df["cumulative net profit"] = df["daily net profit"].cumsum()
    df["drawdown"] = (
        df["cumulative net profit"] - df["cumulative net profit"].cummax()
    )

    return df


def add_summary_stats(stats_df):
    df = stats_df
    traded_columns = [f"eop {s} shares traded" for s in sorted_symbols_list]
    valid_rows = df.index >= moving_avg_days
    daily_profit = df.loc[valid_rows, "daily net profit"]
    average_net_investment = df.loc[valid_rows, "net investment amount"].mean()
    average_gross_investment = df.loc[valid_rows, "gross investment amount"].mean()
    total_profit = daily_profit.sum()
    number_of_days = daily_profit.notna().sum()

    annualized_return = (
        total_profit / average_gross_investment * trading_days_per_year / number_of_days
        if average_gross_investment > 0 and number_of_days > 0
        else np.nan
    )

    daily_std = daily_profit.std()
    annualized_sharpe = (
        daily_profit.mean() / daily_std * np.sqrt(trading_days_per_year)
        if daily_std != 0 and not pd.isna(daily_std)
        else np.nan
    )
    return pd.DataFrame(
        [{
            "symbols": "_".join(sorted_symbols_list),
            "moving avg days": moving_avg_days,

            "total net profit": total_profit,
            "total borrow cost": df.loc[valid_rows, "daily borrow cost"].sum(),
            "average daily short investment": df.loc[valid_rows, "short investment amount"].mean(),
            "average daily net investment": average_net_investment,
            "average daily gross investment": average_gross_investment,
            "annualized return on avg gross investment": annualized_return,
            "annualized Sharpe": annualized_sharpe,
            "maximum drawdown": df["drawdown"].min(),
            "position changes": df["new position"].ne(df["current position"]).sum(),
            "total shares traded": df[traded_columns].abs().sum().sum(),
        }]
    )


async def main():

    report("Starting full backtest pipeline")

    closes_df = pd.read_csv(prices_path, index_col="date", parse_dates=["date"])
    closes_df = closes_df[sorted_symbols_list]
    report(f"Downloaded {len(closes_df):,} completed price rows")
    print(closes_df)
    enhanced_df = enhance_prices(closes_df)
    report(
        f"Enhanced-price calculations and date filtering complete: "
        f"{len(enhanced_df):,} rows remain"
    )
    moving_avg_df = calculate_moving_avg(enhanced_df)
    report("Moving avg calculations complete")
    position_base_df = moving_avg_df

    completed_results = []
    summary_dfs = []

  
    completed_count = 0
    filename_root = f"{'_'.join(sorted_symbols_list)}_{moving_avg_days}"

    position_df = add_positions(position_base_df)
    shares_df = add_shares(position_df)
    final_df = add_stats(shares_df)
    summary_dfs.append(add_summary_stats(final_df))
    filename = (
        f"{filename_root}_.csv"
    )
    completed_results.append((filename, final_df))

    report("All calculations complete; beginning final save")
    output_directory.mkdir(parents=True, exist_ok=True)

    '''
    for save_count, (filename, final_df) in enumerate(completed_results, start=1):
    '''

    final_df.to_csv(output_directory / filename)
    

    summary_output_directory.mkdir(parents=True, exist_ok=True)
    summary_df = pd.concat(summary_dfs, ignore_index=True)
    summary_filename = f"{filename_root}_summary.csv"
    summary_df.to_csv(summary_output_directory / summary_filename, index=False)
    report(f"Saved consolidated summary: {summary_filename}")

    
    

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
