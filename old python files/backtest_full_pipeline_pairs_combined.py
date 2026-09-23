# --- imports ---
import asyncio
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS - edit these values as needed
# Keep the anchor symbol last in the list.
# ============================================================
sorted_symbols_list = ["SPMB", "MBB"]
prices_path = Path(__file__).resolve().parent / "historical prices" / "stat arb prices.csv"

start_date = pd.Timestamp("2024-01-02")
end_date = pd.Timestamp("2024-12-15")

moving_avg_days = 20

# Fixed combined-strategy hurdles.
# Long signals are normally negative; short signals are normally positive.
long_entry_hurdle = -0.0001 * 10
long_exit_hurdle = -0.0001 * 5
short_entry_hurdle = 0.0001 * 10
short_exit_hurdle = 0.0001 * 5

ibkr_port = 7496
lookback_period = "5 Y"
length_of_each_period = "1 day"
use_regular_trading_hours = True
prices_to_use = "TRADES"

annual_borrow_cost = 0.01  # 1.00% annualized; change as needed
commission_per_share = 0.005
dollar_constant = 100_000
trading_days_per_year = 252
output_directory = Path("backtests")
summary_output_directory = Path("summary_stats")


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


def calculate_unit_prices(enhanced_df):
    df = enhanced_df.copy()
    anchor = sorted_symbols_list[-1]

    for symbol in sorted_symbols_list:
        ratio_column = f"eop {anchor} / eop {symbol}"
        eop_average_column = f"eop {anchor} / {symbol} moving avg"
        bop_average_column = eop_average_column.replace("eop", "bop")
        df[ratio_column] = df[f"eop {anchor} price"] / df[f"eop {symbol} price"]
        df[bop_average_column] = 0
        df[eop_average_column] = df[ratio_column].rolling(moving_avg_days).mean()
        df[bop_average_column] = df[eop_average_column].shift(1)
        df[f"eop {symbol} unit price"] = (
            df[bop_average_column] * df[f"eop {symbol} price"]
        )

    for symbol in sorted_symbols_list:
        df[f"{symbol} unit price pct diff"] = np.log(
            df[f"eop {symbol} unit price"] / df[f"eop {anchor} unit price"]
        )

    non_anchor = sorted_symbols_list[0]
    df["unit price pct diff"] = (
        df[f"{non_anchor} unit price pct diff"]
        - df[f"{anchor} unit price pct diff"]
    )
    return df


def add_positions(base_df):
    df = base_df.copy()

    df["go long"] = df["unit price pct diff"] < long_entry_hurdle
    df["exit long"] = df["unit price pct diff"] > long_exit_hurdle
    df["go short"] = df["unit price pct diff"] > short_entry_hurdle
    df["exit short"] = df["unit price pct diff"] < short_exit_hurdle

    df["current_position"] = 0
    df["new_position"] = 0

    next_trading_date = df["eop date"].shift(-1)
    df["month end liquidation"] = (
        next_trading_date.notna()
        & (df["eop date"].dt.to_period("M") != next_trading_date.dt.to_period("M"))
    )

    current_column = df.columns.get_loc("current_position")
    new_column = df.columns.get_loc("new_position")

    for row_index in range(moving_avg_days, len(df)):
        current_position = df.iat[row_index - 1, new_column]
        new_position = current_position

        if df["month end liquidation"].iat[row_index]:
            new_position = 0

        elif current_position == 0:
            if df["go long"].iat[row_index]:
                new_position = 1
            elif df["go short"].iat[row_index]:
                new_position = -1

        elif current_position == 1:
            # A short-entry signal takes priority over a long-exit signal,
            # allowing a same-close long -> short reversal.
            if df["go short"].iat[row_index]:
                new_position = -1
            elif df["exit long"].iat[row_index]:
                new_position = 0

        elif current_position == -1:
            # A long-entry signal takes priority over a short-exit signal,
            # allowing a same-close short -> long reversal.
            if df["go long"].iat[row_index]:
                new_position = 1
            elif df["exit short"].iat[row_index]:
                new_position = 0

        df.iat[row_index, current_column] = current_position
        df.iat[row_index, new_column] = new_position

    df["long entry"] = (df["new_position"] == 1) & (df["current_position"] != 1)
    df["short entry"] = (df["new_position"] == -1) & (df["current_position"] != -1)
    df["long to short flip"] = (
        (df["current_position"] == 1) & (df["new_position"] == -1)
    )
    df["short to long flip"] = (
        (df["current_position"] == -1) & (df["new_position"] == 1)
    )

    return df


def add_shares(position_df):
    df = position_df.copy()

    non_anchor = sorted_symbols_list[0]
    anchor = sorted_symbols_list[-1]
    anchor_average = f"bop {anchor} / {anchor} moving avg"
    anchor_shares = (
        df[anchor_average] * dollar_constant / df[f"eop {anchor} price"]
    ).round()

    for symbol in sorted_symbols_list:
        shares_per_unit = f"{symbol} shares per unit"
        prior_average = f"bop {anchor} / {symbol} moving avg"
        df[shares_per_unit] = (anchor_shares * df[prior_average]).round()
        leg_sign = 1 if symbol == non_anchor else -1

        bop_shares = f"bop {symbol} shares"
        eop_shares = f"eop {symbol} shares"
        df[eop_shares] = df["new_position"] * leg_sign * df[shares_per_unit]
        df[bop_shares] = df[eop_shares].shift(1)
        df.iat[moving_avg_days, df.columns.get_loc(bop_shares)] = 0

        df[f"eop {symbol} shares traded"] = df[eop_shares] - df[bop_shares]
        df[f"eop {symbol} shares bought"] = df[f"eop {symbol} shares traded"].clip(lower=0)
        df[f"eop {symbol} shares sold"] = -df[f"eop {symbol} shares traded"].clip(upper=0)

    return df


def add_stats(shares_df):
    df = shares_df.copy()

    profit_columns = []
    commission_columns = []
    investment_columns = []
    short_columns = []
    borrow_columns = []

    for symbol in sorted_symbols_list:
        bop_shares = f"bop {symbol} shares"

        pnl_column = f"{symbol} daily position pnl"
        commission_column = f"{symbol} daily commission"
        investment_column = f"{symbol} investment amount"
        short_column = f"short {symbol} investment amount"
        borrow_column = f"daily {symbol} borrow cost"

        df[f"{symbol} cop"] = df[f"eop {symbol} price"] - df[f"bop {symbol} price"]
        df[f"{symbol} pct cop"] = np.log(df[f"eop {symbol} price"] / df[f"bop {symbol} price"])
        df[pnl_column] = df[bop_shares] * df[f"{symbol} cop"]
        df[commission_column] = -df[f"eop {symbol} shares traded"].abs() * commission_per_share
        df[investment_column] = df[bop_shares] * df[f"bop {symbol} price"]
        df[short_column] = df[investment_column].clip(upper=0)
        df[f"long {symbol} investment amount"] = df[investment_column].clip(lower=0)
        df[borrow_column] = annual_borrow_cost * df[short_column] * df["day count"] / 360

        profit_columns.append(pnl_column)
        commission_columns.append(commission_column)
        investment_columns.append(investment_column)
        short_columns.append(short_column)
        borrow_columns.append(borrow_column)

    df["daily position pnl"] = df[profit_columns].sum(axis=1)
    df["daily total commissions"] = df[commission_columns].sum(axis=1)
    df["short investment amount"] = df[short_columns].sum(axis=1)
    df["daily borrow cost"] = df[borrow_columns].sum(axis=1)

    # Attribute gross trading P&L to the economic position held during the day.
    df["long daily position pnl"] = np.where(
        df["current_position"] == 1, df["daily position pnl"], 0.0
    )
    df["short daily position pnl"] = np.where(
        df["current_position"] == -1, df["daily position pnl"], 0.0
    )

    df["daily net profit"] = (
        df["daily position pnl"]
        + df["daily total commissions"]
        + df["daily borrow cost"]
    )
    df["cumulative net profit"] = df["daily net profit"].cumsum()
    df["drawdown"] = df["cumulative net profit"] - df["cumulative net profit"].cummax()

    df["gross investment amount"] = df[investment_columns].abs().sum(axis=1)
    df["net investment amount"] = df[investment_columns].sum(axis=1)
    return df


def add_summary_stats(stats_df):
    df = stats_df
    valid_rows = df.index >= moving_avg_days
    traded_columns = [f"eop {s} shares traded" for s in sorted_symbols_list]

    daily_net_profit = df.loc[valid_rows, "daily net profit"]
    average_net_investment = df.loc[valid_rows, "net investment amount"].mean()
    average_gross_investment = df.loc[valid_rows, "gross investment amount"].mean()
    total_net_profit = daily_net_profit.sum()
    number_of_days = daily_net_profit.notna().sum()

    annualized_return = (
        total_net_profit / average_gross_investment * trading_days_per_year / number_of_days
        if average_gross_investment > 0 and number_of_days > 0
        else np.nan
    )

    daily_std = daily_net_profit.std()
    annualized_sharpe = (
        daily_net_profit.mean() / daily_std * np.sqrt(trading_days_per_year)
        if daily_std != 0 and not pd.isna(daily_std)
        else np.nan
    )

    return pd.DataFrame([{
        "symbols": "_".join(sorted_symbols_list),
        "moving avg days": moving_avg_days,
        "long entry hurdle": long_entry_hurdle,
        "long exit hurdle": long_exit_hurdle,
        "short entry hurdle": short_entry_hurdle,
        "short exit hurdle": short_exit_hurdle,
        "long position pnl": df.loc[valid_rows, "long daily position pnl"].sum(),
        "short position pnl": df.loc[valid_rows, "short daily position pnl"].sum(),
        "total position pnl": df.loc[valid_rows, "daily position pnl"].sum(),
        "total commissions": df.loc[valid_rows, "daily total commissions"].sum(),
        "total borrow cost": df.loc[valid_rows, "daily borrow cost"].sum(),
        "total net profit": total_net_profit,
        "average daily short investment": df.loc[valid_rows, "short investment amount"].mean(),
        "average daily net investment": average_net_investment,
        "average daily gross investment": average_gross_investment,
        "annualized return on avg gross investment": annualized_return,
        "annualized Sharpe": annualized_sharpe,
        "maximum drawdown": df.loc[valid_rows, "drawdown"].min(),
        "position changes": df.loc[valid_rows, "new_position"].ne(df.loc[valid_rows, "current_position"]).sum(),
        "long entries": df.loc[valid_rows, "long entry"].sum(),
        "short entries": df.loc[valid_rows, "short entry"].sum(),
        "long to short flips": df.loc[valid_rows, "long to short flip"].sum(),
        "short to long flips": df.loc[valid_rows, "short to long flip"].sum(),
        "total shares traded": df.loc[valid_rows, traded_columns].abs().sum().sum(),
    }])


async def main():
    if len(sorted_symbols_list) != 2:
        raise ValueError("This pipeline requires exactly two symbols")
    if not (long_entry_hurdle <= long_exit_hurdle):
        raise ValueError("long_entry_hurdle should be <= long_exit_hurdle")
    if not (short_entry_hurdle >= short_exit_hurdle):
        raise ValueError("short_entry_hurdle should be >= short_exit_hurdle")

    report("Starting combined long/short backtest")
    closes_df = pd.read_csv(prices_path, index_col="date", parse_dates=["date"])
    closes_df = closes_df[sorted_symbols_list]
    report(f"Downloaded {len(closes_df):,} completed price rows")

    enhanced_df = enhance_prices(closes_df)
    report(f"Enhanced-price calculations complete: {len(enhanced_df):,} rows remain")

    unit_price_df = calculate_unit_prices(enhanced_df)
    position_df = add_positions(unit_price_df)
    shares_df = add_shares(position_df)
    final_df = add_stats(shares_df)
    summary_df = add_summary_stats(final_df)
    report("Combined strategy calculations complete")

    filename_root = f"{'_'.join(sorted_symbols_list)}_{moving_avg_days}_combined"
    detail_filename = (
        f"{filename_root}_"
        f"L_{long_entry_hurdle:.4f}_{long_exit_hurdle:.4f}_"
        f"S_{short_entry_hurdle:.4f}_{short_exit_hurdle:.4f}.csv"
    )
    summary_filename = f"{filename_root}_summary.csv"

    output_directory.mkdir(parents=True, exist_ok=True)
    summary_output_directory.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_directory / detail_filename, index=False)
    summary_df.to_csv(summary_output_directory / summary_filename, index=False)

    report(f"Saved detail file: {detail_filename}")
    report(f"Saved summary file: {summary_filename}")
    report("Combined long/short backtest finished successfully")

    print("\nSUMMARY")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
