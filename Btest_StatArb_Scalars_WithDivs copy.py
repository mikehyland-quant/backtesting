from pathlib import Path

import numpy as np
import pandas as pd
import xlwings as xw


PRICES_FILENAME = "NOV 5.csv"
SYMBOL_GROUPS = (
    ("VPU", "IDU", "FUTY", "XLU"),
)

MOVING_AVERAGE_WINDOW = 10
INDICATOR_THRESHOLD = 0.20

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-10-25")

COMM_PER_SH = 0.0055
DIVIDEND_MONTHS = range(1, END_DATE.month + 1)

BACKTESTING_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = BACKTESTING_DIRECTORY.parent

PRICES_PATH = BACKTESTING_DIRECTORY / "historical prices" / PRICES_FILENAME
DATABASE_PATH = (
    PROJECT_DIRECTORY / "trading" / "spreadsheets" / "2026 Fin Inst Database.xlsx"
)


def read_excel_table(
    workbook_path: Path,
    sheet_name: str,
    table_name: str,
) -> pd.DataFrame:
    
    workbook = xw.Book(workbook_path)
    table = workbook.sheets[sheet_name].tables[table_name]
    return table.range.options(pd.DataFrame, header=1, index=False).value


def get_dividends(
    frame: pd.DataFrame,
    symbols: tuple[str, ...],
    dividend_rows: pd.DataFrame,
) -> None:
    
    for symbol in symbols:
        row = dividend_rows.loc[symbol]
        dividend_output_column = f"{symbol} div"
        frame[dividend_output_column] = 0.0

        for month in DIVIDEND_MONTHS:
            get_dividend_column = f"div 2026-{month:02d}"
            get_ex_date_column = f"{get_dividend_column} ex-date"

            ex_date = pd.to_datetime(row[get_ex_date_column], errors="coerce")
            amount = pd.to_numeric(row[get_dividend_column], errors="coerce")

            if pd.isna(ex_date) or pd.isna(amount):
                continue

            frame.loc[frame["date"].eq(ex_date), dividend_output_column] = float(amount)


def build_base_df(
    prices: pd.DataFrame,
    dividend_rows: pd.DataFrame,
    symbols: tuple[str, ...],
) -> tuple[tuple[str, ...], pd.DataFrame]:
    
    base_df = prices.loc[:, ["date", *symbols]].copy()
    get_dividends(base_df, symbols, dividend_rows)

    df_dict = {}
    for symbol in symbols:
        df = base_df[[symbol, f"{symbol} div"]].copy()
        df[f"{symbol} + div"] = df[symbol] + df[f"{symbol} div"]
        df[f"{symbol} cum div"] = df[f"{symbol} div"].cumsum()
        df[f"{symbol} + cum div"] = df[symbol] + df[f"{symbol} cum div"]
        df[f"{symbol} daily rtn"] = np.log(df[f"{symbol} + div"] / df[symbol].shift(1))
        df[f"{symbol} comm rtn"] = np.log((df[symbol] - COMM_PER_SH) / df[symbol])
        df_dict[symbol] = df

    ordered_symbols = tuple(sorted(
        df_dict,
        key=lambda symbol: df_dict[symbol][f"{symbol} + cum div"].iloc[-1],
    ))

    big_df = pd.concat(
        [base_df[["date"]], *(df_dict[symbol] for symbol in ordered_symbols)],
        axis=1,
    )

    return ordered_symbols, big_df


def calc_unit_prices(
    symbols: tuple[str, ...],
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    anchor = symbols[-1]

    for symbol in symbols:
        ratio_col = f"{anchor} + cum div/{symbol} + cum div"
        multiplier_col = f"{anchor}/{symbol} mult"
        df[ratio_col] = df[f"{anchor} + cum div"] / df[f"{symbol} + cum div"]
        df[multiplier_col] = df[ratio_col].rolling(MOVING_AVERAGE_WINDOW).mean()
        df[f"{symbol} unit"] = (
            df[multiplier_col].shift(1) * df[f"{symbol} + cum div"]
        )

    return df


def calc_indicator(
    symbols: tuple[str, ...],
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    unit_cols = [f"{symbol} unit" for symbol in symbols]
    rank_cols = [f"{symbol} rank" for symbol in symbols]
    units = df[unit_cols]
    valid = units.notna().any(axis=1)

    df[rank_cols] = units.rank(
        axis=1, method="max", ascending=False
    ).to_numpy()

    df["max unit sym"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["min unit sym"] = pd.Series(pd.NA, index=df.index, dtype="string")

    df.loc[valid, "max unit sym"] = (
        units.loc[valid].idxmax(axis=1).str.removesuffix(" unit")
    )
    df.loc[valid, "min unit sym"] = (
        units.loc[valid].idxmin(axis=1).str.removesuffix(" unit")
    )

    df["max unit price"] = units.max(axis=1)
    df["min unit price"] = units.min(axis=1)
    df["unit range pct"] = np.log(
        df["max unit price"] / df["min unit price"]
    )

    threshold = df["unit range pct"].quantile(1 - INDICATOR_THRESHOLD)
    df["indicator"] = df["unit range pct"] > threshold

    return df


def determine_positions(
    symbols: tuple[str, ...],
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    unit_cols = [f"{sym} unit" for sym in symbols]

    prices = df[unit_cols].to_numpy()
    indicator = df["indicator"].to_numpy(dtype=bool)

    n = len(df)
    long_idx = np.full(n, -1, dtype=int)
    short_idx = np.full(n, -1, dtype=int)

    for i in range(1, n):

        if indicator[i]:
            long_idx[i] = np.argmin(prices[i])
            short_idx[i] = np.argmax(prices[i])

        elif long_idx[i - 1] != -1:

            prev_long = long_idx[i - 1]
            prev_short = short_idx[i - 1]

            if prices[i, prev_short] > prices[i, prev_long]:
                long_idx[i] = prev_long
                short_idx[i] = prev_short

    df["strat eod pos"] = long_idx != -1

    symbols_array = np.array(symbols, dtype=object)

    df["short sym"] = np.where(
        short_idx >= 0,
        symbols_array[np.maximum(short_idx, 0)],
        None
    )

    df["long sym"] = np.where(
        long_idx >= 0,
        symbols_array[np.maximum(long_idx, 0)],
        None
    )

    for symbol in symbols:
        df[f"{symbol} eod pos"] = np.select(
            [df["short sym"].eq(symbol), df["long sym"].eq(symbol)],
            [-1, 1],
            default=0,
        )

    for symbol in symbols:
        df[f"{symbol} bod pos"] = df[f"{symbol} eod pos"].shift(1)

    return df


def calc_rtns(
    symbols: tuple[str, ...],
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()
    elapsed_years = df["date"].diff().dt.days / 360

    # do not combine for statements so column order is preserved
    for symbol in symbols:
        pos = df[f"{symbol} bod pos"]
        df[f"{symbol} trade exec"] = pos.ne(pos.shift(-1, fill_value=0))

    for symbol in symbols:
        df[f"{symbol} strat comm"] = df[f"{symbol} trade exec"] * df[f"{symbol} comm rtn"]

    for symbol in symbols:
        df[f"{symbol} strat rtn"] = df[f"{symbol} bod pos"] * df[f"{symbol} daily rtn"]

    for symbol in symbols:
        df[f"{symbol} borrow"] = (
            (df[f"{symbol} bod pos"].shift(1) == -1) *
            elapsed_years * 0.01
        )

    for symbol in symbols:
        df[f"{symbol} tot strat rtn"] = (
            df[f"{symbol} strat rtn"] + df[f"{symbol} strat comm"] + df[f"{symbol} borrow"]
        )

    df["strat tot rtn"] = df[[f"{symbol} tot strat rtn" for symbol in symbols]].sum(axis=1)
    df["strat cum rtn"] = df["strat tot rtn"].cumsum()

    return df



def main() -> None:

    xlsx_folder = Path(r"C:\Users\micha\OneDrive\My Trading Data")
    template_file = xlsx_folder / "MULTI TEMPLATE.xlsx"

    if not template_file.is_file():
        raise FileNotFoundError(f"Template does not exist: {template_file}")

    prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    prices = prices.loc[
        prices["date"].between(START_DATE, END_DATE)
    ].copy()

    dividends = read_excel_table(
        DATABASE_PATH,
        sheet_name="Equity Inputs",
        table_name="equity_inputs",
    )
    dividends = dividends.set_index("my_fi_name")

    with xw.App(visible=True, add_book=False) as app:
        for symbols in SYMBOL_GROUPS:
            current = tuple(symbol.upper() for symbol in symbols)
            current_name = "_".join(current)

            print(f"Processing: {current_name}", flush=True)
            ordered_symbols, df = build_base_df(prices, dividends, current)
            df = calc_unit_prices(ordered_symbols, df)

            
            df = calc_indicator(ordered_symbols, df)
            df = determine_positions(ordered_symbols, df)
            df = calc_rtns(ordered_symbols, df)

            wb = app.books.open(str(template_file))
            try:
                wb.sheets["input"].range("A1").options(
                    index=False,
                    header=True,
                ).value = df

                app.calculate()

                filename = "_".join(ordered_symbols) + ".xlsx"
                output_file = xlsx_folder / filename
                wb.save(str(output_file))
                print(f"Saved: {output_file}", flush=True)
            finally:
                wb.close()


if __name__ == "__main__":
    main()
