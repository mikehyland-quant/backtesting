from pathlib import Path

import numpy as np
import pandas as pd
import xlwings as xw

SYMBOL_PAIRS = (
    ("CWB", "ICVT"),

    ("BWX", "IGOV"),
    ("VWOB", "EMB"),
    ("CMF", "VTEC"),
    ("PFFD", "PFF"),
    ("BNDX", "IAGG"),
    ("FALN", "ANGL"),
    
)

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-08-15")
MOVING_AVERAGE_WINDOWS = (10, 20)
DIVIDEND_MONTHS = range(1, 9)


BACKTESTING_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = BACKTESTING_DIRECTORY.parent

PRICES_PATH = BACKTESTING_DIRECTORY / "historical prices" / "stat arb prices.csv"
DATABASE_PATH = (
    PROJECT_DIRECTORY / "trading" / "spreadsheets" / "2026 Fin Inst Database.xlsx"
)
OUTPUT_DIRECTORY = BACKTESTING_DIRECTORY / "backtests"



def read_excel_table(
            workbook_path: Path,
            sheet_name: str,
            table_name: str,
        ) -> pd.DataFrame:
    workbook = xw.Book(workbook_path)
    table = workbook.sheets[sheet_name].tables[table_name]
    return table.range.options(pd.DataFrame, header=1, index=False).value


def add_rolling_ratio_signals(
        frame: pd.DataFrame,
        non_anchor: str,
        anchor: str,
    ) -> None:
    ratio_name = f"{anchor}/{non_anchor}"

    frame[ratio_name] = frame[anchor] / frame[non_anchor]
    frame[f"{ratio_name} avg"] = frame[ratio_name].mean()

    for window in MOVING_AVERAGE_WINDOWS:
        frame[f"{ratio_name} {window} dma"] = frame[ratio_name].rolling(window).mean()

    frame[f"{non_anchor} avg"] = (
        frame[non_anchor] * frame[f"{ratio_name} avg"].shift(1)
    )
    for window in MOVING_AVERAGE_WINDOWS:
        frame[f"{non_anchor} {window} dma"] = (
            frame[non_anchor] * frame[f"{ratio_name} {window} dma"].shift(1)
        )

    frame["avg diff"] = np.log(
        frame[anchor] / frame[f"{non_anchor} avg"]
    )
    for window in MOVING_AVERAGE_WINDOWS:
        frame[f"{window} dma diff"] = np.log(
            frame[anchor] / frame[f"{non_anchor} {window} dma"]
        )


def add_monthly_adjusted_ratio_signal(
        frame: pd.DataFrame,
        non_anchor: str,
        anchor: str,
    ) -> None:
    non_anchor_column = f"{non_anchor}*"
    anchor_column = f"{anchor}*"
    ratio_name = f"{anchor_column}/{non_anchor_column}"

    frame[ratio_name] = frame[anchor_column] / frame[non_anchor_column]

    months = pd.to_datetime(frame["date"]).dt.to_period("M")
    monthly_averages = frame[ratio_name].groupby(months).mean()
    frame["mult"] = months.map(monthly_averages.shift(1))

    scaled_non_anchor_column = f"{non_anchor} * mult"
    frame[scaled_non_anchor_column] = frame[non_anchor] * frame["mult"]
    frame["avg diff mult"] = np.log(
        frame[anchor_column] / frame[scaled_non_anchor_column]
    )


def add_dividends(
        frame: pd.DataFrame,
        non_anchor: str,
        anchor: str,
        dividend_rows: pd.DataFrame,
    ) -> None:
    symbols = (non_anchor, anchor)

    for symbol in symbols:
        frame[f"{symbol} prev div"] = 0.0

    rows = {symbol: dividend_rows.loc[symbol] for symbol in symbols}
    frame_months = pd.to_datetime(frame["date"]).dt.to_period("M")

    for month in DIVIDEND_MONTHS:
        dividend_column = f"div 2026-{month:02d}"
        previous_month = pd.Period(
            year=2026, month=month, freq="M"
        ) - 1

        for symbol in symbols:
            amount = pd.to_numeric(
                rows[symbol][dividend_column],
                errors="coerce",
            )

            if pd.isna(amount) or amount == 0:
                continue

            mask = frame_months == previous_month
            frame.loc[mask, f"{symbol} prev div"] = float(amount)


def build_pair_backtest(
            prices: pd.DataFrame,
            dividend_rows: pd.DataFrame,
            symbols: tuple[str, str],
        ) -> pd.DataFrame:
    if len(symbols) != 2:
        raise ValueError(f"Each symbol group must contain two symbols: {symbols}")

    non_anchor, anchor = symbols
    frame = prices.loc[:, ["date", *symbols]].copy()

    add_rolling_ratio_signals(frame, non_anchor, anchor)
    add_dividends(frame, non_anchor, anchor, dividend_rows)

    for symbol in symbols:
        frame[f"{symbol}*"] = frame[symbol] - frame[f"{symbol} prev div"]

    add_monthly_adjusted_ratio_signal(frame, non_anchor, anchor)

    return frame


def main() -> None:
    prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    prices = prices.loc[prices["date"].between(START_DATE, END_DATE)].copy()

    database = read_excel_table(
        DATABASE_PATH,
        sheet_name="Scalar Inputs Table",
        table_name="scalar_inputs_table",
    )
    dividend_rows = database.set_index("symbol")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for symbols in SYMBOL_PAIRS:
        backtest = build_pair_backtest(prices, dividend_rows, symbols)
        output_path = OUTPUT_DIRECTORY / f"{'_'.join(symbols)}.csv"
        backtest.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
