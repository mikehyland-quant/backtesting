from pathlib import Path

import numpy as np
import pandas as pd
import xlwings as xw

SYMBOL_PAIRS = (
    ("VGT", "FTEC"),

    ("FSTA", "VDC"),
    ("FENY", "VDE"),
    ("FHLC", "VHT"),
    ("FREL", "VNQ"),
    ("FUTY", "VPU"),

    ("XLE", "IYE"),
    ("XLF", "IYF"),
    ("XLRE", "IYR"),
    ("XLK", "IYW"),
    ("XLU", "IDU"),

    ("IYH", "XLV"),
)

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-08-15")
MOVING_AVERAGE_WINDOWS = (10, 20)
DIVIDEND_MONTHS = range(1, 9)


BACKTESTING_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = BACKTESTING_DIRECTORY.parent

PRICES_PATH = BACKTESTING_DIRECTORY / "historical prices" / "sectors.csv"
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


def add_ratio_columns(
            frame: pd.DataFrame,
            non_anchor: str,
            anchor: str,
            *,
            adjusted: bool = False,
        ) -> None:
    marker = "*" if adjusted else ""
    non_anchor_column = f"{non_anchor}{marker}"
    anchor_column = f"{anchor}{marker}"
    ratio_name = f"{anchor_column}/{non_anchor_column}"

    frame[ratio_name] = frame[anchor_column] / frame[non_anchor_column]
    frame[f"{ratio_name} avg"] = frame[ratio_name].mean()

    for window in MOVING_AVERAGE_WINDOWS:
        frame[f"{ratio_name} {window} dma"] = frame[ratio_name].rolling(window).mean()

    frame[f"{non_anchor_column} avg"] = (
        frame[non_anchor_column] * frame[f"{ratio_name} avg"].shift(1)
    )
    for window in MOVING_AVERAGE_WINDOWS:
        frame[f"{non_anchor_column} {window} dma"] = (
            frame[non_anchor_column] * frame[f"{ratio_name} {window} dma"].shift(1)
        )

    diff_marker = "*" if adjusted else ""

    frame[f"avg diff{diff_marker}"] = np.log(
        frame[anchor_column] / frame[f"{non_anchor_column} avg"]
    )
    for window in MOVING_AVERAGE_WINDOWS:
        frame[f"{window} dma diff{diff_marker}"] = np.log(
            frame[anchor_column]
            / frame[f"{non_anchor_column} {window} dma"]
        )


def add_dividends(
            frame: pd.DataFrame,
            symbols: tuple[str, str],
            dividend_rows: pd.DataFrame,
        ) -> None:
    for symbol in symbols:
        frame[f"{symbol} div"] = 0.0

    rows = {symbol: dividend_rows.loc[symbol] for symbol in symbols}

    for month in DIVIDEND_MONTHS:
        dividend_column = f"div 2026-{month:02d}"
        ex_date_column = f"{dividend_column} ex-date"
        ex_dates = {
            symbol: pd.to_datetime(row[ex_date_column], errors="coerce")
            for symbol, row in rows.items()
        }

        if any(pd.isna(ex_date) for ex_date in ex_dates.values()):
            continue

        earlier_symbol = (
            symbols[0] if ex_dates[symbols[0]] < ex_dates[symbols[1]] else symbols[1]
        )
        later_symbol = next(symbol for symbol in symbols if symbol != earlier_symbol)
        amount = pd.to_numeric(
            rows[earlier_symbol][dividend_column], errors="coerce"
        )

        if pd.isna(amount):
            continue

        mask = frame["date"].between(
            ex_dates[earlier_symbol], ex_dates[later_symbol], inclusive="left"
        )
        frame.loc[mask, f"{earlier_symbol} div"] = float(amount)


def build_pair_backtest(
            prices: pd.DataFrame,
            dividend_rows: pd.DataFrame,
            symbols: tuple[str, str],
        ) -> pd.DataFrame:
    if len(symbols) != 2:
        raise ValueError(f"Each symbol group must contain two symbols: {symbols}")

    non_anchor, anchor = symbols
    frame = prices.loc[:, ["date", *symbols]].copy()

    add_ratio_columns(frame, non_anchor, anchor)
    add_dividends(frame, symbols, dividend_rows)

    for symbol in symbols:
        frame[f"{symbol}*"] = frame[symbol] + frame[f"{symbol} div"]

    add_ratio_columns(frame, non_anchor, anchor, adjusted=True)
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
