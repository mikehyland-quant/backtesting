from pathlib import Path

import numpy as np
import pandas as pd
from sympy import symbols
import xlwings as xw

SYMBOL_PAIRS = (  # all of the symbol pairs should be (non-anchor, anchor)
    ("CWB", "ICVT"),
    ("BWX", "IGOV"),
    ("VWOB", "EMB"),
    ("CMF", "VTEC"),
    ("PFFD", "PFF"),
    ("BNDX", "IAGG"),    
    ("FALN", "ANGL"),    
   # ("FUTY", "VPU"),         
)

DIVIDEND_METHOD = 2  # 0 = SMA, 1 = ADD 1ST DIV BETW EX-DATES, 2 = CUMULATIVE DIVS

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-08-31")
MOVING_AVERAGE_WINDOWS = (10, 20)
DIVIDEND_MONTHS = range(1, 10)


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


def calc_log_rtns(
        frame: pd.DataFrame,
        symbols: tuple[str, str],
    ) -> None:
    for symbol in symbols:
        frame[f"{symbol} log rtn"] = np.log(frame[symbol] / frame[symbol].shift(1))


def get_dividends(
        frame: pd.DataFrame,
        symbols: tuple[str, str],
        dividend_rows: pd.DataFrame,
    ) -> None:

    rows = {symbol: dividend_rows.loc[symbol] for symbol in symbols}

    for symbol, row in rows.items():
        dividend_output_column = f"{symbol} div"
        frame[dividend_output_column] = 0.0

        for month in DIVIDEND_MONTHS:
            get_dividend_column = f"div 2026-{month:02d}"
            get_ex_date_column = f"{get_dividend_column} ex-date"

            ex_date = pd.to_datetime(row[get_ex_date_column], errors="coerce")
            amount = pd.to_numeric(row[get_dividend_column], errors="coerce")

            if pd.isna(ex_date) or pd.isna(amount):
                continue

            frame.loc[frame["date"].eq(ex_date), dividend_output_column,] = float(amount)

        frame[f"{symbol} div cumsum"] = frame[f"{symbol} div"].cumsum()

    
def add_dividends(
        frame: pd.DataFrame,
        symbols: tuple[str, str],
        dividend_rows: pd.DataFrame,
    ) -> None:

    for symbol in symbols:
        frame[f"{symbol} div adjustment"] = 0.0
            
    if DIVIDEND_METHOD == 1:
        rows = {symbol: dividend_rows.loc[symbol] for symbol in symbols}

        for month in DIVIDEND_MONTHS:
            dividend_column = f"div 2026-{month:02d}"
            ex_date_column = f"{dividend_column} ex-date"
            ex_dates = {symbol: pd.to_datetime(row[ex_date_column], errors="coerce")
                        for symbol, row in rows.items()}
            
            earlier_symbol = (symbols[0] if ex_dates[symbols[0]] < ex_dates[symbols[1]] else 
                              symbols[1])
            later_symbol = next(symbol for symbol in symbols if symbol != earlier_symbol)

            amount = pd.to_numeric(
                rows[earlier_symbol][dividend_column], errors="coerce")

            if pd.isna(amount):
                continue

            mask = frame["date"].between(ex_dates[earlier_symbol], 
                                         ex_dates[later_symbol], inclusive="left")
            frame.loc[mask, f"{earlier_symbol} div adjustment"] = float(amount)

    elif DIVIDEND_METHOD == 2:
        for symbol in symbols:
            frame[f"{symbol} div adjustment"] = frame[f"{symbol} div cumsum"]

    for symbol in symbols:
        frame[f"{symbol}*"] = frame[symbol] + frame[f"{symbol} div adjustment"]

        
def add_ratio_columns(
            frame: pd.DataFrame,
            symbols: tuple[str, str],
        ) -> None:

    for marker in ("", "*"):
        non_anchor = f"{symbols[0]}{marker}"
        anchor = f"{symbols[1]}{marker}"       
        frame[f"{anchor}/{non_anchor}"] = (
            frame[f"{anchor}"] / frame[f"{non_anchor}"])

        frame[f"{anchor}/{non_anchor} avg"] = frame[f"{anchor}/{non_anchor}"].mean()

        for window in MOVING_AVERAGE_WINDOWS:
            frame[f"{anchor}/{non_anchor} {window} dma"] = frame[f"{anchor}/{non_anchor}"].rolling(window).mean()

        frame[f"{non_anchor} x avg"] = (
            frame[f"{non_anchor}"] * frame[f"{anchor}/{non_anchor} avg"].shift(1))
        
        for window in MOVING_AVERAGE_WINDOWS:
            frame[f"{non_anchor} x {window} dma"] = (
                frame[f"{non_anchor}"] * frame[f"{anchor}/{non_anchor} {window} dma"].shift(1))

        frame[f'log diff {non_anchor} x avg / {anchor}'] = np.log(
            frame[f"{non_anchor} x avg"] / frame[f"{anchor}"])

        for window in MOVING_AVERAGE_WINDOWS:
            frame[f"log diff {non_anchor} x {window} dma / {anchor}"] = np.log(
                frame[f"{non_anchor} x {window} dma"] / frame[f"{anchor}"])


def build_pair_backtest(
            prices: pd.DataFrame,
            dividend_rows: pd.DataFrame,
            symbols: tuple[str, str],
        ) -> pd.DataFrame:
    if len(symbols) != 2:
        raise ValueError(f"Each symbol group must contain two symbols: {symbols}")

    frame = prices.loc[:, ["date", *symbols]].copy()

    calc_log_rtns(frame, symbols)
    get_dividends(frame, symbols, dividend_rows)
    add_dividends(frame, symbols, dividend_rows)    
    add_ratio_columns(frame, symbols)
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
        output_path = OUTPUT_DIRECTORY / f"{'_'.join(symbols)}_{DIVIDEND_METHOD}.csv"
        backtest.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
