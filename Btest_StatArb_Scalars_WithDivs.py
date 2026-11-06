from pathlib import Path

import numpy as np
import pandas as pd
import xlwings as xw

SYMBOL_GROUPS = (  # Each pair is (lo/non-anchor, hi/anchor).

('KBE','KRE'),

)


PRICES_FILENAME = "NOV 3.csv"

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-10-25")
MOVING_AVERAGE_WINDOWS = (10, 20)
DIVIDEND_MONTHS = range(1, 10)

COMM_PER_SH = 0.0055

BACKTESTING_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = BACKTESTING_DIRECTORY.parent

PRICES_PATH = BACKTESTING_DIRECTORY / "historical prices" / PRICES_FILENAME
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


def get_dividends(
    frame: pd.DataFrame,
    symbols: tuple[str, str],
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

        frame[f"{symbol} cum div"] = frame[dividend_output_column].cumsum()


def calc_adj_dividends(
    frame: pd.DataFrame,
    symbols: tuple[str, str],
    dividend_rows: pd.DataFrame,
) -> None:
    for symbol in symbols:
        frame[f"{symbol} div adj"] = 0.0

    rows = {symbol: dividend_rows.loc[symbol] for symbol in symbols}

    for month in DIVIDEND_MONTHS:
        dividend_column = f"div 2026-{month:02d}"
        ex_date_column = f"{dividend_column} ex-date"
        ex_dates = {
            symbol: pd.to_datetime(row[ex_date_column], errors="coerce")
            for symbol, row in rows.items()
        }

        lo_sym, hi_sym = symbols
        earlier_symbol, later_symbol = (
            (lo_sym, hi_sym) if ex_dates[lo_sym] < ex_dates[hi_sym] else (hi_sym, lo_sym)
        )

        amount = pd.to_numeric(rows[earlier_symbol][dividend_column], errors="coerce")

        if pd.isna(amount):
            continue

        mask = frame["date"].between(
            ex_dates[earlier_symbol], ex_dates[later_symbol], inclusive="left"
        )
        frame.loc[mask, f"{earlier_symbol} div adj"] = float(amount)


def calc_prices_with_divs(
    frame: pd.DataFrame,
    symbols: tuple[str, str],
) -> None:
    for symbol in symbols:
        for suffix in ("div", "div adj", "cum div"):
            column = f"{symbol} {suffix}"
            values = frame.pop(column)
            frame.insert(frame.columns.get_loc(symbol), column, values)

        position = frame.columns.get_loc(symbol) + 1
        for offset, (suffix, dividend_suffix) in enumerate(
            (("adj", "div adj"), ("cum", "cum div"))
        ):
            frame.insert(
                position + offset,
                f"{symbol} {suffix}",
                frame[symbol] + frame[f"{symbol} {dividend_suffix}"],
            )


def add_scalar_columns(
    frame: pd.DataFrame,
    symbols: tuple[str, str],
) -> None:
    lo_sym, hi_sym = symbols
    for suffix in ("", " adj", " cum"):
        ratio_column = f"{hi_sym}/{lo_sym}{suffix}"
        ratio = frame[f"{hi_sym}{suffix}"] / frame[f"{lo_sym}{suffix}"]
        frame[ratio_column] = ratio
        frame[f"avg scalar{suffix}"] = ratio.mean()

        for window in MOVING_AVERAGE_WINDOWS:
            frame[f"{window}dma scalar{suffix}"] = ratio.rolling(window).mean()


def add_ratio_columns(
    frame: pd.DataFrame,
    symbols: tuple[str, str],
) -> None:
    lo_sym, hi_sym = symbols
    for suffix in ("", " adj", " cum"):
        for window in MOVING_AVERAGE_WINDOWS:
            scaled_column = f"{lo_sym}{suffix} * {window}dma{suffix}"
            frame[scaled_column] = (
                frame[f"{lo_sym}{suffix}"] * frame[f"{window}dma scalar{suffix}"].shift(1)
            )
            frame[f"ln({lo_sym} * {window}dma){suffix}/{hi_sym}{suffix}"] = np.log(
                frame[scaled_column] / frame[f"{hi_sym}{suffix}"]
            )


def calc_log_rtns(
    frame: pd.DataFrame,
    symbols: tuple[str, str],
) -> None:
    for symbol in symbols:
        frame[f"{symbol} log rtn"] = np.log((frame[symbol] + frame[f'{symbol} div']) / frame[symbol].shift(1))
    frame["comm/sh"] = COMM_PER_SH
    for symbol in symbols:
        frame[f"{symbol} comm"] = np.log(
            (frame[symbol] - frame["comm/sh"]) / frame[symbol]
        )
    frame["1pct borrow"] = -0.01 * frame["date"].diff().dt.days / 360


def build_backtest(
    prices: pd.DataFrame,
    dividend_rows: pd.DataFrame,
    symbols: tuple[str, str],
) -> pd.DataFrame:
    if len(symbols) != 2:
        raise ValueError(f"Each symbol group must contain two symbols: {symbols}")

    frame = prices.loc[:, ["date", *symbols]].copy()

    get_dividends(frame, symbols, dividend_rows)
    calc_adj_dividends(frame, symbols, dividend_rows)
    calc_prices_with_divs(frame, symbols)
    add_scalar_columns(frame, symbols)
    add_ratio_columns(frame, symbols)
    calc_log_rtns(frame, symbols)

    lo_sym, hi_sym = symbols
    frame["ex-date"] = (frame[f"{lo_sym} div"] > 0) | (frame[f"{hi_sym} div"] > 0)

    return frame


def main() -> None:
    xlsx_folder = Path(r"C:\Users\micha\OneDrive\My Trading Data")
    template_file = xlsx_folder / "TEMPLATE.xlsx"

    if not template_file.is_file():
        raise FileNotFoundError(f"Template does not exist: {template_file}")

    prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    prices = prices.loc[
        prices["date"].between(START_DATE, END_DATE)
    ].copy()

    database = read_excel_table(
        DATABASE_PATH,
        sheet_name="Equity Inputs",
        table_name="equity_inputs",
    )
    dividend_rows = database.set_index("my_fi_name")

    with xw.App(visible=True, add_book=False) as app:
        for symbols in SYMBOL_GROUPS:
            pair = tuple(symbol.upper() for symbol in symbols)
            pair_name = "_".join(pair)

            print(f"Processing: {pair_name}", flush=True)
            df = build_backtest(prices, dividend_rows, pair)

            wb = app.books.open(str(template_file))
            try:
                wb.sheets["input"].range("A1").options(
                    index=False,
                    header=True,
                ).value = df

                app.calculate()

                output_file = xlsx_folder / f"{pair_name}2.xlsx"
                wb.save(str(output_file))
                print(f"Saved: {output_file}", flush=True)
            finally:
                wb.close()


if __name__ == "__main__":
    main()
