from pathlib import Path

import pandas as pd
import xlwings as xw

csv_folder = Path(r"C:\Users\micha\git-projects\backtesting\backtests")
xlsx_folder = Path(r"C:\Users\micha\OneDrive\My Trading Data")

if not csv_folder.is_dir():
    raise FileNotFoundError(f"CSV folder does not exist: {csv_folder}")

if not xlsx_folder.is_dir():
    raise FileNotFoundError(f"Excel folder does not exist: {xlsx_folder}")

template_file = xlsx_folder / "template oct 25.xlsx"
if not template_file.is_file():
    raise FileNotFoundError(f"Template does not exist: {template_file}")

csv_files = [
'IYC_VCR.csv',
'IYE_VDE.csv',
'IYF_VFH.csv',
'IYH_VHT.csv',
'IYJ_VIS.csv',
'VNQ_IYR.csv',
'IYK_VDC.csv',
'VGT_IYW.csv',
'IDU_VPU.csv',

]

with xw.App(visible=True, add_book=False) as app:
    for filename in csv_files:
        csv_file = csv_folder / filename

        if not csv_file.is_file():
            print(f"Missing CSV: {csv_file}", flush=True)
            continue

        print(f"Processing: {csv_file.name}", flush=True)
        df_csv = pd.read_csv(csv_file)

        wb = app.books.open(str(template_file))
        try:
            wb.sheets["input"].range("A1").options(
                index=False, header=True
            ).value = df_csv

            app.calculate()

            output_file = xlsx_folder / f"{csv_file.stem}2.xlsx"
            wb.save(str(output_file))
            print(f"Saved: {output_file}", flush=True)
        finally:
            wb.close()