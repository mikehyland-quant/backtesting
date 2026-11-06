from pathlib import Path

import pandas as pd
import xlwings as xw

xlsx_folder = Path(r"C:\Users\micha\OneDrive\My Trading Data")

pairs = [

'PGX_PFFD2',
'PGX_PFF2',
'PFFD_PFF2' 

]

df = pd.DataFrame()

with xw.App(visible=True, add_book=False) as app:
    for pair in pairs:
        workbook_path = xlsx_folder / f"{pair}.xlsx"

        if not workbook_path.exists():
            print(f"Missing workbook: {workbook_path.name}", flush=True)
            continue

        wb = app.books.open(str(workbook_path))
        try:
            values = (
                wb.sheets["stats"]
                .range("o3:ac3")
                .options(ndim=1)
                .value
            )
            df[pair] = values
        finally:
            wb.close()

output_file = xlsx_folder / "temp.xlsx"
df.to_excel(output_file, index=False)
print(f"Saved: {output_file}", flush=True)