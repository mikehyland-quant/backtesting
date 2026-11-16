from pathlib import Path

import pandas as pd
import xlwings as xw


xlsx_folder_suffix = Path("CORPS 3")

pairs = [

('BF B', 'BF A'),
('BRK B', 'BRK A'),
('FOX', 'FOXA'),
('FWONA', 'FWONK'),
('GLIBK', 'GLIBA'),
('GOOG', 'GOOGL'),
('HEI A', 'HEI'),
('LBTYK', 'LBTYA'),
('LEN B', 'LEN'),
('LILAK', 'LILA'),
('MOG A', 'MOG B'),
('NWSA', 'NWS'),
('UA', 'UAA'),
('Z', 'ZG'),

]

xlsx_folder = Path("C:/Users/micha/OneDrive/My Trading Data")

df = pd.DataFrame()

with xw.App(visible=True, add_book=False) as app:
    for pair in pairs:
        str1, str2 = pair
        pair_name = f"{str1}_{str2}"
        workbook_path = xlsx_folder / xlsx_folder_suffix / f"{pair_name} 3.xlsx"

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
            df[pair_name] = values
        finally:
            wb.close()

df = df.T
output_file = xlsx_folder / xlsx_folder_suffix / "scalar extracts.xlsx"
df.to_excel(output_file, index=False)
print(f"Saved: {output_file}", flush=True)
