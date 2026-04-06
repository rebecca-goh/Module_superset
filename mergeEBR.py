import zipfile
import os
import shutil
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
import datetime
import json
import urllib.request
from datetime import datetime


def main():

    #input filenames
    assy_filename = "EBR_ASSY_REPORT_Vettel.csv"
    onestop_filename = "1stop_solution_Vettel_25Nov2025_173931.xlsx"

    #Load input files
    df_assy = pd.read_csv(assy_filename)
    df_onestop = pd.read_excel(onestop_filename, sheet_name="Sheet1")
    df_onestop = df_onestop.iloc[:, :70]
    df_onestop  = df_onestop .reindex(columns=['Target Device', 'Build Name', 'EBR Name (Assy)', 'EBR Sub Lot (Assy)', 'Test Lot#', 'MFG ID (Assy)'])
    df_onestop = df_onestop.rename(columns={'EBR Name (Assy)': 'ebr_no', 'EBR Sub Lot (Assy)': 'assylot_id','MFG ID (Assy)': 'MFG ID', 'Build Name': 'Build Purpose'})
    df_assymerged = pd.merge(df_onestop, df_assy[['assylot_id', 'dieposition', 'material_partnumber', 'material_id']], on='assylot_id', how='right')
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    newfile_path = "module_assy_report" + timestamp + ".csv"
    df_assymerged.to_csv(newfile_path, index=False) 
    #df_assymerged = df_assymerged[df_assymerged['dieposition'].str.contains("F")]
    newfile_path = "module_ebr_assy_report_fbar" + timestamp + ".csv"
    df_assymerged.to_csv(newfile_path, index=False) 

if __name__ == '__main__':
    main()