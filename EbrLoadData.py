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

def download_ebrassyreport():

    # Read idpass.jwt generated from Atropos website
    with open(f"{os.getenv('USERPROFILE')}\\idpass.jwt", "r") as file: access_token = file.read()

    # Generate access token from idpass.jwt
    url = "https://safelock.wsd.sgn.broadcom.net/authendpoint/accesstoken"
    httpHeaders = {"Authorization": f"Bearer {access_token}"}
    req = urllib.request.Request(url, headers=httpHeaders)
    response = urllib.request.urlopen(req)
    access_token = response.read().decode("utf-8")

    # Query web service
    url = "http://10.202.128.107/v2/genealogyapi/ProcessQuery"
    payload = {
        "table": "ebr_assy_report",
        "columns": "*",
        "syscol": "",
        "args": {
            "ebr_no": "VETLX61M48,VETLX61M50,VETLX61M54,VETLX61A03"
        }
    }

    jsonstr = json.dumps(payload)
    httpHeaders = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    req = urllib.request.Request(url, headers=httpHeaders, data=jsonstr.encode("utf-8"))
    response = urllib.request.urlopen(req)
    jsonstr = json.dumps(payload)
    httpHeaders = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    req = urllib.request.Request(url, headers=httpHeaders, data=jsonstr.encode("utf-8"))
    response = urllib.request.urlopen(req)
    df = pd.read_table(response, index_col=None, sep=',')
    df.to_csv("ebr_assy_report.csv", index=False)

def main():
    download_ebrassyreport()

    #input filenames
    assy_filename = "ebr_assy_report.csv"
    onestop_filename = "1stop_solution.xlsx"

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
    newfile_path = "module_assy_report_fbar" + timestamp + ".csv"
    df_assymerged.to_csv(newfile_path, index=False) 

if __name__ == '__main__':
    main()