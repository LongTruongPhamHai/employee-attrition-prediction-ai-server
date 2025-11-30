import pandas as pd
from config.settings import RAW_DATA
from common.overview import raw_data_overview


def overview_service():
    df_raw = pd.read_csv(RAW_DATA)
    result = raw_data_overview(df_raw, "Raw data", target_col="Attrition")

    return result
