import polars as pl

import yaml


def load_exp_info():
    exp_info_path = r"z:\slap_mi\analysis_materials\exp_info.csv"
    return pl.read_csv(exp_info_path)


def load_sync_info():
    sync_info_path = r"z:\slap_mi\analysis_materials\sync_info.yaml"
    with open(sync_info_path, "r") as file:
        sync_info = yaml.safe_load(file)
    return sync_info
