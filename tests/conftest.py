import pytest

from data_loader import GUNNERUS, clean_motion, read_csv, resample_window, source_bytes, source_mapping


@pytest.fixture(scope="session")
def gunnerus():
    return clean_motion(read_csv(source_bytes(GUNNERUS)), source_mapping(GUNNERUS), source=GUNNERUS)


@pytest.fixture(scope="session")
def train(gunnerus):
    return resample_window(gunnerus, 0, 0, 10)
