import gzip

import numpy as np
import pandas as pd
import pytest

from data_loader import (ACE, ColumnMapping, clean_motion, detect_columns, read_csv, resample_window,
                         source_bytes, source_mapping, validate_holdouts)

MAPPING = ColumnMapping("time", "Roll", "Pitch", "s", "deg", "deg", 1, 1)


def test_real_source_columns_and_sampling(gunnerus):
    assert gunnerus.quality.input_rows == 15597
    assert gunnerus.quality.observation_dt == 0.5
    np.testing.assert_allclose(gunnerus.angles_deg[0], [-0.889893, -1.43921])
    assert gunnerus.origin.startswith("2023-10-31T10:00:00.241")
    with pytest.raises(ValueError, match="방향"):
        source_mapping(ACE)
    frame = read_csv(source_bytes(ACE))
    assert detect_columns(frame.columns) == {
        "time": "pc_date_utc", "clock": "pc_time_hhmmsssss", "roll": "platform_roll", "pitch": "platform_pitch"
    }
    ace = clean_motion(frame, source_mapping(ACE, 1, -1), source=ACE)
    assert ace.quality.input_rows == 900
    assert ace.quality.observation_dt == 1.0
    assert ace.origin.startswith("2017-01-26T13:20:00.531")
    np.testing.assert_allclose(ace.angles_deg[0], [-1.828, 1.395])


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("method", ["linear", "pchip"])
def test_csv_units_and_interpolation(compressed, method):
    original = b"Time_ms;Roll_rad;Pitch_deg\n0;0;1\n1000;0.02;3\n2000;0.04;5\n"
    buffer = gzip.compress(original) if compressed else original
    raw = read_csv(buffer)
    assert detect_columns(raw)["roll"] == "Roll_rad"
    assert detect_columns(raw)["time"] == "Time_ms"
    mapping = ColumnMapping("Time_ms", "Roll_rad", "Pitch_deg", "ms", "rad", "deg", -1, 1)
    window = resample_window(clean_motion(raw, mapping), 0, 0, 2, 0.25, method=method)
    assert window.window.observations == 3
    np.testing.assert_allclose(window.time_s, np.linspace(0, 2, 9))
    np.testing.assert_allclose(window.angles_deg[:, 0], -np.rad2deg(window.time_s * 0.02))
    np.testing.assert_allclose(window.angles_deg[:, 1], 1 + 2 * window.time_s)
    assert raw["Roll_rad"].tolist() == [0, 0.02, 0.04]


def test_missing_short_gap_outliers_and_long_gap():
    df = pd.DataFrame({"time": [0, 2, 1, 2, 3, 4, 10, 11, "bad"],
                       "Roll": [0, 2, None, 44, 999, 4, 10, 11, 0],
                       "Pitch": [0, 0, 0, 0, 0, 0, 0, 0, 0]})
    clean = clean_motion(df, MAPPING, max_gap=2)
    assert clean.quality.bad_time_rows == 1
    assert clean.quality.bad_angle_rows == 2
    assert clean.quality.duplicate_rows == 1
    assert clean.segments == ((0, 3), (3, 5))
    values = resample_window(clean, 0, 0, 4, 0.5)
    np.testing.assert_allclose(values.angles_deg[:, 0], values.time_s)
    with pytest.raises(ValueError, match="연속 구간"):
        resample_window(clean, 0, 0, 11)
    assert df.Roll.iloc[4] == 999


def test_holdouts_are_disjoint_and_only_interpolate_inside(gunnerus, train):
    with pytest.raises(ValueError, match="공유"):
        validate_holdouts(train.window, resample_window(gunnerus, 0, 9, 15).window)
    b = resample_window(gunnerus, 0, 1560, 1570)
    c = resample_window(gunnerus, 0, 2830, 2840)
    validate_holdouts(train.window, b.window, c.window)
    with pytest.raises(ValueError, match="공유"):
        validate_holdouts(train.window, b.window, b.window)
    assert b.source_elapsed_s[0] >= 1560
    assert b.source_elapsed_s[-1] <= b.window.observed_end
    assert b.window.manifest()["start_utc"].startswith("2023-10-31T10:26:00")
    same_upload = clean_motion(read_csv(source_bytes(gunnerus.source)), source_mapping(gunnerus.source))
    with pytest.raises(ValueError, match="공유"):
        validate_holdouts(train.window, resample_window(same_upload, 0, 0, 10).window)
