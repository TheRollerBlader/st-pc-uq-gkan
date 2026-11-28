import numpy as np
from stpcuq_gkan.noaa_ingester import NoaaSwpcIngester


class TestNoaaSwpcIngester:
    def test_newell_coupling_computation(self):
        # Pure northward IMF (by=0, bz=10): theta = 0, clock_term = 0
        c_north = NoaaSwpcIngester.compute_newell_coupling(speed=400.0, by=0.0, bz=10.0)
        assert c_north == 0.0

        # Strong southward IMF (by=0, bz=-15): theta = pi, sin(pi/2) = 1
        c_south = NoaaSwpcIngester.compute_newell_coupling(speed=600.0, by=0.0, bz=-15.0)
        assert c_south > 5000.0

    def test_payload_parsing_and_alignment(self):
        ingester = NoaaSwpcIngester(southward_bz_threshold=-10.0, high_speed_threshold=650.0)

        mag_json = [
            ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "lon_gsm", "lat_gsm", "bt"],
            ["2026-11-12 10:00:00.000", "-2.5", "4.0", "-12.5", "120.0", "-45.0", "13.3"],
            ["2026-11-12 10:05:00.000", "-1.0", "2.0", "-4.0", "110.0", "-25.0", "4.5"],
        ]

        plasma_json = [
            ["time_tag", "density", "speed", "temperature"],
            ["2026-11-12 10:00:00.000", "8.5", "520.0", "120000.0"],
            ["2026-11-12 10:05:00.000", "4.2", "410.0", "85000.0"],
        ]

        mag_map = ingester.parse_mag_payload(mag_json)
        plasma_map = ingester.parse_plasma_payload(plasma_json)

        assert len(mag_map) == 2
        assert len(plasma_map) == 2

        obs = ingester.align_and_build_series(mag_map, plasma_map)
        assert len(obs) == 2

        # First observation has bz = -12.5, which is <= -10.0 anomaly threshold
        assert obs[0].time_tag == "2026-11-12 10:00:00.000"
        assert obs[0].bz == -12.5
        assert obs[0].speed == 520.0
        assert obs[0].density == 8.5
        assert obs[0].is_anomaly is True

        # Second observation is normal
        assert obs[1].bz == -4.0
        assert obs[1].is_anomaly is False

        # Convert to numpy tensor
        arr = ingester.to_solar_driver_array(obs)
        assert arr.shape == (2, 5)
        np.testing.assert_allclose(arr[0], [-2.5, 4.0, -12.5, 520.0, 8.5], rtol=1e-4)

    def test_utc_epoch_kp_and_live_window_fetch(self):
        ingester = NoaaSwpcIngester()
        epoch = ingester.parse_utc_epoch("2026-11-26 09:00:00.000")
        assert epoch is not None and epoch > 1_700_000_000

        kp_payload = [
            ["time_tag", "Kp", "a_running", "station_count"],
            ["2026-11-26 06:00:00.000", "5.33", "56", "8"],
            ["2026-11-26 09:00:00.000", "6.00", "80", "8"],
        ]
        kp_map = ingester.parse_kp_payload(kp_payload)
        assert kp_map["2026-11-26 09:00:00.000"] == 6.0

        mag_json = [
            ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "lon_gsm", "lat_gsm", "bt"],
            ["2026-11-26 09:00:00.000", "-3.0", "5.0", "-14.0", "120.0", "-50.0", "15.1"],
        ]
        plasma_json = [
            ["time_tag", "density", "speed", "temperature"],
            ["2026-11-26 09:00:00.000", "11.2", "680.0", "190000.0"],
        ]

        def fake_fetch(url: str):
            if "mag-5-minute" in url:
                return mag_json
            if "plasma-5-minute" in url:
                return plasma_json
            return kp_payload

        window = ingester.fetch_live_swpc_window(fetch_json=fake_fetch, include_kp=True)
        assert len(window["observations"]) == 1
        assert window["observations"][0].timestamp_epoch == epoch
        assert window["anomaly_count"] == 1
        assert window["solar_drivers"].shape == (1, 5)
        assert window["kp_series"]["2026-11-26 09:00:00.000"] == 6.0
