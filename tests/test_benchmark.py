from app.benchmark import summarize_latencies


def test_summarize_latencies_reports_milliseconds_and_throughput():
    summary = summarize_latencies([0.01, 0.02, 0.03])

    assert summary["min_ms"] == 10.0
    assert summary["mean_ms"] == 20.0
    assert summary["median_ms"] == 20.0
    assert summary["max_ms"] == 30.0
    assert summary["throughput_images_per_second"] == 50.0
