from src.oms.retry_policy import RetryPolicy


def test_should_retry_within_limit():
    policy = RetryPolicy(max_attempts=4)
    assert policy.should_retry(0) is True
    assert policy.should_retry(3) is True
    assert policy.should_retry(4) is False


def test_delay_within_cap():
    policy = RetryPolicy(base_delay=1.0, cap=30.0)
    for attempt in range(10):
        d = policy.delay(attempt)
        assert 0 <= d <= 30.0


def test_delay_increases_with_attempt():
    """Average delay should grow with attempt number."""
    policy = RetryPolicy(base_delay=1.0, cap=60.0)
    samples = 200
    avg0 = sum(policy.delay(0) for _ in range(samples)) / samples
    avg3 = sum(policy.delay(3) for _ in range(samples)) / samples
    assert avg3 > avg0
