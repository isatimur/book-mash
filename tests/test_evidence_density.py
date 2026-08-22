from book_mash.judges.evidence_density import _label_for_density


def test_label_strong_above_threshold():
    # 1 claim per 250 words -> strong
    assert _label_for_density(claims_count=4, word_count=1000) == "strong"


def test_label_moderate():
    # 1 claim per 500 words
    assert _label_for_density(claims_count=2, word_count=1000) == "moderate"


def test_label_weak():
    # 1 claim per 800 words
    assert _label_for_density(claims_count=1, word_count=800) == "weak"


def test_label_fail():
    # 0 claims
    assert _label_for_density(claims_count=0, word_count=500) == "fail"
    # 1 per 1500 words
    assert _label_for_density(claims_count=1, word_count=1500) == "fail"
