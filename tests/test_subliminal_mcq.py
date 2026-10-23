from src.subliminal_mcq import (
    normalize_mcq_prediction,
    open_equal,
    parse_embedded_choices,
    score_mcq_response,
)


def test_mcq_letter_and_choice_text_normalization():
    options = ['red apple', 'green pear', 'blue berry']
    assert normalize_mcq_prediction(r'\text{B}', options) == 'B'
    assert normalize_mcq_prediction('blue berry.', options) == 'C'
    assert normalize_mcq_prediction('unknown', options) is None


def test_chartqa_numeric_tolerance_is_five_percent():
    assert open_equal('104', '100', 'chartqa')
    assert not open_equal('106', '100', 'chartqa')
    assert open_equal('Yes.', 'yes', 'chartqa')


def test_embedded_choices_and_boxed_scoring():
    options = parse_embedded_choices('Question\nOptions: A: cat, B: dog, C: bird, D: fish.')
    assert options == ['cat', 'dog', 'bird', 'fish.']
    row = {'_question_type': 'multiple-choice', '_options': options,
           '_answer': 'C', '_gold_answers': ['C']}
    assert score_mcq_response(r'Reasoning. Final: \boxed{C}', row, 'mmstar')[-1]
