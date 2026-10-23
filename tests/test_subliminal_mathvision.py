import json

import pyarrow as pa
import pyarrow.parquet as pq

from src.subliminal_mathvision import (
    build_query,
    canonical_math,
    mathvision_equal,
    normalize_mathvision_prediction,
    score_mathvision_response,
)


def test_mathvision_symbolic_equivalence_and_dataset_typos():
    assert mathvision_equal(r'\frac{2}{6}', r'$\frac{1}{3}$')
    assert mathvision_equal(r'4\sqrt{3} + 6', r'$6+4\sqrt{3}$')
    assert mathvision_equal(r'\frac{36}{13}', '\x0crac{36}{13}')
    assert mathvision_equal('0.5', r'\frac{1}{2}')
    assert mathvision_equal('1/2', '1:2')
    assert canonical_math(r'$69^{\circ}$') == '69'


def test_mathvision_multiple_choice_mapping_and_no_guessing():
    options = ['12', '24', '36', '48', '60']
    assert normalize_mathvision_prediction('E', options) == 'E'
    assert normalize_mathvision_prediction(r'\frac{12}{1}', options) == 'A'
    assert normalize_mathvision_prediction('', options) is None
    row = {'answer': '60', 'options': options}
    assert score_mathvision_response(r'Final: \boxed{E}', row)[-1]


def test_query_formats_real_choices_and_removes_image_marker():
    query = build_query({'question': 'Find x.\n<image1>', 'options': ['2', '3', '4', '5', '6']},
                        'Give a boxed answer.')
    assert '<image1>' not in query
    assert '(A) 2' in query and '(E) 6' in query
    assert query.endswith('Give a boxed answer.')
    literal = build_query({'question': 'Pick the diagram. <image1>',
                           'options': ['A', 'B', 'C', 'D', 'E']}, 'Answer.')
    assert 'Choices:' not in literal


def test_scorer_handles_boxed_nested_latex():
    row = {'answer': r'$\frac{\sqrt{6}}{6}$', 'options': []}
    extraction, prediction, answer, correct = score_mathvision_response(
        r'Work. Final: \boxed{\frac{1}{\sqrt{6}}}', row)
    assert extraction == r'\frac{1}{\sqrt{6}}'
    assert prediction == extraction and answer == row['answer'] and correct
