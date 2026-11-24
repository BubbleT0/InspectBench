#!/usr/bin/env python3
"""Bound the composite-score credit from failed clean outputs.

This preserves the benchmark scores and denominators. Only the hypothetical
credit for failed clean images is changed from one to zero. Component rounding
is propagated outward; no image-level inference or rescoring is performed.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, localcontext
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = Decimal
LAMBDA = D('0.7')
CLEAN_COUNT = D(321)
ROUNDING = D('0.00005')


def composite(positive: Decimal, clean: Decimal) -> Decimal:
    """Calculate the fixed composite score for components in the unit interval."""
    if not (D(0) <= positive <= D(1) and D(0) <= clean <= D(1)):
        raise ValueError('Both score components must lie in [0, 1].')
    return D(100) * (LAMBDA * positive.sqrt() + (D(1) - LAMBDA) * clean.sqrt()) ** 2


def score_credit(positive: Decimal, clean: Decimal, failures: int) -> Decimal:
    """Calculate the maximum clean-failure credit at fixed component values."""
    if failures < 0:
        raise ValueError('Failure count must be nonnegative.')
    d = min(D(failures) / CLEAN_COUNT, clean)
    weight = D(1) - LAMBDA
    return D(100) * (weight ** 2 * d + D(2) * LAMBDA * weight * positive.sqrt()
                     * (clean.sqrt() - (clean - d).sqrt()))


def audit() -> dict:
    """Calculate rounding-safe bounds and check them against score differences."""
    rows = json.loads((ROOT / 'data/results_renamed.json').read_text())
    output = []
    checked = 0
    with localcontext() as context:
        context.prec = 50
        for row in rows:
            failures = int(row['failed'])
            if failures != row['failed']:
                raise ValueError('Nonintegral failure count.')
            p, c = D(str(row['positive_soft'])), D(str(row['clean_soft']))
            d = min(D(failures) / CLEAN_COUNT, c)
            value = score_credit(p, c, failures)
            direct = composite(p, c) - composite(p, c - d)
            if abs(value - direct) > D('1e-45'):
                raise AssertionError('Expanded bound and direct score subtraction disagree.')
            p_high = min(D(1), p + ROUNDING)
            c_low, c_high = max(D(0), c - ROUNDING), min(D(1), c + ROUNDING)
            # At fixed p, the credit peaks at c = failures / clean_count.
            # It decreases beyond that point; clip the peak to the rounding interval.
            c_worst = min(c_high, max(c_low, D(failures) / CLEAN_COUNT))
            upper = score_credit(p_high, c_worst, failures)
            for p_test in [max(D(0), p - ROUNDING), p, p_high]:
                for c_test in [c_low, c, c_high]:
                    for f_test in range(failures + 1):
                        checked += 1
                        if score_credit(p_test, c_test, f_test) > upper + D('1e-45'):
                            raise AssertionError('A rounding/count corner exceeds the bound.')
            output.append({
                'model': row['model'], 'procedure': row['procedure'],
                'failed_image_outputs': failures,
                'positive_component': str(p), 'clean_component': str(c),
                'bound_from_rounded_components': str(value),
                'rounding_safe_upper_bound': str(upper),
                'published_upper_bound': str(upper.quantize(D('0.001'), rounding=ROUND_CEILING)),
            })
        ranking = sorted(rows, key=lambda x: x['score'], reverse=True)
        top_scores = [D(str(r['score'])) for r in ranking[:3]]
        high_sensitivity = [r for r in rows if 1-r['image_fnr'] >= .97]
        return {
            'definition': 'Maximum S credit from replacing failed clean-image zero scores by empty-set scores of one, with fixed denominators and all other contributions unchanged.',
            'formula': '100*((1-lambda)^2*d + 2*lambda*(1-lambda)*sqrt(Qplus)*(sqrt(Qminus)-sqrt(Qminus-d)))',
            'd': 'min(failed_image_outputs/321, Qminus)',
            'component_half_rounding_unit': str(ROUNDING),
            'bound_rounding': 'upward to 0.001 S points',
            'all_failures_clean_worst_case': True,
            'measured_scores_changed': False,
            'formula_subtraction_checks': len(rows),
            'rounding_and_failure_count_corner_checks': checked,
            'configurations': output,
            'top3_adjacent_score_gaps': [str(top_scores[i]-top_scores[i+1]) for i in range(2)],
            'top3_max_score_gap': str(max(top_scores)-min(top_scores)),
            'high_sensitivity_clean_fpr_range_pct': [100*min(r['clean_fpr'] for r in high_sensitivity),100*max(r['clean_fpr'] for r in high_sensitivity)],
            'small_recall_checkpoints': [{k:r[k] for k in ['model','procedure','recall_01','image_fnr']} for r in rows if r['model'] in ['Decision-SFT-Balanced+','Decision-SFT-Pool+']],
        }


def main() -> None:
    """Write the reproducible numerical audit next to the unchanged result data."""
    result = audit()
    (ROOT / 'data/wording_numeric_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in result['configurations']:
        if row['failed_image_outputs']:
            print(f"{row['model']} / {row['procedure']}: n_fail={row['failed_image_outputs']}; Delta S <= {row['published_upper_bound']}")
    print(f"Verified {result['formula_subtraction_checks']} formulas and {result['rounding_and_failure_count_corner_checks']} rounding/count corners.")


if __name__ == '__main__':
    main()
