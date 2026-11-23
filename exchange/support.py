"""Exact original-setting support of an ordered calibration chain.

Every stage is applied to the input that actually reaches it: the stage's
closed input range and its non-zero denominator both restrict the *original*
setting axis.  Each conversion is a Mobius map with rational coefficients, so
pulling an allowed input interval back through the previous stages yields
exact intervals with rational (or unbounded) endpoints.  The support can be a
union of components and may be punctured by an intermediate pole.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import inf

from bench.fractional import FractionalMap, compose_maps
from bench.stages import CalibrationStage
from exchange.calibration_body import SupportInterval


@dataclass(frozen=True)
class _Span:
    lo: Fraction | float
    lo_closed: bool
    hi: Fraction | float
    hi_closed: bool


def _value_order(value) -> tuple:
    if value == -inf:
        return (0, 0)
    if value == inf:
        return (2, 0)
    return (1, value)


def _ordered_make(first, first_closed, second, second_closed) -> _Span | None:
    if _value_order(second) < _value_order(first):
        first, first_closed, second, second_closed = second, second_closed, first, first_closed
    return _make(first, first_closed, second, second_closed)


def _make(lo, lo_closed, hi, hi_closed) -> _Span | None:
    if lo != -inf and hi != inf:
        if lo == hi and not (lo_closed and hi_closed):
            return None
        if lo > hi:
            return None
    if lo == -inf or lo == inf:
        lo_closed = False
    if hi == inf or hi == -inf:
        hi_closed = False
    return _Span(lo, lo_closed, hi, hi_closed)


def _to_support(span: _Span) -> SupportInterval:
    lower = None if span.lo == -inf else span.lo
    upper = None if span.hi == inf else span.hi
    return SupportInterval(lower, upper,
                           span.lo_closed if lower is not None else False,
                           span.hi_closed if upper is not None else False)


_FULL_SPAN = _Span(-inf, False, inf, False)


def _contains(span: _Span, value: Fraction) -> bool:
    left = span.lo == -inf or value > span.lo or (span.lo_closed and value == span.lo)
    right = span.hi == inf or value < span.hi or (span.hi_closed and value == span.hi)
    return left and right


def _later_lower(first, first_closed, second, second_closed):
    if first > second:
        return first, first_closed
    if second > first:
        return second, second_closed
    return first, first_closed and second_closed


def _earlier_upper(first, first_closed, second, second_closed):
    if first < second:
        return first, first_closed
    if second < first:
        return second, second_closed
    return first, first_closed and second_closed


def _intersect(first: _Span, second: _Span) -> _Span | None:
    lo, lo_closed = _later_lower(first.lo, first.lo_closed, second.lo, second.lo_closed)
    hi, hi_closed = _earlier_upper(first.hi, first.hi_closed, second.hi, second.hi_closed)
    return _make(lo, lo_closed, hi, hi_closed)


def _subtract_point(span: _Span, point: Fraction) -> list[_Span]:
    if not _contains(span, point):
        return [span]
    pieces = []
    if span.lo != point:
        piece = _make(span.lo, span.lo_closed, point, False)
        if piece is not None:
            pieces.append(piece)
    if span.hi != point:
        piece = _make(point, False, span.hi, span.hi_closed)
        if piece is not None:
            pieces.append(piece)
    return pieces


def _merge(first: _Span, second: _Span) -> _Span:
    if first.hi == inf or second.hi == inf:
        hi, hi_closed = inf, False
    elif first.hi > second.hi:
        hi, hi_closed = first.hi, first.hi_closed
    elif second.hi > first.hi:
        hi, hi_closed = second.hi, second.hi_closed
    else:
        hi, hi_closed = first.hi, first.hi_closed or second.hi_closed
    return _make(first.lo, first.lo_closed, hi, hi_closed)


def _sort_key(span: _Span):
    return (0, 0) if span.lo == -inf else (1, span.lo)


def _normalize(spans: list[_Span]) -> list[_Span]:
    ordered = sorted(spans, key=_sort_key)
    result: list[_Span] = []
    for span in ordered:
        if not result:
            result.append(span)
            continue
        previous = result[-1]
        if previous.hi == inf or span.lo == -inf or span.lo < previous.hi:
            result[-1] = _merge(previous, span)
        elif span.lo == previous.hi and (previous.hi_closed or span.lo_closed):
            result[-1] = _merge(previous, span)
        else:
            result.append(span)
    return result


def _intersect_unions(first: list[_Span], second: list[_Span]) -> list[_Span]:
    pieces = []
    for left in first:
        for right in second:
            piece = _intersect(left, right)
            if piece is not None:
                pieces.append(piece)
    return _normalize(pieces)


def _allowed_inputs(stage: CalibrationStage) -> list[_Span]:
    limits = stage.input_range
    if limits.lower is None and limits.upper is None:
        span = _FULL_SPAN
    else:
        span = _make(Fraction(limits.lower), True, Fraction(limits.upper), True)
    pole = stage.conversion.pole
    if pole is None:
        return [span]
    return _subtract_point(span, pole)


def _inverse_value(a: Fraction, b: Fraction, c: Fraction, d: Fraction, y: Fraction) -> Fraction:
    return (d * y - b) / (a - c * y)


def _linear_preimage(a: Fraction, b: Fraction, d: Fraction, span: _Span) -> list[_Span]:
    ratio = d / a

    def back(y, closed):
        if y == inf:
            return (inf if ratio > 0 else -inf), False
        if y == -inf:
            return (-inf if ratio > 0 else inf), False
        return (d * y - b) / a, closed

    lo, lo_closed = back(span.lo, span.lo_closed)
    hi, hi_closed = back(span.hi, span.hi_closed)
    piece = _ordered_make(lo, lo_closed, hi, hi_closed)
    return [] if piece is None else [piece]


def _mobius_preimage(a: Fraction, b: Fraction, c: Fraction, d: Fraction,
                     span: _Span) -> list[_Span]:
    pole = -d / c
    asymptote = a / c
    pieces = []
    # Two open branches of the conversion, split at the domain pole.  The
    # branch at -infinity (x < pole) attains the horizontal asymptote only in
    # the limit x -> -inf; the branch at +infinity in the limit x -> +inf.
    for side in (-1, 1):
        probe = pole + side
        image = (a * probe + b) / (c * probe + d)
        if image > asymptote:
            ray = _make(asymptote, False, inf, False)
        else:
            ray = _make(-inf, False, asymptote, False)
        bounded = _intersect(span, ray)
        if bounded is None:
            continue
        branch_infinity = -inf if side < 0 else inf

        def back(y, closed):
            if y == asymptote:
                return branch_infinity, False
            if y == inf or y == -inf:
                return pole, False
            return _inverse_value(a, b, c, d, y), closed

        lo, lo_closed = back(bounded.lo, bounded.lo_closed)
        hi, hi_closed = back(bounded.hi, bounded.hi_closed)
        piece = _ordered_make(lo, lo_closed, hi, hi_closed)
        if piece is not None:
            pieces.append(piece)
    return pieces


def _preimage(conversion: FractionalMap, span: _Span) -> list[_Span]:
    a, b, c, d = conversion.coefficients
    if a * d == b * c:
        value = b / d if d else a / c
        if not _contains(span, value):
            return []
        if c == 0:
            return [_FULL_SPAN]
        return _subtract_point(_FULL_SPAN, -d / c)
    if c == 0:
        return _linear_preimage(a, b, d, span)
    return _mobius_preimage(a, b, c, d, span)


def chain_support(stages: tuple[CalibrationStage, ...]) -> tuple[SupportInterval, ...]:
    conversion = FractionalMap(1, 0, 0, 1)
    support = [_FULL_SPAN]
    for stage in stages:
        pieces: list[_Span] = []
        for allowed in _allowed_inputs(stage):
            pieces.extend(_preimage(conversion, allowed))
        support = _intersect_unions(support, pieces)
        if not support:
            return ()
        try:
            conversion = compose_maps(stage.conversion, conversion)
        except ZeroDivisionError:
            return ()
    return tuple(_to_support(span) for span in support)
