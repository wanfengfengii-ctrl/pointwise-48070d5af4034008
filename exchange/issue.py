from dataclasses import dataclass

from bench.fractional import FractionalMap, compose_maps
from bench.stages import CalibrationStage
from exchange.calibration_body import CalibrationBody, encode_body
from exchange.envelope import PieceOrigin, encode_envelope
from exchange.support import chain_support
from station.revision_store import RevisionStore


@dataclass(frozen=True)
class IssuedPiece:
    origin: PieceOrigin
    wire: str


def _calibration_body(stages: tuple[CalibrationStage, ...]) -> CalibrationBody:
    support = chain_support(stages)
    if not support:
        return CalibrationBody(None, ())
    terminal = FractionalMap(1, 0, 0, 1)
    try:
        for stage in stages:
            terminal = compose_maps(stage.conversion, terminal)
    except ZeroDivisionError:
        return CalibrationBody(None, ())
    return CalibrationBody(terminal, support)


def issue(store: RevisionStore, revision_id: int) -> IssuedPiece:
    revision = store.get(revision_id)
    origin = PieceOrigin(revision.instrument.instrument_id, revision.instrument.unit,
                         revision.revision_id, revision.evidence.source)
    body = _calibration_body(revision.stages)
    return IssuedPiece(origin, encode_envelope(origin, encode_body(body)))
