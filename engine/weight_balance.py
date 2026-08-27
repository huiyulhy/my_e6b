"""Weight and balance: gross weight and centre of gravity from loading rows.

Calculates gross weight and cg location but rows have to be 
entered by the user

Future: point in polygon checks for wnb envelopes

units: in, lb, lb-in
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    """One line of the loading form: a named mass at an arm.

    A negative weight is allowed -- fuel burned off is entered that way, and a
    zero-fuel CG is a normal thing to want. 
    """

    name: str
    weight_lb: float
    arm_in: float

    @property
    def moment_in_lb(self) -> float:
        return self.weight_lb * self.arm_in


@dataclass(frozen=True)
class Loading:
    """What a set of stations adds up to."""

    gross_weight_lb: float
    total_moment_in_lb: float
    cg_in: float
    stations: tuple[Station, ...]


def compute(stations: list[Station]) -> Loading:
    """
    Calculate the gross weight and cg location
    """
    if not stations:
        raise ValueError("Enter at least one loading station.")
    gross = sum(station.weight_lb for station in stations)
    moment = sum(station.moment_in_lb for station in stations)
    if gross <= 0.0:
        raise ValueError(
            "Total weight must be greater than zero to have a centre of gravity."
        )
    return Loading(
        gross_weight_lb=gross,
        total_moment_in_lb=moment,
        cg_in=moment / gross,
        stations=tuple(stations),
    )
