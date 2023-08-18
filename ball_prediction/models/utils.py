from enum import Enum

import numpy as np


def radians_to_revolutions(radians_per_second_sequence):
    revolutions_per_second_sequence = [
        rads / (2 * np.pi) for rads in radians_per_second_sequence
    ]
    return revolutions_per_second_sequence


def revolutions_to_radians(revolutions_per_second_sequence):
    radians_per_second_sequence = [
        revs * 2 * np.pi for revs in revolutions_per_second_sequence
    ]
    return radians_per_second_sequence


class ContactType(Enum):
    RACKET = "racket"
    TABLE = "table"
    UNKNOWN = "unknown"
