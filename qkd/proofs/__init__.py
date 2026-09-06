# -*- coding: utf-8 -*-
"""
Security proofs sub-package for the QKD framework.
This package exposes the core abstract base class for all proofs and the
concrete implementations for different QKD protocols.
"""

# Import the base class to make it available for type hinting and extension.
from .base import FiniteKeyProof

# Import the concrete proof implementations to make them part of the public API.
from .lim2014 import Lim2014Proof
from .tight import BB84TightProof
from .mdi import MDIQKDProof
from .paper_2009_individual import Paper2009IndividualAttackProof
from .wang2005 import Wang2005Proof
# --- FIX: Added Ma2005AsymptoticProof to the import list ---
from .ma2005 import (
    Ma2005VacuumWeakProof,
    Ma2005OneDecoyProof,
    Ma2005GeneralTwoDecoyProof,
    Ma2005AsymptoticProof
)

# Define the public API for this sub-package.
__all__ = [
    "FiniteKeyProof",
    "Lim2014Proof",
    "BB84TightProof",
    "MDIQKDProof",
    "Paper2009IndividualAttackProof",
    "Wang2005Proof",
    "Ma2005VacuumWeakProof",
    "Ma2005OneDecoyProof",
    "Ma2005GeneralTwoDecoyProof",
    "Ma2005AsymptoticProof",
]
