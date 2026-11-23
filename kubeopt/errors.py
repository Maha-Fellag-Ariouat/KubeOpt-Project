"""Exceptions KubeOpt."""
from __future__ import annotations


class KubeOptError(Exception):
    """Classe de base des erreurs KubeOpt."""


class DataValidationError(KubeOptError, ValueError):
    """Donnée d'entrée invalide (dimension, signe, valeur non finie, incohérence).

    C'est une erreur de données, pas un problème d'optimisation (B1 §3.2).
    """


class InvalidPlacementError(KubeOptError):
    """Placement ou flotte violant au moins une contrainte dure ((4), (6)-(10), (8bis))."""

    def __init__(self, violations):
        self.violations = tuple(violations)
        shown = "; ".join(self.violations[:5])
        extra = f" (+{len(self.violations) - 5} autres)" if len(self.violations) > 5 else ""
        super().__init__(f"{len(self.violations)} violation(s) : {shown}{extra}")
