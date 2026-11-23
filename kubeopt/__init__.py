"""KubeOpt : optimisation de la flotte EC2 (Workstream B). Étapes B2-a (données) et B2-b (placement, métriques)."""
from .errors import DataValidationError, InvalidPlacementError, KubeOptError
from .evaluation import Evaluation, evaluate
from .annealing import (
    Phase2Result, SAConfig, SAResult, TwoPhaseResult, balance_annealing, initial_solution, multistart_annealing,
    simulated_annealing, two_phase_annealing,
)
from .baselines import (
    BASELINES, OPEN_BASELINES, BaselineResult, cluster_nodes, merge_runs, mono_type_fleet, open_assignment,
    run_all_baselines, run_all_open_baselines, run_baseline, run_open_baseline, server_lower_bound,
)
from .generator import (
    PAPER_FLEETS, GeneratorConfig, generate_cluster_instance, generate_instance, paper_example,
)
from .models import (
    BW, CPU, DIMENSIONS, RAM, UNITS, Instance, InstanceType, Reservation, Scenario, Workload,
)
from .metrics import ChangeMetrics, ScenarioMetrics, change_metrics, compute_scenario_metrics, fleet_violation
from .placement import place, place_all
from .solution import OVERFLOW, Solution, node_types_from_composition
from .units import IntegerView, to_integer_view
from .validator import ValidationReport, validate_placement, validate_solution

__version__ = "0.6.0"
__all__ = [
    "BW", "CPU", "RAM", "DIMENSIONS", "UNITS", "DataValidationError", "KubeOptError",
    "GeneratorConfig", "Instance", "InstanceType", "IntegerView", "PAPER_FLEETS",
    "Reservation", "Scenario", "Workload", "generate_instance", "paper_example", "to_integer_view",
    "InvalidPlacementError", "Evaluation", "evaluate", "ChangeMetrics", "ScenarioMetrics", "change_metrics",
    "compute_scenario_metrics", "fleet_violation", "place", "place_all", "OVERFLOW", "Solution",
    "node_types_from_composition", "ValidationReport", "validate_placement", "validate_solution",
    "BASELINES", "BaselineResult", "cluster_nodes", "run_all_baselines", "run_baseline", "server_lower_bound",
    "generate_cluster_instance", "OPEN_BASELINES", "run_all_open_baselines", "run_open_baseline",
    "open_assignment", "merge_runs", "mono_type_fleet",
    "SAConfig", "SAResult", "initial_solution", "simulated_annealing", "multistart_annealing",
    "Phase2Result", "TwoPhaseResult", "balance_annealing", "two_phase_annealing",
]
