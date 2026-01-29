"""Knowledge base for OpenFOAM solvers."""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Solver:
    """Represents an OpenFOAM solver with its characteristics."""

    name: str
    category: str  # incompressible, compressible, multiphase, etc
    time_type: str  # steady, transient
    description: str
    required_fields: List[str]
    optional_fields: List[str]
    typical_applications: List[str]
    recommended_schemes: Dict[str, str]
    recommended_solvers: Dict[str, Dict[str, Any]]


# Solver database
SOLVERS = {
    "simpleFoam": Solver(
        name="simpleFoam",
        category="incompressible",
        time_type="steady",
        description="Steady-state solver for incompressible, turbulent flow",
        required_fields=["U", "p"],
        optional_fields=["k", "epsilon", "omega", "nut", "T"],
        typical_applications=[
            "External aerodynamics",
            "Internal flows (pipes, ducts)",
            "HVAC simulations",
            "Bluff body flows",
        ],
        recommended_schemes={
            "grad": "Gauss linear",
            "div(phi,U)": "bounded Gauss linearUpwind grad(U)",
            "div(phi,k)": "bounded Gauss upwind",
            "div(phi,epsilon)": "bounded Gauss upwind",
            "laplacian": "Gauss linear corrected",
        },
        recommended_solvers={
            "p": {"solver": "GAMG", "tolerance": 1e-6, "relTol": 0.1},
            "U": {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": 1e-6},
        },
    ),
    "pimpleFoam": Solver(
        name="pimpleFoam",
        category="incompressible",
        time_type="transient",
        description="Transient solver for incompressible, turbulent flow using PIMPLE algorithm",
        required_fields=["U", "p"],
        optional_fields=["k", "epsilon", "omega", "nut", "T"],
        typical_applications=[
            "Transient aerodynamics",
            "Vortex dynamics",
            "Flow around moving objects",
            "Oscillating flows",
        ],
        recommended_schemes={
            "ddt": "Euler",
            "grad": "Gauss linear",
            "div(phi,U)": "bounded Gauss linearUpwind grad(U)",
            "laplacian": "Gauss linear corrected",
        },
        recommended_solvers={
            "p": {"solver": "GAMG", "tolerance": 1e-6, "relTol": 0.05},
            "U": {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": 1e-6},
        },
    ),
    "buoyantSimpleFoam": Solver(
        name="buoyantSimpleFoam",
        category="incompressible",
        time_type="steady",
        description="Steady-state solver for incompressible flow with heat transfer and buoyancy",
        required_fields=["U", "p", "T"],
        optional_fields=["k", "epsilon", "omega", "nut"],
        typical_applications=[
            "Natural convection",
            "Forced convection with heating",
            "Buoyant jets",
            "Stratified flows",
        ],
        recommended_schemes={
            "grad": "Gauss linear",
            "div(phi,U)": "bounded Gauss linearUpwind grad(U)",
            "div(phi,h)": "bounded Gauss upwind",
            "laplacian": "Gauss linear corrected",
        },
        recommended_solvers={
            "p_rgh": {"solver": "GAMG", "tolerance": 1e-6, "relTol": 0.1},
            "T": {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": 1e-7},
        },
    ),
    "rhoCentralFoam": Solver(
        name="rhoCentralFoam",
        category="compressible",
        time_type="transient",
        description="Transient solver for compressible flow using central-upwind scheme",
        required_fields=["rho", "U", "e"],
        optional_fields=["k", "epsilon"],
        typical_applications=[
            "Supersonic flows",
            "Shock waves",
            "Transonic aerodynamics",
            "Compressible jets",
        ],
        recommended_schemes={
            "ddt": "Euler",
            "flux": "Kurganov",
        },
        recommended_solvers={
            "e": {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": 1e-6}
        },
    ),
}


def get_solver_by_name(name: str) -> Solver | None:
    """Get solver information by name."""
    return SOLVERS.get(name)


def list_solvers_by_category(category: str) -> List[str]:
    """List solver names in a given category."""
    return [
        name for name, solver in SOLVERS.items() if solver.category == category
    ]


def list_all_solvers() -> List[str]:
    """List all available solvers."""
    return list(SOLVERS.keys())
