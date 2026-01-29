"""Knowledge base for OpenFOAM boundary conditions."""

from dataclasses import dataclass
from typing import List


@dataclass
class BoundaryCondition:
    """Represents an OpenFOAM boundary condition type."""

    name: str
    description: str
    typical_use_cases: List[str]
    parameters: List[str]
    field_types: List[str]  # U, p, k, epsilon, omega, T, etc
    example: str


# Boundary condition database
BOUNDARY_CONDITIONS = {
    "fixedValue": BoundaryCondition(
        name="fixedValue",
        description="Prescribe a fixed value at the boundary",
        typical_use_cases=[
            "Inlet velocity specification",
            "Outlet pressure setting",
            "Wall temperature",
        ],
        parameters=["value"],
        field_types=["all"],
        example="type fixedValue;\nvalue uniform (1 0 0);",
    ),
    "zeroGradient": BoundaryCondition(
        name="zeroGradient",
        description="Zero normal gradient condition",
        typical_use_cases=[
            "Outlet boundary (zero gradient assumption)",
            "Symmetry planes",
            "Far-field boundaries",
        ],
        parameters=[],
        field_types=["all"],
        example="type zeroGradient;",
    ),
    "fixedFluxPressure": BoundaryCondition(
        name="fixedFluxPressure",
        description="Pressure BC for velocity-pressure coupling",
        typical_use_cases=[
            "Walls with viscous effects",
            "Internal boundaries in coupled solvers",
        ],
        parameters=[],
        field_types=["p", "p_rgh"],
        example="type fixedFluxPressure;\n",
    ),
    "wallFunctions": BoundaryCondition(
        name="nutUWallFunction",
        description="Wall function for turbulent kinetic energy viscosity",
        typical_use_cases=[
            "Wall-bounded turbulent flows",
            "High Reynolds number flows",
        ],
        parameters=["Cmu", "kappa", "E"],
        field_types=["nut"],
        example='type nutUWallFunction;\nCmu 0.09;\nkappa 0.41;\nE 9.8;',
    ),
    "turbulentIntensityKineticEnergyInlet": BoundaryCondition(
        name="turbulentIntensityKineticEnergyInlet",
        description="Inlet BC for k based on turbulent intensity",
        typical_use_cases=[
            "Turbulent inlet specification",
            "Experimental data with turbulence intensity",
        ],
        parameters=["intensity", "U_ref"],
        field_types=["k"],
        example="type turbulentIntensityKineticEnergyInlet;\nintensity 0.05;\nU_ref 10;",
    ),
    "symmetry": BoundaryCondition(
        name="symmetry",
        description="Symmetry plane boundary condition",
        typical_use_cases=[
            "Symmetric flow domains",
            "Domain reduction for computational efficiency",
        ],
        parameters=[],
        field_types=["all"],
        example="type symmetry;",
    ),
    "cyclicAMI": BoundaryCondition(
        name="cyclicAMI",
        description="Arbitrary mesh interface for rotational periodicity",
        typical_use_cases=[
            "Turbomachinery (pumps, turbines)",
            "Rotating machinery",
        ],
        parameters=["neighbourPatch"],
        field_types=["all"],
        example="type cyclicAMI;\nneighbourPatch inlet;",
    ),
}


def get_bc_by_name(name: str) -> BoundaryCondition | None:
    """Get boundary condition information by name."""
    return BOUNDARY_CONDITIONS.get(name)


def get_bc_for_field(field_name: str) -> List[str]:
    """Get suitable BCs for a given field."""
    suitable_bcs = []
    for bc_name, bc in BOUNDARY_CONDITIONS.items():
        if field_name in bc.field_types or "all" in bc.field_types:
            suitable_bcs.append(bc_name)
    return suitable_bcs


def list_all_bcs() -> List[str]:
    """List all available boundary conditions."""
    return list(BOUNDARY_CONDITIONS.keys())
