"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_case_dir(tmp_path):
    """Create a temporary OpenFOAM case directory structure."""
    case_path = tmp_path / "test_case"
    case_path.mkdir()

    # Create standard directories
    for dirname in ["0", "constant", "system"]:
        (case_path / dirname).mkdir()

    return case_path


@pytest.fixture
def sample_u_file():
    """Sample velocity field file content."""
    return """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }

    outlet
    {
        type            zeroGradient;
    }

    wall
    {
        type            noSlip;
    }
}
"""


@pytest.fixture
def sample_p_file():
    """Sample pressure field file content."""
    return """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }

    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }

    wall
    {
        type            zeroGradient;
    }
}
"""
