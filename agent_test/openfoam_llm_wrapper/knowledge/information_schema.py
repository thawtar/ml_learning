"""Schema and validation for case information collection."""

from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Enumeration of field types."""

    STRING = "string"
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"


class FieldValidator:
    """Validator for individual fields."""

    def __init__(
        self,
        field_type: FieldType,
        required: bool = True,
        description: str = "",
        valid_values: Optional[List[Any]] = None,
    ):
        """
        Initialize field validator.

        Args:
            field_type: Type of the field
            required: Whether field is required
            description: Description of the field
            valid_values: List of valid values (for enum-like fields)
        """
        self.field_type = field_type
        self.required = required
        self.description = description
        self.valid_values = valid_values

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a value against this field's schema.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            if self.required:
                return False, f"Required field is missing"
            return True, None

        # Type checking
        if self.field_type == FieldType.STRING:
            if not isinstance(value, str):
                return False, f"Expected string, got {type(value).__name__}"
        elif self.field_type == FieldType.FLOAT:
            try:
                float(value)
            except (ValueError, TypeError):
                return False, f"Expected numeric value, got {value}"
        elif self.field_type == FieldType.INTEGER:
            try:
                int(value)
            except (ValueError, TypeError):
                return False, f"Expected integer, got {value}"
        elif self.field_type == FieldType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Expected boolean, got {type(value).__name__}"
        elif self.field_type == FieldType.LIST:
            if not isinstance(value, list):
                return False, f"Expected list, got {type(value).__name__}"
        elif self.field_type == FieldType.DICT:
            if not isinstance(value, dict):
                return False, f"Expected dictionary, got {type(value).__name__}"

        # Value checking
        if self.valid_values and value not in self.valid_values:
            return (
                False,
                f"Value '{value}' not in valid options: {self.valid_values}",
            )

        return True, None


# Schema definitions for each category
PHYSICS_SCHEMA = {
    "flow_type": FieldValidator(
        FieldType.STRING,
        required=True,
        description="Type of flow: incompressible or compressible",
        valid_values=["incompressible", "compressible"],
    ),
    "time_type": FieldValidator(
        FieldType.STRING,
        required=True,
        description="Steady-state or transient",
        valid_values=["steady", "transient"],
    ),
    "reynolds_number": FieldValidator(
        FieldType.FLOAT,
        required=False,
        description="Characteristic Reynolds number",
    ),
    "turbulence_model": FieldValidator(
        FieldType.STRING,
        required=False,
        description="Turbulence model (laminar, kEpsilon, kOmega, etc)",
        valid_values=[
            "laminar",
            "kEpsilon",
            "kOmega",
            "kOmegaSST",
            "spalartAllmaras",
            "LES",
        ],
    ),
    "special_physics": FieldValidator(
        FieldType.LIST,
        required=False,
        description="Special physics like heat transfer, combustion, etc",
    ),
}

GEOMETRY_SCHEMA = {
    "description": FieldValidator(
        FieldType.STRING,
        required=True,
        description="Geometry description including dimensions",
    ),
    "dimension": FieldValidator(
        FieldType.STRING,
        required=False,
        description="2D or 3D",
        valid_values=["2D", "3D"],
    ),
    "characteristic_length": FieldValidator(
        FieldType.FLOAT,
        required=False,
        description="Characteristic length dimension",
    ),
    "symmetries": FieldValidator(
        FieldType.LIST,
        required=False,
        description="Symmetry planes (e.g., x-symmetry, y-symmetry)",
    ),
}

BOUNDARY_CONDITIONS_SCHEMA = {
    "patches": FieldValidator(
        FieldType.DICT,
        required=True,
        description="Dictionary of boundary condition patches",
    ),
}

FLUID_PROPERTIES_SCHEMA = {
    "density": FieldValidator(
        FieldType.FLOAT,
        required=False,
        description="Fluid density (kg/m³)",
    ),
    "viscosity": FieldValidator(
        FieldType.FLOAT,
        required=False,
        description="Dynamic viscosity (Pa·s)",
    ),
    "specific_heat": FieldValidator(
        FieldType.FLOAT,
        required=False,
        description="Specific heat (J/kg·K)",
    ),
    "thermal_conductivity": FieldValidator(
        FieldType.FLOAT,
        required=False,
        description="Thermal conductivity (W/m·K)",
    ),
}

SOLVER_SCHEMA = {
    "solver": FieldValidator(
        FieldType.STRING,
        required=True,
        description="OpenFOAM solver name",
        valid_values=[
            "simpleFoam",
            "pimpleFoam",
            "buoyantSimpleFoam",
            "rhoCentralFoam",
            "interFoam",
            "chtMultiRegionSimpleFoam",
        ],
    ),
    "reason": FieldValidator(
        FieldType.STRING,
        required=False,
        description="Reason for solver selection",
    ),
}

MESH_SCHEMA = {
    "type": FieldValidator(
        FieldType.STRING,
        required=False,
        description="Mesh generation method",
        valid_values=["blockMesh", "snappyHexMesh", "external"],
    ),
    "cell_count_estimate": FieldValidator(
        FieldType.INTEGER,
        required=False,
        description="Estimated number of cells",
    ),
    "refinement_regions": FieldValidator(
        FieldType.LIST,
        required=False,
        description="Regions requiring mesh refinement",
    ),
}

SIMULATION_GOALS_SCHEMA = {
    "target_outputs": FieldValidator(
        FieldType.LIST,
        required=False,
        description="What to compute (drag, lift, pressure_drop, etc)",
    ),
    "convergence_criteria": FieldValidator(
        FieldType.FLOAT,
        required=False,
        description="Convergence tolerance",
    ),
    "simulation_duration": FieldValidator(
        FieldType.FLOAT,
        required=False,
        description="Simulation time for transient cases (s)",
    ),
}

ADVANCED_SCHEMA = {
    "relaxation_factors": FieldValidator(
        FieldType.DICT,
        required=False,
        description="Custom relaxation factors",
    ),
    "solver_tolerances": FieldValidator(
        FieldType.DICT,
        required=False,
        description="Custom solver tolerances",
    ),
    "numerical_schemes": FieldValidator(
        FieldType.DICT,
        required=False,
        description="Custom numerical schemes",
    ),
}

# Complete schema mapping
FULL_SCHEMA = {
    "physics": PHYSICS_SCHEMA,
    "geometry": GEOMETRY_SCHEMA,
    "boundary_conditions": BOUNDARY_CONDITIONS_SCHEMA,
    "fluid_properties": FLUID_PROPERTIES_SCHEMA,
    "solver": SOLVER_SCHEMA,
    "mesh": MESH_SCHEMA,
    "simulation_goals": SIMULATION_GOALS_SCHEMA,
    "advanced": ADVANCED_SCHEMA,
}

# Required categories (must have at least one entry)
REQUIRED_CATEGORIES = ["physics", "geometry", "boundary_conditions", "solver"]

# Required fields per category (must be present and non-empty)
REQUIRED_FIELDS_BY_CATEGORY = {
    "physics": ["flow_type", "time_type"],
    "geometry": ["description"],
    "boundary_conditions": ["patches"],
    "solver": ["solver"],
}


class ValidationResult:
    """Result of validation."""

    def __init__(self):
        """Initialize validation result."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.is_valid: bool = True

    def add_error(self, message: str) -> None:
        """Add an error."""
        self.errors.append(message)
        self.is_valid = False
        logger.warning(f"Validation error: {message}")

    def add_warning(self, message: str) -> None:
        """Add a warning."""
        self.warnings.append(message)
        logger.info(f"Validation warning: {message}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_field(
    category: str, field_name: str, value: Any
) -> tuple[bool, Optional[str]]:
    """
    Validate a single field.

    Args:
        category: Category name
        field_name: Field name
        value: Value to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if category not in FULL_SCHEMA:
        return False, f"Unknown category: {category}"

    schema = FULL_SCHEMA[category]

    if field_name not in schema:
        return False, f"Unknown field: {category}.{field_name}"

    validator = schema[field_name]
    return validator.validate(value)


def validate_category(category: str, data: Dict[str, Any]) -> ValidationResult:
    """
    Validate all fields in a category.

    Args:
        category: Category name
        data: Dictionary of field data

    Returns:
        ValidationResult object
    """
    result = ValidationResult()

    if category not in FULL_SCHEMA:
        result.add_error(f"Unknown category: {category}")
        return result

    schema = FULL_SCHEMA[category]

    # Check each field in the schema
    for field_name, validator in schema.items():
        value = data.get(field_name)

        is_valid, error_msg = validator.validate(value)

        if not is_valid:
            if validator.required:
                result.add_error(
                    f"{category}.{field_name}: {error_msg}"
                )
            else:
                result.add_warning(
                    f"{category}.{field_name}: {error_msg}"
                )

    return result


def validate_case_info(case_info: Dict[str, Any]) -> ValidationResult:
    """
    Validate complete case information.

    Args:
        case_info: The collected case information dictionary

    Returns:
        ValidationResult object
    """
    result = ValidationResult()
    collected_data = case_info.get("collected_data", {})

    # Check required categories
    for category in REQUIRED_CATEGORIES:
        if category not in collected_data:
            result.add_error(f"Missing required category: {category}")
            continue

        category_data = collected_data[category]

        # Check required fields in category
        required_fields = REQUIRED_FIELDS_BY_CATEGORY.get(category, [])
        for field_name in required_fields:
            if field_name not in category_data or not category_data[field_name]:
                result.add_error(
                    f"Missing required field: {category}.{field_name}"
                )

    # Validate each category
    for category, data in collected_data.items():
        if data:  # Only validate non-empty categories
            category_result = validate_category(category, data)
            result.errors.extend(category_result.errors)
            result.warnings.extend(category_result.warnings)

    # Update overall validity
    result.is_valid = len(result.errors) == 0

    return result


def get_field_description(category: str, field_name: str) -> Optional[str]:
    """
    Get the description of a field.

    Args:
        category: Category name
        field_name: Field name

    Returns:
        Field description or None
    """
    if category not in FULL_SCHEMA:
        return None

    schema = FULL_SCHEMA[category]

    if field_name not in schema:
        return None

    return schema[field_name].description


def get_valid_values(category: str, field_name: str) -> Optional[List[Any]]:
    """
    Get valid values for a field.

    Args:
        category: Category name
        field_name: Field name

    Returns:
        List of valid values or None
    """
    if category not in FULL_SCHEMA:
        return None

    schema = FULL_SCHEMA[category]

    if field_name not in schema:
        return None

    return schema[field_name].valid_values


def get_required_fields() -> Dict[str, List[str]]:
    """
    Get list of all required fields by category.

    Returns:
        Dictionary mapping category to required field names
    """
    return REQUIRED_FIELDS_BY_CATEGORY.copy()
