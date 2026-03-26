import keyword
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .structs import CallType, InputField, ModelStep, OutputField, Workflow
from .utils import detect_cycles

SUPPORTED_TYPES = {"str", "int", "float", "bool", "list[str]", "list[int]", "list[float]", "list[bool]"}

# Constants for validation
MAX_FIELD_NAME_LENGTH = 50
MAX_DESCRIPTION_LENGTH = 200
MAX_SYSTEM_PROMPT_LENGTH = 4000
MAX_TEMPERATURE = 10.0

from loguru import logger


class ValidationErrorType(Enum):
    """Types of validation errors that can occur"""

    INPUTS = "inputs"
    OUTPUTS = "outputs"
    STEP = "step"
    DAG = "dag"
    VARIABLE = "variable"
    TYPE = "type"
    GENERAL = "general"
    NAMING = "naming"
    LENGTH = "length"
    RANGE = "range"
    UNSUPPORTED_MODEL = "unsupported_model"


@dataclass
class ValidationError:
    """Represents a validation error with type and message"""

    error_type: ValidationErrorType
    message: str
    step_id: Optional[str] = None
    field_name: Optional[str] = None

    def __str__(self):
        subject = ""
        if self.step_id:
            subject = f"Model step '{self.step_id}'"
        if self.field_name:
            if self.step_id:
                subject = f"Field '{self.step_id}.{self.field_name}'"
            else:
                subject = f"Field '{self.field_name}'"
        return f"{self.error_type.value}: {subject} - {self.message}"


class WorkflowValidationError(ValueError):
    """Base class for workflow validation errors"""

    def __init__(self, errors: list[ValidationError]):
        self.errors = errors
        super().__init__(f"Workflow validation failed with {len(errors)} errors")


def _parse_variable_reference(var: str) -> tuple[Optional[str], str]:
    """Extracts step_id and field_name from variable reference"""
    parts = var.split(".")
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def _get_step_dependencies(step: ModelStep) -> set[str]:
    """Gets set of step IDs that this step depends on"""
    deps = set()
    for field in step.input_fields:
        step_id, _ = _parse_variable_reference(field.variable)
        if step_id:
            deps.add(step_id)
    return deps


def create_step_dep_graph(workflow: Workflow) -> dict[str, set[str]]:
    """Creates a dependency graph of steps"""
    dep_graph: dict[str, set[str]] = {}
    for step_id, step in workflow.steps.items():
        dep_graph[step_id] = _get_step_dependencies(step)
    return dep_graph


class WorkflowValidator:
    """Validates workflows for correctness and consistency"""

    def __init__(
        self,
        min_temperature: float = 0,
        max_temperature: float = MAX_TEMPERATURE,
        max_field_name_length: int = MAX_FIELD_NAME_LENGTH,
        max_description_length: int = MAX_DESCRIPTION_LENGTH,
        max_system_prompt_length: int = MAX_SYSTEM_PROMPT_LENGTH,
        allowed_model_names: Optional[list[str]] = None,
        required_input_vars: Optional[list[str]] = None,
        required_output_vars: Optional[list[str]] = None,
    ):
        self.errors: list[ValidationError] = []
        self.workflow: Optional[Workflow] = None
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.max_field_name_length = max_field_name_length
        self.max_description_length = max_description_length
        self.max_system_prompt_length = max_system_prompt_length
        self.required_input_vars = required_input_vars
        self.required_output_vars = required_output_vars
        self.allowed_model_names = set(allowed_model_names) if allowed_model_names else None

    def validate(self, workflow: Workflow, allow_empty: bool = False) -> bool:
        validated = self._validate(workflow, allow_empty)
        if not validated:
            raise WorkflowValidationError(self.errors)
        return True

    def _validate(self, workflow: Workflow, allow_empty: bool = False) -> bool:
        """Main validation entry point
        Args:
            workflow: The workflow to validate.
            allow_empty: If True, empty workflow is allowed. This flag is used to validate the intermediate states while User edits the workflow.
        """
        self.errors = []
        self.workflow = workflow

        # Basic workflow validation
        if not self._validate_workflow_basic(workflow, allow_empty):
            return False

        # If it's a single-step workflow, use simple validation
        if len(workflow.steps) == 1:
            return self.validate_simple_workflow(workflow, allow_empty)

        # Otherwise use complex validation
        return self.validate_complex_workflow(workflow, allow_empty)

    def _validate_required_inputs(self, workflow: Workflow, allow_empty: bool = False) -> bool:
        """Validates that the workflow has the correct inputs"""

        required_input_vars = self.required_input_vars or []
        input_vars = set(workflow.inputs)
        for req_var in required_input_vars:
            if req_var in input_vars:
                continue
            self.errors.append(
                ValidationError(ValidationErrorType.INPUTS, f"Workflow must have '{req_var}' as an input")
            )
            return False

        for input_var in input_vars:
            if not self._is_valid_external_input(input_var):
                self.errors.append(
                    ValidationError(ValidationErrorType.VARIABLE, f"Invalid input variable format: {input_var}")
                )
                return False
        return True

    def _validate_required_outputs(self, workflow: Workflow, allow_empty: bool = False) -> bool:
        """Validates that the workflow has the correct outputs"""

        required_output_vars = self.required_output_vars or []
        output_vars = set(workflow.outputs)
        for req_var in required_output_vars:
            if req_var in output_vars:
                continue
            self.errors.append(
                ValidationError(ValidationErrorType.OUTPUTS, f"Workflow must produce '{req_var}' as an output")
            )
            return False

        # Validate output variables
        for output_name, output_var in workflow.outputs.items():
            logger.debug(f"Output name: {output_name}, Output var: {output_var}")
            if not output_var:
                if allow_empty:
                    continue
                self.errors.append(
                    ValidationError(ValidationErrorType.VARIABLE, f"Missing output variable for {output_name}")
                )
                return False

            # Check if output variable references a valid step output
            if not self._is_valid_variable_reference(output_var):
                self.errors.append(
                    ValidationError(ValidationErrorType.VARIABLE, f"Invalid output variable reference: {output_var}")
                )
                return False

            # Verify the output field exists in the referenced step
            step_id, field_name = _parse_variable_reference(output_var)
            logger.debug(f"Step ID: {step_id}, Field name: {field_name}, Workflow steps: {workflow.steps.keys()}")
            if step_id not in workflow.steps:
                self.errors.append(
                    ValidationError(ValidationErrorType.VARIABLE, f"Referenced model step '{step_id}' not found")
                )
                return False

            ref_step = workflow.steps[step_id]
            if not any(field.name == field_name for field in ref_step.output_fields):
                self.errors.append(
                    ValidationError(
                        ValidationErrorType.VARIABLE,
                        f"Output field '{field_name}' not found in model step '{step_id}'",
                        step_id,
                        field_name,
                    )
                )
                return False
        return True

    def validate_input_outputs(self, workflow: Workflow, allow_empty: bool = False) -> bool:
        """Validates the input and output variables"""

        self._validate_required_inputs(workflow, allow_empty)
        self._validate_required_outputs(workflow, allow_empty)

        # Check for atleast one input
        if not workflow.inputs:
            self.errors.append(ValidationError(ValidationErrorType.GENERAL, "Workflow must contain at least one input"))

        # Check for atleast one output
        if not workflow.outputs:
            self.errors.append(
                ValidationError(ValidationErrorType.GENERAL, "Workflow must contain at least one output")
            )

        return len(self.errors) == 0

    def validate_simple_workflow(self, workflow: Workflow, allow_empty: bool = False) -> bool:
        """Validates a single-step workflow"""
        if not self.workflow:
            return False

        # Get the single step
        step = next(iter(workflow.steps.values()))

        # Validate the step itself
        if not self._validate_step(step, allow_empty):
            return False

        return True

    def validate_complex_workflow(self, workflow: Workflow, allow_empty: bool = False) -> bool:
        """Validates a multi-step workflow"""
        if not self.workflow:
            return False

        # Validate each step
        for step in workflow.steps.values():
            if not self._validate_step(step, allow_empty):
                return False

        dep_graph = create_step_dep_graph(workflow)
        if cycle_step_id := detect_cycles(dep_graph):
            self.errors.append(
                ValidationError(
                    ValidationErrorType.DAG, f"Circular dependency detected involving step: {cycle_step_id}"
                )
            )
            return False

        # Check for orphaned steps (steps that aren't used by any other step)
        used_steps = set()
        for deps in dep_graph.values():
            used_steps.update(deps)
        for step_id in workflow.steps:
            if step_id not in used_steps and not any(
                output_var and _parse_variable_reference(output_var)[0] == step_id
                for output_var in workflow.outputs.values()
            ):
                self.errors.append(ValidationError(ValidationErrorType.DAG, f"Orphaned step detected: {step_id}"))
                return False

        # Validate variable dependencies
        if not self._validate_variable_dependencies(workflow):
            return False

        return True

    def _validate_workflow_basic(self, workflow: Workflow, allow_empty: bool = False) -> bool:
        """Validates basic workflow properties"""

        # Check the workflow inputs and outputs
        if not self.validate_input_outputs(workflow, allow_empty):
            return False

        # Check for empty workflow
        if not workflow.steps:
            if allow_empty:
                return True
            self.errors.append(ValidationError(ValidationErrorType.GENERAL, "Workflow must contain at least one step"))
            return False

        # Check for step ID consistency
        for step_id, step in workflow.steps.items():
            if step_id != step.id:
                self.errors.append(
                    ValidationError(ValidationErrorType.STEP, f"Step ID mismatch: {step_id} != {step.id}", step_id)
                )
                return False
        return True

    def _validate_step(self, step: ModelStep, allow_empty: bool = False) -> bool:
        """Validates a single step"""
        # Validate required fields

        model_name = step.get_full_model_name()

        if model_name == "/" and not allow_empty:
            self.errors.append(
                ValidationError(ValidationErrorType.STEP, "Model name and provider cannot be empty", step.id)
            )
            return False

        # Check if the model names are allowed
        if self.allowed_model_names and model_name not in self.allowed_model_names:
            self.errors.append(
                ValidationError(
                    ValidationErrorType.UNSUPPORTED_MODEL, f"Model name '{model_name}' is not allowed", step.id
                )
            )
            return False

        if not step.id or not step.call_type:
            self.errors.append(ValidationError(ValidationErrorType.STEP, "Step missing required fields", step.id))
            return False

        # Validate step ID and name
        if not self._is_valid_identifier(step.id):
            self.errors.append(
                ValidationError(
                    ValidationErrorType.NAMING,
                    f"Invalid step ID format: {step.id}. Must be a valid identifier.",
                    step.id,
                )
            )
            return False

        # Validate temperature for LLM call type
        if step.call_type == CallType.LLM:
            if step.temperature is None:
                self.errors.append(
                    ValidationError(ValidationErrorType.STEP, "LLM step must specify temperature", step.id)
                )
                return False

            if not self.min_temperature <= step.temperature <= self.max_temperature:
                self.errors.append(
                    ValidationError(
                        ValidationErrorType.RANGE,
                        f"Temperature must be between {self.min_temperature} and {self.max_temperature}",
                        step.id,
                    )
                )
                return False

        # Validate system prompt for LLM call type
        if step.call_type == CallType.LLM:
            if not step.system_prompt:
                self.errors.append(
                    ValidationError(ValidationErrorType.STEP, "LLM step must specify system prompt", step.id)
                )
                return False

            if len(step.system_prompt) > self.max_system_prompt_length:
                self.errors.append(
                    ValidationError(
                        ValidationErrorType.LENGTH,
                        f"System prompt exceeds maximum length of {self.max_system_prompt_length} characters",
                        step.id,
                    )
                )
                return False

        # Validate input fields
        input_names = set()
        for field in step.input_fields:
            if not self._validate_input_field(field, allow_empty):
                return False
            if field.name and field.name in input_names:
                self.errors.append(
                    ValidationError(
                        ValidationErrorType.STEP, f"Duplicate input field name: {field.name}", step.id, field.name
                    )
                )
                return False
            if field.name:
                input_names.add(field.name)

        # Validate output fields
        output_names = set()
        for field in step.output_fields:
            if not self._validate_output_field(field, allow_empty):
                return False
            if field.name and field.name in output_names:
                self.errors.append(
                    ValidationError(
                        ValidationErrorType.STEP, f"Duplicate output field name: {field.name}", step.id, field.name
                    )
                )
                return False
            if field.name:
                output_names.add(field.name)

        return True

    def _validate_input_field(self, field: InputField, allow_empty: bool = False) -> bool:
        """Validates an input field"""
        # Validate required fields (skip when allow_empty for intermediate editing state)
        if not allow_empty and (not field.name or not field.description or not field.variable):
            self.errors.append(
                ValidationError(ValidationErrorType.STEP, "Input field missing required fields", field_name=field.name)
            )
            return False
        if allow_empty and not field.name and not field.description and not (field.variable or "").strip():
            # Fully empty row is ok when editing
            return True

        # Validate field name
        if not self._is_valid_identifier(field.name, allow_empty):
            self.errors.append(
                ValidationError(
                    ValidationErrorType.NAMING,
                    f"Invalid field name format: {field.name}. Must be a valid Python identifier.",
                    field_name=field.name,
                )
            )
            return False

        # Validate field name length
        if len(field.name) > self.max_field_name_length:
            self.errors.append(
                ValidationError(
                    ValidationErrorType.LENGTH,
                    f"Field name exceeds maximum length of {self.max_field_name_length} characters",
                    field_name=field.name,
                )
            )
            return False

        # Validate description length
        if len(field.description) > self.max_description_length:
            self.errors.append(
                ValidationError(
                    ValidationErrorType.LENGTH,
                    f"Description exceeds maximum length of {self.max_description_length} characters",
                    field_name=field.name,
                )
            )
            return False

        # Validate variable reference
        if not self._is_valid_variable_reference(field.variable):
            self.errors.append(
                ValidationError(
                    ValidationErrorType.VARIABLE,
                    f"Invalid variable reference: {field.variable}",
                    field_name=field.name,
                )
            )
            return False

        return True

    def _validate_output_field(self, field: OutputField, allow_empty: bool = False) -> bool:
        """Validates an output field"""
        # Validate required fields (skip when allow_empty for intermediate editing state)
        if not allow_empty and (not field.name or not field.description):
            self.errors.append(
                ValidationError(ValidationErrorType.STEP, "Output field missing required fields", field_name=field.name)
            )
            return False
        if allow_empty and not field.name and not field.description:
            # Fully empty row is ok when editing
            return True

        # Validate field name
        if not self._is_valid_identifier(field.name, allow_empty):
            self.errors.append(
                ValidationError(
                    ValidationErrorType.NAMING,
                    f"Invalid field name format: {field.name}. Must be a valid Python identifier.",
                    field_name=field.name,
                )
            )
            return False

        # Validate field name length
        if len(field.name) > self.max_field_name_length:
            self.errors.append(
                ValidationError(
                    ValidationErrorType.LENGTH,
                    f"Field name exceeds maximum length of {self.max_field_name_length} characters",
                    field_name=field.name,
                )
            )
            return False

        # Validate description length
        if len(field.description) > self.max_description_length:
            self.errors.append(
                ValidationError(
                    ValidationErrorType.LENGTH,
                    f"Description exceeds maximum length of {self.max_description_length} characters",
                    field_name=field.name,
                )
            )
            return False

        # Validate type
        if field.type not in SUPPORTED_TYPES:
            self.errors.append(
                ValidationError(
                    ValidationErrorType.TYPE, f"Unsupported output type: {field.type}", field_name=field.name
                )
            )
            return False

        return True

    def _validate_simple_workflow_variables(self, workflow: Workflow) -> bool:
        """Validates variables in a simple workflow"""
        step = next(iter(workflow.steps.values()))

        # Validate input variables
        for input_var in workflow.inputs:
            if not self._is_valid_external_input(input_var):
                self.errors.append(
                    ValidationError(ValidationErrorType.VARIABLE, f"Invalid input variable format: {input_var}")
                )
                return False

        # Validate output variables
        for output_name, output_var in workflow.outputs.items():
            if output_var and not self._is_valid_variable_reference(output_var):
                self.errors.append(
                    ValidationError(ValidationErrorType.VARIABLE, f"Invalid output variable reference: {output_var}")
                )
                return False

        return True

    def _validate_variable_dependencies(self, workflow: Workflow) -> bool:
        """Validates variable dependencies between steps"""
        # Build variable dependency graph
        var_graph: dict[str, set[str]] = {}

        def create_var_dep_graph(workflow: Workflow) -> dict[str, set[str]]:
            var_graph: dict[str, set[str]] = {}
            for step_id, step in workflow.steps.items():
                for field in step.input_fields:
                    if field.variable not in var_graph:
                        var_graph[field.variable] = set()
                    # Add dependency from input variable to step's outputs
                    for output in step.output_fields:
                        var_graph[field.variable].add(f"{step_id}.{output.name}")
            return var_graph

        # Check for cycles in variable dependencies
        var_graph = create_var_dep_graph(workflow)
        if cycle_var := detect_cycles(var_graph):
            self.errors.append(
                ValidationError(ValidationErrorType.VARIABLE, f"Circular variable dependency detected: {cycle_var}")
            )
            return False

        # Validate external input existence
        external_inputs = set(workflow.inputs)
        for step in workflow.steps.values():
            for field in step.input_fields:
                step_id, field_name = _parse_variable_reference(field.variable)
                if not step_id and field_name not in external_inputs:
                    self.errors.append(
                        ValidationError(
                            ValidationErrorType.VARIABLE,
                            f"External input '{field_name}' not found in workflow inputs",
                            field_name=field_name,
                        )
                    )
                    return False

        return True

    def _is_valid_variable_reference(self, var: str | None, allow_empty: bool = True) -> bool:
        """Validates if a variable reference is properly formatted"""
        if not self.workflow:
            return False
        if var is None:
            return allow_empty
        parts = var.split(".")
        if len(parts) == 1:
            return True  # External input
        if len(parts) != 2:
            return False
        step_id, field_name = parts
        return step_id in self.workflow.steps and any(
            field.name == field_name for field in self.workflow.steps[step_id].output_fields
        )

    def _is_valid_external_input(self, var: str) -> bool:
        """Validates if a variable is a valid external input"""
        if not var:
            return False
        if not self._is_valid_identifier(var):
            return False
        if keyword.iskeyword(var):
            return False
        if "." in var:  # External inputs should not contain dots
            return False
        return True

    def _is_valid_identifier(self, name: str, allow_empty: bool = False) -> bool:
        """Validates if a string is a valid Python identifier"""
        if name and name.strip():
            return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))
        return allow_empty
