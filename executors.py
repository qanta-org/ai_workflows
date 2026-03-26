"""
Core workflow execution functionality.

This module handles the execution of defined workflows, including input processing,
dependency-based execution order, model calling, and output collection. It integrates
with the litellm library to handle model interactions.

Key components:
- Utility functions for input/output transformation
- Input processing and validation
- Model step execution with support for log probabilities
- Complete workflow execution with dependency resolution
- Support for both simple (single-step) and multi-step workflows
- Structured output collection with intermediate results

The module orchestrates the execution of steps in the correct order based on their
dependencies and manages the flow of data between steps. It supports:
- Full content tracking for debugging
- Log probability calculation for specific steps
- Flexible input/output transformations
- Error handling and validation
"""

from typing import Any, TypedDict

import pydantic
from loguru import logger

from .configs import FUNCTION_MAP, TYPE_MAP
from .errors import WorkflowError
from .llms import completion
from .structs import InputField, ModelStep, OutputField, Workflow
from .utils import create_dependency_graph, topological_sort


def get_type(type_str: str) -> type:
    """
    Converts a type string to its corresponding Python type.

    This function maps type strings to their actual Python type objects. It first checks
    the TYPE_MAP dictionary for predefined mappings, and if not found, falls back to
    evaluating the type string directly.

    Args:
        type_str (str): A string representation of a type (e.g., "str", "int", "list[str]")

    Returns:
        type: The corresponding Python type object

    Note:
        Uses eval() for non-predefined types, which has security implications if used
        with untrusted input. This is intended for internal use with validated type strings.
    """
    return TYPE_MAP.get(type_str, eval(type_str))


def create_processed_inputs(model_step: ModelStep, available_vars: dict[str, Any]) -> dict[str, Any]:
    """
    Creates processed inputs for a model step.

    This function extracts and processes the required inputs for a model step based on
    its input field definitions. It retrieves values from the available variables dictionary
    and applies any specified transformations.

    Args:
        model_step (ModelStep): The model step for which to create processed inputs.
        available_vars (dict[str, Any]): Dictionary of variables available for use as inputs.
                                         Keys are variable names, values are the variable values.

    Returns:
        dict[str, Any]: A dictionary of processed inputs ready for use by the model step.
                        Keys are input field names, values are the processed input values.

    Raises:
        WorkflowError: If a required variable is not found in available_vars,
                       or if a specified transformation function is not available.

    Example:
        >>> available_vars = {"step1.output": "Hello World"}
        >>> create_processed_inputs(model_step, available_vars)
        {"input_field_name": "HELLO WORLD"}  # If upper transformation was specified
    """
    processed_inputs: dict[str, Any] = {}
    for input_field in model_step.input_fields:
        var = input_field.variable
        value = available_vars[var]
        if input_field.func is not None:
            func = FUNCTION_MAP.get(input_field.func)
            func = func or eval(input_field.func)
            value = func(value)
        processed_inputs[input_field.name] = value
    return processed_inputs


class ModelStepResult(TypedDict):
    """
    Result of executing a model step.

    This TypedDict contains the outputs and metadata from executing a single model step,
    including the processed output values, the full response content, and log probability
    information when requested.

    Attributes:
        outputs (dict[str, Any]): A dictionary of processed outputs from the model step,
                                with keys matching the output field names.
        content (str | None): The full content of the model's response, only populated
                            if return_full_content is True.
        logprob (float | None): The log probability of the model step output, only populated
                               if logprobs is True.
    """

    # A dictionary of processed outputs from the model step,
    # with keys matching the output field names.
    outputs: dict[str, Any]

    # The full content of the model step.
    content: str | None

    # The log probability of the model step output if requested.
    logprob: float | None


class WorkflowOutput(TypedDict):
    """
    Result of executing a complete workflow.

    This TypedDict contains the outputs and metadata from executing a workflow,
    including final outputs, intermediate values, step contents, and log probabilities.

    Attributes:
        final_outputs (dict[str, Any]): The final output values produced by the workflow,
                                      with keys matching the names defined in workflow.outputs.
        intermediate_outputs (dict[str, Any]): All computed values during workflow execution,
                                             including both external inputs and outputs from all steps.
        step_contents (dict[str, Any]): Full response content for each step, keyed by step ID.
                                      Only populated if return_full_content is True.
        logprob (float | None): The log probability of the specified step's output.
                               Only populated if logprob_step is specified.
    """

    # A dictionary of the workflow's outputs, with keys matching the variables defined in workflow.outputs.
    final_outputs: dict[str, Any]

    # A dictionary of all computed values during workflow execution, including intermediate results.
    intermediate_outputs: dict[str, Any]

    # A dictionary of step contents, only populated if return_full_content is True.
    step_contents: dict[str, Any]

    # The log probability of the workflow output if requested.
    logprob: float | None


# %%
def execute_model_step(
    model_step: ModelStep,
    available_vars: dict[str, Any],
    return_full_content: bool = False,
    logprobs: bool = False,
) -> ModelStepResult:
    """
    Executes a model step using the provided available variables.

    This function handles the complete execution of a model step, including:
    1. Processing inputs using variable references and transformations
    2. Constructing the appropriate prompt for the model
    3. Calling the model via litellm with structured output
    4. Processing and validating the model's response
    5. Applying any output transformations

    The function supports different providers and model types through the litellm
    integration, allowing for a consistent interface regardless of the underlying model.

    Args:
        model_step (ModelStep): The model step to execute, containing model details,
                               input/output specifications, and system prompt.
        available_vars (dict[str, Any]): A dictionary of all variables available to this step,
                                        including outputs from previous steps and external inputs.
        return_full_content (bool, optional): If True, includes the full model response content
                                             in the result. Defaults to False.
        logprobs (bool, optional): If True, calculates and returns log probability information
                                  for the model response. Defaults to False.

    Returns:
        ModelStepResult: A TypedDict containing processed outputs, optional full content,
                        and optional log probability information.

    Raises:
        WorkflowError: If there's an error in input processing, model execution,
                      or output validation.

    Example:
        >>> step = ModelStep(
        ...     id="summarize",
        ...     model="gpt-3.5-turbo",
        ...     provider="openai",
        ...     call_type="llm",
        ...     system_prompt="Summarize the text",
        ...     input_fields=[InputField(name="text", variable="input_text", description="Text to summarize")],
        ...     output_fields=[OutputField(name="summary", type="str", description="Summary of the text")]
        ... )
        >>> result = execute_model_step(step, {"input_text": "Long text to be summarized..."})
        >>> summary = result["outputs"]["summary"]
    """
    # Ensure inputs are processed using the specified functions in input_fields.
    processed_inputs = create_processed_inputs(model_step, available_vars)

    # Extract images only when the step explicitly declares an image input variable.
    images = None
    image_field_names = [
        field.name
        for field in model_step.input_fields
        if field.variable in {"question_images", "leadin_images", "part_images", "images"}
    ]
    if image_field_names:
        collected_images: list[Any] = []
        for image_field_name in image_field_names:
            if image_field_name not in processed_inputs:
                continue
            value = processed_inputs.pop(image_field_name)
            if value is None:
                continue
            if isinstance(value, list):
                collected_images.extend(value)
            else:
                collected_images.append(value)
        if collected_images:
            images = collected_images
            logger.info(f"[Multimodal Debug] Step {model_step.id}: Found {len(images)} images via input variable(s)")
    
    if images is None:
        logger.info(f"[Multimodal Debug] Step {model_step.id}: No images found - text-only input")

    # Construct the input prompt for the model
    input_str = "\n".join(f"{k}: {v}" for k, v in processed_inputs.items())
    
    # Add note about images if present
    if images:
        input_str += f"\n\nNote: {len(images)} image(s) are included with this question. Please analyze the image(s) along with the text to answer the question."
    
    step_result = f"Inputs: \n{input_str}"

    # Define the expected output fields and their types
    fields = {
        field.name: (get_type(field.type), pydantic.Field(..., description=field.description))
        for field in model_step.output_fields
    }
    ModelResponse = pydantic.create_model("ModelResponse", **fields)

    # Log image status before API call
    if images:
        logger.info(f"[Multimodal Debug] Step {model_step.id}: Calling {model_step.provider}/{model_step.model} with {len(images)} images: {images}")
    else:
        logger.info(f"[Multimodal Debug] Step {model_step.id}: Calling {model_step.provider}/{model_step.model} with no images (text-only)")
    
    # Execute the model step using litellm
    api_response = completion(
        model=f"{model_step.provider}/{model_step.model}",
        system=model_step.system_prompt,
        prompt=step_result,
        response_format=ModelResponse,
        temperature=model_step.temperature,
        logprobs=logprobs,
        images=images,
    )

    # Map the parsed response to the output fields
    outputs = {field.name: api_response["output"][field.name] for field in model_step.output_fields}
    result = ModelStepResult(outputs=outputs, content=None, logprob=None)
    if return_full_content:
        result["content"] = api_response["content"]
    if logprobs:
        result["logprob"] = api_response.get("logprob")
    return result


def execute_multi_step_workflow(
    workflow: Workflow,
    input_values: dict[str, Any],
    return_full_content: bool = False,
    logprob_step: str | None = None,
) -> WorkflowOutput:
    """
    Execute the given workflow as a computational graph.

    This function orchestrates the complete execution of a workflow by:

    1. Validating and populating initial values using the provided external inputs
    2. Building a dependency graph between workflow steps
    3. Determining a valid execution order using topological sorting
    4. Executing each step in the correct order, with inputs from previous steps
    5. Collecting and returning the final outputs

    The execution process ensures that all dependencies are satisfied before a step
    is executed, and that the data flows correctly between steps according to the
    variable references defined in each step's input fields.

    Args:
        workflow (Workflow): The workflow to execute, containing steps, their
                            dependencies, and input/output specifications.
        input_values (dict[str, Any]): External input values to be used by the workflow.
                                      Keys should match the required workflow.inputs.
        return_full_content (bool, optional): If True, returns the full content of each step.
                                             Defaults to False.
        logprob_step (str, optional): The ID of the step to use for log probability calculation.
                                      Defaults to None.

    Returns:
        WorkflowOutput: A dictionary of workflow outputs, including final outputs, intermediate outputs, and step contents.

    Raises:
        UnknownVariableError: If an input_field references a variable that is not
                             provided externally nor produced by any step.
        CyclicDependencyError: If the workflow contains a circular dependency that
                              prevents a valid execution order.
        FunctionNotFoundError: If a transformation function specified in input_fields.func
                              or output_fields.func is not available.
        WorkflowError: For any other workflow-related errors, such as missing required inputs.

    Example:
        >>> workflow = Workflow(
        ...     steps={
        ...         "extract": ModelStep(...),  # A step that extracts entities
        ...         "analyze": ModelStep(...)   # A step that analyzes the entities
        ...     },
        ...     inputs=["text"],
        ...     outputs={"sentiment": "analyze.sentiment", "entities": "extract.entities"}
        ... )
        >>> final_outputs, computed_values, step_contents = execute_workflow(workflow, {"text": "Apple is launching a new product tomorrow."})
        >>> print(final_outputs["sentiment"])
        "positive"
        >>> print(final_outputs["entities"])
        ["Apple", "product"]
    """
    # Step 1: Pre-populate computed values with external workflow inputs.
    computed_values: dict[str, Any] = {}
    for var in workflow.inputs:
        if var not in input_values:
            raise WorkflowError(f"Missing required workflow input: {var}")
        computed_values[var] = input_values[var]
    
    # Also include any additional input values (like images for multimodal support)
    # that aren't explicitly declared but may be used by steps
    for var, value in input_values.items():
        if var not in computed_values:
            computed_values[var] = value

    # Step 2: Build dependency graph among model steps.
    # For each step, examine its input_fields. If an input is not in the pre-populated external inputs,
    # then it is expected to be produced by some step. Otherwise, raise an error.
    dependencies = create_dependency_graph(workflow, input_values)

    # Step 3: Determine the execution order of the steps using topological sort.
    # Raises an error if a cycle is detected.
    execution_order = topological_sort(dependencies)

    # Step 4: Execute steps in topological order.
    step_contents: dict[str, Any] = {}
    logprob = None
    for step_id in execution_order:
        step = workflow.steps[step_id]
        return_logprobs = logprob_step == step_id
        # Execute the step
        result = execute_model_step(
            step, computed_values, return_full_content=return_full_content, logprobs=return_logprobs
        )
        if return_logprobs:
            logprob = result["logprob"]
        if return_full_content:
            step_contents[step_id] = result["content"]
        outputs = {f"{step_id}.{k}": v for k, v in result["outputs"].items()}
        computed_values.update(outputs)

    # Step 5: Gather and return workflow outputs.
    final_outputs: dict[str, Any] = {}
    for target, var in workflow.outputs.items():
        if var not in computed_values:
            raise WorkflowError(
                f"Workflow output variable {var} was not produced. Computed values: {computed_values.keys()}"
            )
        final_outputs[target] = computed_values[var]

    return WorkflowOutput(
        final_outputs=final_outputs,
        intermediate_outputs=computed_values,
        step_contents=step_contents,
        logprob=logprob,
    )


def execute_simple_workflow(
    workflow: Workflow,
    input_values: dict[str, Any],
    return_full_content: bool = False,
    logprob_step: bool | str = False,
) -> WorkflowOutput:
    """
    Execute a simple workflow with a single step.

    This is an optimized version of workflow execution for workflows containing only one step.
    It bypasses the dependency graph building and topological sorting steps, providing a more
    direct execution path for simple workflows.

    Args:
        workflow (Workflow): The workflow to execute, which must contain exactly one step.
        input_values (dict[str, Any]): External input values to be used by the workflow.
                                     Keys should match the required workflow.inputs.
        return_full_content (bool, optional): If True, includes the full model response content
                                            in the result. Defaults to False.
        logprobs (bool, optional): If True, calculates and returns log probability information
                                  for the model response. Defaults to False.

    Returns:
        WorkflowOutput: A TypedDict containing the workflow outputs, intermediate values,
                       optional step contents, and optional log probability information.

    Raises:
        WorkflowError: If the workflow has more than one step or if required inputs are missing.

    Example:
        >>> workflow = Workflow(
        ...     steps={"extract": ModelStep(...)},
        ...     inputs=["text"],
        ...     outputs={"entities": "extract.entities"}
        ... )
        >>> result = execute_simple_workflow(workflow, {"text": "Apple is launching a new product."})
        >>> entities = result["final_outputs"]["entities"]
    """
    if len(workflow.steps) != 1:
        raise WorkflowError("Simple workflow must have exactly one step")

    # Get the single step
    step = list(workflow.steps.values())[0]

    logprobs = logprob_step is True or logprob_step == step.id

    # Validate inputs
    for var in workflow.inputs:
        if var not in input_values:
            raise WorkflowError(f"Missing required workflow input: {var}")

    # Execute the step (input_values already contains all inputs including optional ones like images)
    step_result = execute_model_step(step, input_values, return_full_content=return_full_content, logprobs=logprobs)
    step_outputs = step_result["outputs"]
    step_contents = {step.id: step_result["content"]} if return_full_content else {}
    # Prepare the final outputs
    final_outputs = {}
    for target, var in workflow.outputs.items():
        if var.startswith(f"{step.id}."):
            output_key = var.split(".", 1)[1]
            if output_key in step_outputs:
                final_outputs[target] = step_outputs[output_key]
            else:
                raise WorkflowError(f"Workflow output variable {var} was not produced")
        else:
            raise WorkflowError(f"Invalid output mapping: {var} does not match step ID {step.id}")

    # Prepare computed values (prefixed with step ID)
    computed_values = input_values | {f"{step.id}.{k}": v for k, v in step_outputs.items()}

    return WorkflowOutput(
        final_outputs=final_outputs,
        intermediate_outputs=computed_values,
        step_contents=step_contents,
        logprob=step_result.get("logprob"),
    )


def execute_workflow(
    workflow: Workflow,
    input_values: dict[str, Any],
    return_full_content: bool = False,
    logprob_step: str | bool = False,
) -> WorkflowOutput:
    """
    Main entry point for executing workflows of any complexity.

    This function serves as a router that delegates to the appropriate specialized
    execution function based on the complexity of the workflow:
    - For single-step workflows, it calls execute_simple_workflow
    - For multi-step workflows, it calls execute_multi_step_workflow

    This abstraction allows callers to use a consistent interface regardless of
    the workflow's complexity.

    Args:
        workflow (Workflow): The workflow to execute, containing steps, their
                           dependencies, and input/output specifications.
        input_values (dict[str, Any]): External input values to be used by the workflow.
                                     Keys should match the required workflow.inputs.
        return_full_content (bool, optional): If True, includes the full model response
                                            content in the result. Defaults to False.
        logprob_step (str | bool, optional): Either a string with the ID of the step for which
                                           to calculate log probability, or a boolean flag.
                                           If False, no log probabilities are calculated.
                                           Defaults to False.

    Returns:
        WorkflowOutput: A TypedDict containing the workflow outputs, intermediate values,
                       optional step contents, and optional log probability information.

    Raises:
        WorkflowError: For any workflow-related errors, such as missing required inputs,
                      circular dependencies, or invalid variable references.

    Example:
        >>> workflow = Workflow(
        ...     steps={"extract": ModelStep(...), "analyze": ModelStep(...)},
        ...     inputs=["text"],
        ...     outputs={"sentiment": "analyze.sentiment"}
        ... )
        >>> result = execute_workflow(
        ...     workflow,
        ...     {"text": "Apple is launching a new product."},
        ...     return_full_content=True,
        ...     logprob_step="analyze"
        ... )
        >>> print(result["final_outputs"]["sentiment"])
        "positive"
    """
    if len(workflow.steps) > 1:
        return execute_multi_step_workflow(workflow, input_values, return_full_content, logprob_step)
    else:
        return execute_simple_workflow(workflow, input_values, return_full_content, logprob_step)


def run_examples():
    """
    Runs example workflows demonstrating key functionality and error handling.

    This function creates and executes three different example workflows to showcase:

    1. Successful workflow execution:
       - A linear two-step workflow with proper dependency flow
       - Input transformation using the 'upper' function
       - Output transformation using the 'lower' function
       - Proper variable passing between steps

    2. Cyclic dependency detection:
       - A workflow with two steps that depend on each other circularly
       - Demonstrates the error handling for cyclic dependencies
       - Shows how the system prevents infinite execution loops

    3. Unknown variable detection:
       - A workflow that references a variable not provided as input or by any step
       - Demonstrates validation of variable references
       - Shows error handling for missing dependencies

    Each example prints its result or the error encountered, making this function
    useful for testing and demonstration purposes.

    Returns:
        None: This function prints its results and doesn't return a value.
    """
    print("Example 1: Successful Workflow Execution")
    # Example 1: Simple linear workflow.
    # External input "input.value" is provided. Two steps:
    #  - step1 takes "input.value" and produces "step1.result".
    #  - step2 uses "step1.result" and produces "step2.final".

    workflow_success = Workflow(
        steps={
            "step1": ModelStep(
                id="step1",
                model="gpt-4o-mini",
                provider="OpenAI",
                call_type="llm",
                system_prompt="Step1 processing",
                input_fields=[InputField(name="value", description="Input value", variable="input.value")],
                output_fields=[OutputField(name="result", description="Processed result", type="str", func="upper")],
            ),
            "step2": ModelStep(
                id="step2",
                model="gpt-4o-mini",
                provider="OpenAI",
                call_type="llm",
                system_prompt="Step2 processing",
                input_fields=[InputField(name="result", description="Result from step1", variable="step1.result")],
                output_fields=[OutputField(name="final", description="Final output", type="str", func="lower")],
            ),
        },
        inputs=["input.value"],
        outputs={"final": "step2.final"},
    )
    input_values_success = {"input.value": "Hello, World!"}
    try:
        outputs = execute_workflow(workflow_success, input_values_success)
        print("Workflow outputs:", outputs)
    except WorkflowError as e:
        print("Workflow failed with error:", e)

    print("\nExample 2: Cyclic Dependency Workflow")
    # Example 2: Cyclic dependency.
    # stepA depends on an output from stepB and vice versa.
    workflow_cycle = Workflow(
        steps={
            "stepA": ModelStep(
                id="stepA",
                model="gpt-4o-mini",
                provider="OpenAI",
                call_type="llm",
                system_prompt="StepA processing",
                input_fields=[
                    InputField(name="input", description="Input from stepB", variable="stepB.output", func="identity")
                ],
                output_fields=[OutputField(name="output", description="Output from A", type="str", func="upper")],
            ),
            "stepB": ModelStep(
                id="stepB",
                model="gpt-4o-mini",
                provider="OpenAI",
                call_type="llm",
                system_prompt="StepB processing",
                input_fields=[
                    InputField(name="input", description="Input from stepA", variable="stepA.output", func="identity")
                ],
                output_fields=[OutputField(name="output", description="Output from B", type="str", func="upper")],
            ),
        },
        inputs=[],  # no external inputs
        outputs={"output": "stepB.output"},
    )
    try:
        outputs = execute_workflow(workflow_cycle, {})
        print("Workflow outputs:", outputs)
    except WorkflowError as e:
        print("Workflow failed with error:", e)

    print("\nExample 3: Unknown Variable Dependency Workflow")
    # Example 3: A workflow that references a variable not provided as an input or produced by any step.
    workflow_unknown = Workflow(
        steps={
            "stepX": ModelStep(
                id="stepX",
                model="gpt-4o-mini",
                provider="OpenAI",
                call_type="llm",
                system_prompt="StepX processing",
                input_fields=[
                    InputField(
                        name="input", description="Non-existent input", variable="nonexistent.value", func="identity"
                    )
                ],
                output_fields=[OutputField(name="output", description="Output from X", type="str", func="upper")],
            )
        },
        inputs=[],  # no external inputs
        outputs={"output": "stepX.output"},
    )
    try:
        outputs = execute_workflow(workflow_unknown, {})
        print("Workflow outputs:", outputs)
    except WorkflowError as e:
        print("Workflow failed with error:", e)


if __name__ == "__main__":
    # create example of model_step
    model_step = ModelStep(
        id="step1",
        model="gpt-4o-mini",
        provider="OpenAI",
        call_type="llm",
        system_prompt="You are a simple NLP tool that takes a string, and a number N, and return the first N entities in the string, and the total count of entities in the string.",
        input_fields=[
            InputField(name="sentence", description="The sentence to process", variable="sentence", func=None),
            InputField(name="n", description="The number of entities to return", variable="n", func=None),
        ],
        output_fields=[
            OutputField(
                name="entities",
                description="The first N entities in the string as a list of strings",
                type="list[str]",
                func=None,
            ),
            OutputField(name="count", description="The total count of entities in the string", type="int", func=None),
        ],
    )

    processed_inputs = {"sentence": "Abdul Akbar is a good person, but Jesus is the son of God.", "n": 3}
    processed_inputs = create_processed_inputs(model_step, processed_inputs)
    print(processed_inputs)

    run_examples()

# %%

# Example usage
if __name__ == "__main__":
    # Define a simple model step
    model_step = ModelStep(
        id="step1",
        model="gpt-4o-mini",
        provider="OpenAI",
        call_type="llm",
        system_prompt="You are a simple NLP tool that takes a string, and a number N, and return the first N entities in the string, and the total count of entities in the string.",
        input_fields=[
            InputField(name="sentence", description="The sentence to process", variable="sentence", func=None),
            InputField(name="n", description="The number of entities to return", variable="n", func=None),
        ],
        output_fields=[
            OutputField(
                name="entities",
                description="The first N entities in the string as a list of strings",
                type="list[str]",
                func=None,
            ),
            OutputField(name="count", description="The total count of entities in the string", type="int", func=None),
        ],
    )

    # Define processed inputs
    processed_inputs = {"sentence": "Abdul Akbar is a good person, but Jesus is the son of God.", "n": 3}

    # Execute the model step
    outputs = execute_model_step(model_step, processed_inputs)
    print(outputs)
