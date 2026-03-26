import time
from typing import Any, Iterable, TypedDict

from anthropic import APIStatusError
from cohere.core.api_error import ApiError
from loguru import logger
from openai import APIError

from .errors import ProviderAPIError, WorkflowExecutionError
from .executors import WorkflowOutput, execute_workflow
from .structs import TossupWorkflow, Workflow


def _get_workflow_response(
    workflow: Workflow, available_vars: dict[str, Any], logprob_step: bool | str = False
) -> tuple[WorkflowOutput, float]:
    """Get response from executing a complete workflow."""
    try:
        start_time = time.time()
        workflow_output = execute_workflow(
            workflow, available_vars, return_full_content=True, logprob_step=logprob_step
        )
        response_time = time.time() - start_time
        return workflow_output, response_time
    except APIStatusError as e:
        logger.error(f"Anthropic API error {e.status_code}: {e}")
        raise ProviderAPIError(workflow, "Anthropic", "Quota exceeded")
    except APIError as e:
        logger.error(f"OpenAI API error {e.status_code}: {e}")
        if e.code == 429:
            raise ProviderAPIError(workflow, "OpenAI", "Quota / Rate limit exceeded")
        raise ProviderAPIError(workflow, "OpenAI", e.message)
    except ApiError as e:
        logger.error(f"Cohere API error {e.status_code}: {e}")
        if e.status_code == 402:
            raise ProviderAPIError(workflow, "Cohere", "Payment required")
        raise ProviderAPIError(workflow, "Cohere", e.body["message"])
    except ValueError as e:
        logger.exception(f"Error Message: {e}")
        raise e
    except Exception as e:
        logger.exception(f"Error Message: {e}")
        raise WorkflowExecutionError(workflow)


class TossupResult(TypedDict):
    guess: str  # the model's best guess
    confidence: float  # confidence score (0-1)
    logprob: float | None  # log probability of the answer
    buzz: bool  # whether the agent buzzed
    question_fragment: str  # prefix of the question text so far
    run_idx: int  # 1-indexed question run index
    response_time: float
    step_contents: list[str]  # string content outputs of each step
    step_outputs: dict[str, Any]


class BonusResult(TypedDict):
    guess: str
    confidence: float
    logprob: float | None
    explanation: str
    response_time: float
    step_contents: list[str]
    step_outputs: dict[str, Any]


class QuizBowlTossupAgent:
    """Agent for handling tossup questions with multiple steps in the workflow."""

    external_input_variable = "question_text"
    output_variables = ["answer", "confidence"]

    def __init__(self, workflow: TossupWorkflow):
        """Initialize the multi-step tossup agent.

        Args:
            workflow: The workflow containing multiple steps
            buzz_threshold: Confidence threshold for buzzing
        """
        self.workflow = workflow
        self.output_variables = list(workflow.outputs.keys())

        # Validate input variables
        if self.external_input_variable not in workflow.inputs:
            raise ValueError(f"External input variable {self.external_input_variable} not found in workflow inputs")

        # Validate output variables
        for out_var in self.output_variables:
            if out_var not in workflow.outputs:
                raise ValueError(f"Output variable {out_var} not found in workflow outputs")

    def _single_run(self, question_run: str | dict, run_idx: int) -> TossupResult:
        """
        Process a single question run (supports both text-only and multimodal).
        
        Args:
            question_run: Either a string (text-only) or dict with text/images (multimodal)
            run_idx: The position of the question run (1-indexed)

        Returns:
            A TossupResult containing the answer, confidence, logprob, buzz, question fragment, position, step contents, response time, and step outputs
        """
        # Normalize input: convert string to dict format for consistency
        if isinstance(question_run, str):
            question_input = {"text": question_run, "images": [], "is_multimodal": False}
        else:
            question_input = question_run
        
        # Build workflow input with text (required)
        workflow_input = {
            self.external_input_variable: question_input["text"],
            # Keep this optional in practice: default to empty so workflows that
            # declare question_images do not fail on text-only runs.
            "question_images": [],
        }
        should_send_images = any(
            field.variable == "question_images"
            for step in self.workflow.steps.values()
            for field in step.input_fields
        )
        
        # Add multimodal data if present
        is_multimodal = question_input.get("is_multimodal", False)
        images = question_input.get("images", [])
        
        # Log multimodal status for debugging
        logger.info(f"[Multimodal Debug] Run {run_idx}: is_multimodal={is_multimodal}, images={images}, has_multimodal_tokens={'multimodal_tokens' in question_input}")
        
        # Resolve image paths if they're relative
        if is_multimodal and should_send_images:
            # Try to extract images from multimodal_tokens if images list is empty
            if not images and "multimodal_tokens" in question_input and question_input["multimodal_tokens"]:
                try:
                    from src.multimodal_utils import resolve_image_path_from_token
                except ImportError:
                    from multimodal_utils import resolve_image_path_from_token  # type: ignore
                base_dir = question_input.get("_base_dir", "")
                question_id = question_input.get("qid", "")
                image_paths = []
                for token in question_input["multimodal_tokens"]:
                    if isinstance(token, dict) and token.get("type") == "image":
                        resolved = resolve_image_path_from_token(token, base_dir, question_id)
                        if resolved:
                            image_paths.append(resolved)
                        elif token.get("path"):
                            image_paths.append(token["path"])
                if image_paths:
                    images = image_paths
                    logger.info(f"[Multimodal Debug] Extracted {len(image_paths)} images from multimodal_tokens: {image_paths}")
            
            # Pass through images (paths, URLs, or embedded PIL from Hub dataset)
            if images:
                import os
                from pathlib import Path
                resolved_images = []
                for img in images:
                    # Embedded image (e.g. PIL from Hub dataset) — pass through as-is
                    if not isinstance(img, str):
                        resolved_images.append(img)
                        continue
                    img_path = img
                    # If already absolute URL or path, use as-is
                    if os.path.isabs(img_path) or img_path.startswith(("http://", "https://")):
                        resolved_images.append(img_path)
                    else:
                        # Try to resolve relative path
                        possible_paths = [
                            img_path,
                            f"../model-simulation/multimodal_test/{img_path}",
                            f"model-simulation/multimodal_test/{img_path}",
                            os.path.join(os.getcwd(), "model-simulation", "multimodal_test", img_path),
                        ]
                        found = False
                        for path in possible_paths:
                            abs_path = os.path.abspath(path)
                            if os.path.exists(abs_path):
                                resolved_images.append(abs_path)
                                logger.info(f"[Multimodal Debug] Resolved image path: {img_path} -> {abs_path}")
                                found = True
                                break
                        if not found:
                            resolved_images.append(img_path)
                            logger.warning(f"[Multimodal Debug] Could not resolve image path: {img_path}, using as-is")
                
                if resolved_images:
                    workflow_input["question_images"] = resolved_images
                    logger.info(
                        f"[Multimodal Debug] Added {len(resolved_images)} images to workflow_input.question_images"
                    )
                else:
                    logger.warning(f"[Multimodal Debug] No valid image paths after resolution")
            elif is_multimodal:
                logger.warning(f"[Multimodal Debug] is_multimodal=True but no images found")
        elif is_multimodal and not should_send_images:
            logger.info(
                "[Multimodal Debug] question_images is not used by any step; running text-only for this run"
            )
        else:
            logger.info(f"[Multimodal Debug] Not multimodal or no images - text-only question")
        
        answer_var_step = self.workflow.outputs["answer"].split(".")[0]
        workflow_output, response_time = _get_workflow_response(
            self.workflow, workflow_input, logprob_step=answer_var_step
        )
        final_outputs = workflow_output["final_outputs"]
        buzz = self.workflow.buzzer.run(final_outputs["confidence"], logprob=workflow_output["logprob"])
        
        # Use text for question_fragment display
        question_fragment = question_input["text"]
        
        result: TossupResult = {
            "run_idx": run_idx,
            "guess": final_outputs["answer"],
            "confidence": final_outputs["confidence"],
            "logprob": workflow_output["logprob"],
            "buzz": buzz,
            "question_fragment": question_fragment,
            "step_contents": workflow_output["step_contents"],
            "step_outputs": workflow_output["intermediate_outputs"],  # Include intermediate step outputs
            "response_time": response_time,
        }
        return result

    def run(self, question_runs: list[str] | list[dict], early_stop: bool = True) -> Iterable[TossupResult]:
        """
        Process a tossup question and decide when to buzz based on confidence.
        Supports both text-only (list[str]) and multimodal (list[dict]) question runs.

        Args:
            question_runs: Progressive reveals of the question (text strings or multimodal dicts)
            early_stop: Whether to stop after the first buzz

        Yields:
            Dict containing:
                - answer: The model's answer
                - confidence: Confidence score
                - buzz: Whether to buzz
                - question_fragment: Current question text
                - position: Current position in question
                - step_contents: String content outputs of each step
                - response_time: Time taken for response
                - step_outputs: Outputs from each step
        """
        for i, question_run in enumerate(question_runs):
            # Execute the complete workflow
            result = self._single_run(question_run, i + 1)

            yield result

            # If we've reached the confidence threshold, buzz and stop
            if early_stop and result["buzz"]:
                if i + 1 < len(question_runs):
                    yield self._single_run(question_runs[-1], len(question_runs))
                return


class QuizBowlBonusAgent:
    """Agent for handling bonus questions with multiple steps in the workflow."""

    external_input_variables = ["leadin", "part"]
    output_variables = ["answer", "confidence", "explanation"]

    def __init__(self, workflow: Workflow):
        """Initialize the multi-step bonus agent.

        Args:
            workflow: The workflow containing multiple steps
        """
        self.workflow = workflow
        self.output_variables = list(workflow.outputs.keys())

        # Validate input variables
        for input_var in self.external_input_variables:
            if input_var not in workflow.inputs:
                raise ValueError(f"External input variable {input_var} not found in workflow inputs")

        # Validate output variables
        for out_var in self.output_variables:
            if out_var not in workflow.outputs:
                raise ValueError(f"Output variable {out_var} not found in workflow outputs")

    def run(self, leadin: str | dict, part: str | dict) -> BonusResult:
        """Process a bonus part with the given leadin and part.
        Supports both text-only and multimodal (image) bonuses.

        Args:
            leadin: The leadin text or dict with text, images, is_multimodal (multimodal bonus)
            part: The part text or dict with text, images, is_multimodal (multimodal bonus)

        Returns:
            Dict containing:
                - answer: The model's answer
                - confidence: Confidence score
                - explanation: Explanation for the answer
                - step_contents: String content outputs of each step
                - response_time: Time taken for response
                - step_outputs: Outputs from each step
        """
        # Normalize to text and collect images (leadin first, then part)
        leadin_text = leadin if isinstance(leadin, str) else leadin.get("text", "")
        part_text = part if isinstance(part, str) else part.get("text", "")

        workflow_input = {
            "leadin": leadin_text,
            "part": part_text,
            # Optional image channels: defaults avoid missing-input errors
            # when workflows include these vars but a run has no images.
            "leadin_images": [],
            "part_images": [],
        }

        # Multimodal: combine leadin images + part images (order matches prompt)
        leadin_images = [] if isinstance(leadin, str) else leadin.get("images") or []
        part_images = [] if isinstance(part, str) else part.get("images") or []

        if isinstance(leadin, dict) and not leadin_images and leadin.get("multimodal_tokens"):
            try:
                from src.multimodal_utils import resolve_image_path_from_token
            except ImportError:
                from multimodal_utils import resolve_image_path_from_token  # type: ignore
            base_dir = leadin.get("_base_dir", "")
            qid = leadin.get("qid", "")
            for token in leadin["multimodal_tokens"]:
                if isinstance(token, dict) and token.get("type") == "image":
                    resolved = resolve_image_path_from_token(token, base_dir, qid)
                    if resolved:
                        leadin_images.append(resolved)
                    elif token.get("path"):
                        leadin_images.append(token["path"])

        if isinstance(part, dict) and not part_images and part.get("multimodal_tokens"):
            try:
                from src.multimodal_utils import resolve_image_path_from_token
            except ImportError:
                from multimodal_utils import resolve_image_path_from_token  # type: ignore
            base_dir = part.get("_base_dir", "")
            qid = part.get("qid", "")
            for token in part["multimodal_tokens"]:
                if isinstance(token, dict) and token.get("type") == "image":
                    resolved = resolve_image_path_from_token(token, base_dir, qid)
                    if resolved:
                        part_images.append(resolved)
                    elif token.get("path"):
                        part_images.append(token["path"])

        # Resolve paths for relative/local images (same as tossup agent),
        # but keep leadin/part image channels separate.
        def _resolve_images(images: list):
            import os

            resolved_images = []
            for img in images:
                if not isinstance(img, str):
                    resolved_images.append(img)
                    continue
                img_path = img
                if os.path.isabs(img_path) or img_path.startswith(("http://", "https://")):
                    resolved_images.append(img_path)
                else:
                    possible_paths = [
                        img_path,
                        f"../model-simulation/multimodal_test/{img_path}",
                        f"model-simulation/multimodal_test/{img_path}",
                        os.path.join(os.getcwd(), "model-simulation", "multimodal_test", img_path),
                    ]
                    found = False
                    for path in possible_paths:
                        abs_path = os.path.abspath(path)
                        if os.path.exists(abs_path):
                            resolved_images.append(abs_path)
                            found = True
                            break
                    if not found:
                        resolved_images.append(img_path)
            return resolved_images

        workflow_input["leadin_images"] = _resolve_images(list(leadin_images))
        workflow_input["part_images"] = _resolve_images(list(part_images))

        workflow_output, response_time = _get_workflow_response(
            self.workflow,
            workflow_input,
        )
        final_outputs = workflow_output["final_outputs"]
        return {
            "guess": final_outputs["answer"],
            "confidence": final_outputs["confidence"],
            "logprob": workflow_output["logprob"],
            "explanation": final_outputs["explanation"],
            "step_contents": workflow_output["step_contents"],
            "response_time": response_time,
            "step_outputs": workflow_output["intermediate_outputs"],
        }


# Example usage
if __name__ == "__main__":
    # Load the Quizbowl dataset
    from datasets import load_dataset

    from shared.workflows.factory import create_quizbowl_bonus_workflow, create_quizbowl_tossup_workflow

    ds_name = "qanta-challenge/leaderboard_co_set"
    ds = load_dataset(ds_name, split="train")

    # Create the agents with multi-step workflows
    tossup_workflow = create_quizbowl_tossup_workflow()
    tossup_agent = QuizBowlTossupAgent(workflow=tossup_workflow, buzz_threshold=0.9)

    bonus_workflow = create_quizbowl_bonus_workflow()
    bonus_agent = QuizBowlBonusAgent(workflow=bonus_workflow)

    # Example for tossup mode
    print("\n=== TOSSUP MODE EXAMPLE ===")
    sample_question = ds[30]
    print(sample_question["question_runs"][-1])
    print(sample_question["gold_label"])
    print()
    question_runs = sample_question["question_runs"]

    results = tossup_agent.run(question_runs, early_stop=True)
    for result in results:
        print(result["step_contents"])
        print(f"Guess at position {result['position']}: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print("Step outputs:", result["step_outputs"])
        if result["buzz"]:
            print("Buzzed!\n")

    # Example for bonus mode
    print("\n=== BONUS MODE EXAMPLE ===")
    sample_bonus = ds[31]  # Assuming this is a bonus question
    leadin = sample_bonus["leadin"]
    parts = sample_bonus["parts"]

    print(f"Leadin: {leadin}")
    for i, part in enumerate(parts):
        print(f"\nPart {i + 1}: {part['part']}")
        result = bonus_agent.run(leadin, part["part"])
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Explanation: {result['explanation']}")
        print(f"Response time: {result['response_time']:.2f}s")
        print("Step outputs:", result["step_outputs"])
