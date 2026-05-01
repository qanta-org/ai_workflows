import json
from concurrent import futures

from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from .metrics import evaluate_prediction, helpfulness_score
from .qb_agents import QuizBowlBonusAgent, QuizBowlTossupAgent

try:
    from src.multimodal_utils import get_text_with_placeholders, resolve_image_path_from_token
except ImportError:
    from multimodal_utils import get_text_with_placeholders, resolve_image_path_from_token  # type: ignore


def _parse_multimodal_tokens(example: dict):
    """Parse multimodal_tokens from example, return None if text-only."""
    if "multimodal_tokens" not in example or not example["multimodal_tokens"]:
        return None
    
    tokens = example["multimodal_tokens"]
    
    # Handle string format (from CSV/JSONL)
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except json.JSONDecodeError:
            return None
    
    # Validate token structure
    if not isinstance(tokens, list):
        return None
    
    for token in tokens:
        if not isinstance(token, dict):
            return None
        if "type" not in token or "position" not in token:
            return None
        if token["type"] == "text" and "content" not in token:
            return None
        if token["type"] == "image":
            if not token.get("path") and not token.get("hash_key") and not token.get("hashKey"):
                return None
        # delay: requires position only; duration is optional
        if token["type"] == "delay":
            continue
    
    return tokens


def get_question_runs(example: dict) -> list[dict]:
    """
    Get progressive question runs, supporting both text-only and multimodal.
    
    Returns:
        List of question run dicts, each containing:
        - text: str (text content up to this position)
        - images: list[str] (image paths/URLs up to this position)
        - multimodal_tokens: list[dict] (full token sequence up to this position) or None
        - is_multimodal: bool
    """
    multimodal_tokens = _parse_multimodal_tokens(example)
    
    if not multimodal_tokens:
        # Text-only: backward compatibility
        tokens = example["question"].split()
        question_runs = []
        for run_idx in example["run_indices"]:
            question_runs.append({
                "text": " ".join(tokens[: run_idx + 1]),
                "images": [],
                "multimodal_tokens": None,
                "is_multimodal": False
            })
        return question_runs
    
    # Multimodal: process tokens
    sorted_tokens = sorted(multimodal_tokens, key=lambda x: x["position"])
    question_runs = []
    
    # Log for debugging
    from loguru import logger
    logger.info(f"[Multimodal Debug] Creating question runs: {len(sorted_tokens)} tokens, run_indices={example.get('run_indices', [])}")
    
    base_dir = example.get("_base_dir", "")
    question_id = example.get("qid", "")
    embedded_images = example.get("images")  # From Hub: list of PIL/Image in token order
    for run_idx in example["run_indices"]:
        tokens_up_to_idx = sorted_tokens[: run_idx + 1]
        # Text with placeholders: <img:hash_key> and <delay>
        text = get_text_with_placeholders(tokens_up_to_idx)
        images = []
        if embedded_images and not base_dir:
            # Use embedded images (Hub dataset): i-th image token -> embedded_images[i]
            image_index = 0
            for token in tokens_up_to_idx:
                if token.get("type") == "image":
                    if image_index < len(embedded_images):
                        images.append(embedded_images[image_index])
                    image_index += 1
        else:
            for token in tokens_up_to_idx:
                if token["type"] == "image":
                    resolved = resolve_image_path_from_token(token, base_dir, question_id)
                    if resolved:
                        images.append(resolved)
                    elif token.get("path"):
                        images.append(token["path"])
        
        question_run = {
            "text": text,
            "images": images,
            "multimodal_tokens": tokens_up_to_idx,
            "is_multimodal": True,
            "_base_dir": base_dir,
            "qid": question_id,
        }
        
        logger.info(f"[Multimodal Debug] Question run at index {run_idx}: text_length={len(question_run['text'])}, images={len(images)}, image_paths={images}")
        question_runs.append(question_run)
    
    return question_runs


def run_and_evaluate_tossup(
    agent: QuizBowlTossupAgent,
    example: dict,
    return_extras: bool = False,
    early_stop: bool = True,
):
    results = []
    question_runs = get_question_runs(example)
    try:
        run_outputs = agent.run(question_runs, early_stop=early_stop)
        for run_output in run_outputs:
            if return_extras:
                run_out = run_output
            else:
                run_out = {k: run_output[k] for k in ["guess", "confidence", "buzz", "run_idx"]}
            run_out["correct"] = evaluate_prediction(run_out["guess"], example["clean_answers"])
            # This is 1-indexed token-position
            run_out["token_position"] = example["run_indices"][run_output["run_idx"] - 1] + 1
            results.append(run_out)
    except Exception as e:
        preview = example.get("question") or ""
        if question_runs:
            first = question_runs[0]
            if isinstance(first, dict):
                preview = first.get("text") or preview
            else:
                preview = str(first)
        logger.error(
            "Tossup run failed qid={} (run_indices={}): {}",
            example.get("qid"),
            example.get("run_indices"),
            e,
        )
        logger.error("Question text preview (first run): {}", preview[:1200] if preview else "(empty)")
        raise
    return {
        "qid": example["qid"],
        "run_outputs": results,
    }


def run_and_eval_tossup_dataset(
    agent: QuizBowlTossupAgent,
    dataset: Dataset,
    early_stop: bool = True,
    num_workers: int = 4,
    return_extras: bool = False,
    tqdm_provider: tqdm = tqdm,
):
    def process_example(example: dict):
        return run_and_evaluate_tossup(agent, example, return_extras, early_stop)

    outputs_map = {}
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_example = {executor.submit(process_example, example): example for example in dataset}

        # Use tqdm to show progress as futures complete
        for future in tqdm_provider(
            futures.as_completed(future_to_example), total=len(dataset), desc="Running Tossups"
        ):
            result = future.result()
            outputs_map[result["qid"]] = result

    tossup_outputs = [outputs_map[qid] for qid in dataset["qid"]]
    return tossup_outputs


def _get_bonus_leadin_and_part_for_part(example: dict, part_index: int) -> tuple[str | dict, str | dict]:
    """
    Get leadin and part inputs for one bonus part, supporting text-only and multimodal.
    Returns (leadin_input, part_input) where each is either a string (text) or a dict with
    text, images, is_multimodal, and optionally multimodal_tokens, _base_dir, qid.
    """
    qid = example.get("qid", "")
    base_dir = example.get("_base_dir", "")
    leadin_tokens = example.get("leadin_multimodal_tokens") or []
    parts = example["parts"]
    part = parts[part_index]
    part_tokens = part.get("multimodal_tokens") or []

    # Embedded images (Hub dataset): leadin_images, part_images, part_image_splits
    leadin_images = example.get("leadin_images")
    part_images_flat = example.get("part_images")
    part_image_splits = example.get("part_image_splits") or [0] * (len(parts) + 1)
    if not part_image_splits and part_images_flat is not None:
        part_image_splits = [0, len(part_images_flat)]

    is_leadin_multimodal = len(leadin_tokens) > 0
    is_part_multimodal = len(part_tokens) > 0
    is_multimodal = is_leadin_multimodal or is_part_multimodal

    if not is_multimodal:
        return example["leadin"], part["question"]

    # Build leadin input (text + images)
    leadin_text = get_text_with_placeholders(leadin_tokens) if leadin_tokens else example["leadin"]
    leadin_images_list = []
    if leadin_images is not None:
        leadin_images_list = list(leadin_images)
    else:
        for token in leadin_tokens:
            if token.get("type") == "image":
                resolved = resolve_image_path_from_token(token, base_dir, qid)
                if resolved:
                    leadin_images_list.append(resolved)
                elif token.get("path"):
                    leadin_images_list.append(token["path"])

    # Build part input (text + images)
    part_text = get_text_with_placeholders(part_tokens) if part_tokens else part["question"]
    part_images_list = []
    if part_images_flat is not None and part_image_splits:
        start = part_image_splits[part_index]
        end = part_image_splits[part_index + 1]
        part_images_list = list(part_images_flat[start:end])
    else:
        for token in part_tokens:
            if token.get("type") == "image":
                resolved = resolve_image_path_from_token(token, base_dir, qid)
                if resolved:
                    part_images_list.append(resolved)
                elif token.get("path"):
                    part_images_list.append(token["path"])

    leadin_input = {
        "text": leadin_text,
        "images": leadin_images_list,
        "is_multimodal": is_leadin_multimodal,
        "multimodal_tokens": leadin_tokens if is_leadin_multimodal else None,
        "_base_dir": base_dir,
        "qid": qid,
    }
    part_input = {
        "text": part_text,
        "images": part_images_list,
        "is_multimodal": is_part_multimodal,
        "multimodal_tokens": part_tokens if is_part_multimodal else None,
        "_base_dir": base_dir,
        "qid": qid,
    }
    return leadin_input, part_input


def run_and_evaluate_bonus(agent: QuizBowlBonusAgent, example: dict, return_extras: bool = False) -> dict:
    results = []
    for i, part in enumerate(example["parts"], start=1):
        leadin_input, part_input = _get_bonus_leadin_and_part_for_part(example, i - 1)
        try:
            result = agent.run(leadin_input, part_input)
            if return_extras:
                result = result
            else:
                result = {k: result[k] for k in ["guess", "confidence", "explanation"]}
        except Exception as e:
            logger.error(f"Error running {example['qid']} part {i}: {e}")
            result = {"guess": "ERROR", "confidence": 0.0, "explanation": "Error producing answer."}
        result["number"] = i
        result["correct"] = evaluate_prediction(result["guess"], part["clean_answers"])
        results.append(result)
    return {
        "qid": example["qid"],
        "part_outputs": results,
    }


def run_and_eval_bonus_dataset(
    agent: QuizBowlBonusAgent,
    dataset: Dataset,
    num_workers: int = 4,
    return_extras: bool = False,
    tqdm_provider: tqdm = tqdm,
) -> list[dict]:
    def process_example(example: dict) -> dict:
        return run_and_evaluate_bonus(agent, example, return_extras)

    outputs_map = {}
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_example = {executor.submit(process_example, example): example for example in dataset}

        # Use tqdm to show progress as futures complete
        for future in tqdm_provider(
            futures.as_completed(future_to_example), total=len(dataset), desc="Running Bonuses"
        ):
            result = future.result()
            outputs_map[result["qid"]] = result

    bonus_outputs = [outputs_map[qid] for qid in dataset["qid"]]
    return bonus_outputs


def inject_bonus_metrics_example(model_output: dict, example: dict, max_explanation_tokens: int = 30) -> dict:
    results = []
    for part, output in zip(example["parts"], model_output["part_outputs"]):
        question_text = f"Leadin: {example['leadin']}\n\nQuestion: {part['question']}"
        answer_refs = part["clean_answers"]
        explanation = output["explanation"]
        guess = output["guess"]
        confidence = output["confidence"]
        expl_tokens = explanation.split(" ")
        if len(expl_tokens) > max_explanation_tokens:
            explanation = " ".join(expl_tokens[:max_explanation_tokens]) + "...[TRUNCATED]"
        h_scores = helpfulness_score(question_text, answer_refs, guess, explanation)
        h_scores["helper_confidence"] = confidence
        results.append(h_scores)
    return {"scores": results}


def inject_bonus_metrics(system_outputs: list[dict], dataset: Dataset, num_workers: int = 4) -> list[dict]:
    outputs_map = {}
    for bonus_output in system_outputs:
        qid = bonus_output["qid"]
        outputs_map[qid] = bonus_output

    def process_example(example: dict) -> dict:
        qid = example["qid"]
        system_output = outputs_map[qid]
        return inject_bonus_metrics_example(system_output, example)

    scored_examples = dataset.map(process_example, num_proc=num_workers)
    for e, o in zip(scored_examples, system_outputs):
        o["scores"] = e["scores"]
    return system_outputs
