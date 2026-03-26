# %%
from .structs import (
    Buzzer,
    BuzzerMethod,
    CallType,
    InputField,
    ModelStep,
    OutputField,
    TossupWorkflow,
    Workflow,
)

INITIAL_SYS_PROMPT = """You are a  helpful performant question answering bot.
Given a question clue, output your most likely guess in a couple words with a calibrated confidence for the guess.
"""


def create_empty_bonus_workflow():
    return Workflow(
        inputs=["leadin", "part", "leadin_images", "part_images"],
        outputs={"answer": None, "confidence": None, "explanation": None},
        steps={},
    )


def create_empty_tossup_workflow():
    return TossupWorkflow(
        inputs=["question_text", "question_images"],
        outputs={"answer": None, "confidence": None},
        steps={},
    )


def create_first_step_input_fields() -> list[InputField]:
    return [
        InputField(
            name="question",
            description="The question text progressively revealed so far.",
            variable="question_text",
        )
    ]


def create_empty_input_field() -> list[InputField]:
    return [InputField(name="", description="", variable="question_text")]


def create_quizbowl_simple_step_initial_setup():
    return ModelStep(
        id="simple_step",
        name="Quizbowl Simple Step",
        model="",
        provider="",
        temperature=0.7,
        call_type="llm",
        system_prompt=INITIAL_SYS_PROMPT,
        input_fields=[
            InputField(name="question", description="The question to answer", variable="question"),
        ],
        output_fields=[
            OutputField(name="answer", description="The most likely answer", type="str"),
            OutputField(name="confidence", description="The confidence of the answer", type="float"),
        ],
    )


def create_new_llm_step(step_id: str, name: str) -> ModelStep:
    return ModelStep(
        id=step_id,
        name=name,
        model="gpt-4o",
        provider="OpenAI",
        call_type="llm",
        temperature=0.7,
        system_prompt="",
        input_fields=create_empty_input_field(),
        output_fields=[OutputField(name="", description="")],
    )


def create_first_llm_step() -> ModelStep:
    return ModelStep(
        id="A",
        name="",
        model="gpt-4o",
        provider="OpenAI",
        call_type="llm",
        temperature=0.7,
        system_prompt="",
        input_fields=[create_first_step_input_fields()],
        output_fields=[OutputField(name="", description="")],
    )


def create_simple_qb_tossup_workflow():
    return TossupWorkflow(
        inputs=["question_text", "question_images"],
        outputs={"answer": "A.answer", "confidence": "A.confidence"},
        steps={
            "A": ModelStep(
                id="A",
                name="Tossup Agent",
                model="gpt-4o-mini",
                provider="OpenAI",
                call_type="llm",
                temperature=0.3,
                system_prompt="""You are a professional quizbowl player answering tossup questions.
Given a progressively revealed question (which may include both text and images), provide your best guess at the answer and your confidence level.

Your task:
1. Analyze the clues provided in the question text AND any images that are included
2. If images are present, carefully examine them as they contain important visual information needed to answer the question
3. Determine the most likely answer based on all available information (text and images)
4. Assess your confidence in your answer on a scale from 0.0 (complete guess) to 1.0 (absolute certainty)

Keep your answer direct and concise, limited to a couple of words.
Your confidence should reflect how certain you are based on the clues revealed so far (including visual clues from images if present).""",
                input_fields=[InputField(name="question", description="The question text", variable="question_text")],
                output_fields=[
                    OutputField(
                        name="answer",
                        description="The best guess at the answer to the question",
                        type="str",
                    ),
                    OutputField(
                        name="confidence",
                        description="The confidence in the answer, ranging from 0 to 1 in increments of 0.05.",
                        type="float",
                    ),
                ],
            )
        },
        buzzer=Buzzer(
            confidence_threshold=0.75,
            prob_threshold=None,
            method=BuzzerMethod.AND,
        ),
    )


BONUS_SYS_PROMPT = """You are a quizbowl player answering bonus questions. For each part:
1. Read the leadin and part carefully
2. Provide a concise answer
3. Rate your confidence (0-1)
4. Explain your reasoning

Format your response as:
ANSWER: <your answer>
CONFIDENCE: <0-1>
EXPLANATION: <your reasoning>"""


def create_simple_qb_bonus_workflow() -> Workflow:
    """Create a simple model step for bonus questions."""
    return Workflow(
        inputs=["leadin", "part", "leadin_images", "part_images"],
        outputs={"answer": "A.answer", "confidence": "A.confidence", "explanation": "A.explanation"},
        steps={
            "A": ModelStep(
                id="A",
                name="Bonus Agent",
                model="gpt-4o-mini",
                provider="OpenAI",
                temperature=0.3,
                call_type=CallType.LLM,
                system_prompt=BONUS_SYS_PROMPT,
                input_fields=[
                    InputField(
                        name="question_leadin",
                        description="The leadin text for the bonus question",
                        variable="leadin",
                    ),
                    InputField(
                        name="question_part",
                        description="The specific part text to answer",
                        variable="part",
                    ),
                ],
                output_fields=[
                    OutputField(name="answer", description="The predicted answer", type="str"),
                    OutputField(name="confidence", description="Confidence in the answer (0-1)", type="float"),
                    OutputField(name="explanation", description="Short explanation for the answer", type="str"),
                ],
            )
        },
    )
