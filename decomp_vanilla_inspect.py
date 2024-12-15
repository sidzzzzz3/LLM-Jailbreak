from inspect_ai import Task, task, Epochs, eval
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.scorer import choice
from solvers_inspect import decomposition_solver_mcq
from typing import Any

# Default epochs to run eval for
DEFAULT_EPOCHS = 4

@task
def gpqa_decomposition_task(
    cot: bool = True,
    epochs: int = DEFAULT_EPOCHS,
    max_iterations: int = 3,
    max_decompositions: int = 3,
):
    return Task(
        dataset=csv_dataset(
            csv_file="https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
            sample_fields=record_to_sample,
        ),
        plan=[
            decomposition_solver_mcq(
                max_iterations=max_iterations,
                max_decompositions=max_decompositions,
            )
        ],
        scorer=choice(),  # Use the standard multiple_choice scorer
        epochs=Epochs(epochs, "max"),
    )

def record_to_sample(record: dict[str, Any]) -> Sample:
    # Map records to inspect_ai Sample
    choices = [
        str(record["Correct Answer"]),
        str(record["Incorrect Answer 1"]),
        str(record["Incorrect Answer 2"]),
        str(record["Incorrect Answer 3"]),
    ]
    # The correct answer is always "A" before shuffling
    target = "A"

    # Optionally shuffle the choices to avoid position bias
    import random
    random.seed(42)
    combined = list(zip(choices, ["A", "B", "C", "D"]))
    random.shuffle(combined)
    shuffled_choices, labels = zip(*combined)
    target_index = labels.index("A")
    target_label = chr(65 + target_index)  # Convert index to letter

    return Sample(
        input=record["Question"],
        choices=shuffled_choices,
        target=target_label,
        id=record["Record ID"],
    )

if __name__ == "__main__":
    # Run the task decomposition on GPQA dataset
    task = gpqa_decomposition_task(
        cot=True,
        epochs=1,
        max_iterations=3,
        max_decompositions=3,
    )
    eval(
        task,
        epochs=Epochs(1, "max"),
        max_connections=10000,
        log_dir="test",
        model="openai/gpt-3.5-turbo",
    )