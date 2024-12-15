from inspect_ai import Task, task, Epochs, Task, eval
from inspect_ai.solver._task_state import Sample
from inspect_ai.scorer import at_least
from inspect_ai.solver import solver, Generate, TaskState, chain_of_thought
from data_inspect import  harm_bench_dataset
from solvers_inspect import  decomposition_attack_solver, decomposition_attack_solver_sequential
from scorers_inspect import final_scorer

import numpy as np

# randomly sample from the dataset
np.random.seed(42)

jb_reducer = at_least(1, 10)
harm_bench_dataset = [
    Sample(input=item.input, target=item.target, metadata=item.metadata, sandbox=item.sandbox if hasattr(item, 'sandbox') else None)
    for item in harm_bench_dataset
][:30]

@task
def task_decomposition_task(
    dataset: list = harm_bench_dataset,
    epochs: int = 10,
    max_iterations: int = 5,
    max_decompositions: int = 1,
    max_decomposition_rounds: int = 1,
    decomp_cot: bool = True,
):
    """
    Task Decomposition task within Inspect with optional Chain of Thought
    """
    return Task(
        dataset=dataset,
        plan=[decomposition_attack_solver_sequential(
            max_iterations=max_iterations, 
            max_decompositions=max_decompositions, 
            max_decomposition_rounds=max_decomposition_rounds,
            use_pair_immediately = False,
            decomp_cot=decomp_cot,
            use_strongreject_scorer = True,
        )],
        scorer=final_scorer(),
        epochs=Epochs(epochs, jb_reducer)
    )
    

if __name__ == "__main__":
    for dataset in harm_bench_dataset:
        # You can toggle `use_chain_of_thought` to True if you want to use CoT
        task = task_decomposition_task(dataset=[dataset], max_iterations=3, max_decompositions=3, max_decomposition_rounds=2)
        eval(task, epochs=Epochs(2, "max"), max_connections=10000, log_dir="PAIR-decomp-sequential-SR-claude_3_3_2", model="together/mistralai/Mixtral-8x22B-Instruct-v0.1")
