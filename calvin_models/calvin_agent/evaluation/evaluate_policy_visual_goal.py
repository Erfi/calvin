"""
Modified version of evaluate_policy.py.
This modification is because this version uses visual goals instead of language goals.

Methods presented here resemble those in rollout/rollout_long_horizon_visual_goal.py
"""

from typing import Dict
from collections import Counter
import logging
from pathlib import Path
import sys
import json

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.utils import get_log_dir, temp_seed
import numpy as np
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EP_LEN = 360


def load_start_end_tasks(start_end_tasks_file):
    start_end_tasks = Path(start_end_tasks_file).expanduser()
    assert start_end_tasks.is_file(), f"{str(start_end_tasks)} not found"
    with open(start_end_tasks) as f:
        tasks = json.load(f)
    return tasks


def get_file_list(data_dir, extension=".npz"):
    """retrieve a list of files inside a folder using glob"""
    dir_path = Path(data_dir).expanduser()
    assert dir_path.is_dir(), f"{data_dir} is not a valid dir path"
    return list(dir_path.glob(f"*{extension}"))


def get_step_to_file(data_dir):
    """Create mapping from step to file"""
    step_to_file = {}
    file_list = get_file_list(data_dir)
    for file in file_list:
        step = int(file.stem.split("_")[-1])
        step_to_file[step] = file
    return step_to_file


def get_state_info_from_step(step_to_file: Dict, step: int):
    data = np.load(step_to_file[step], allow_pickle=True)
    return {
        "robot_obs": data["robot_obs"],
        "scene_obs": data["scene_obs"],
    }


def get_offline_sequences(start_end_tasks, num_sequences, num_tasks_per_rollout) -> Dict:
    """
    Looking for sequences that are completed one after the other.

    Args:
        start_end_tasks: A dictionary mapping initial frame indices to a list of completed tasks and their frame indices.
        num_sequences: The number of sequences to return.
        num_tasks_per_rollout: The number of tasks to complete in each rollout sequence.

    Returns:
        A dictionary mapping initial frame indices to a list of completed (tasks names and their frame indices) e.g.:
        {
        seq1_initial_frame_idx: [("completed_task1_name", completed_task1_frame_idx), ("completed_task2_name", completed_task2_frame_idx), ...],
        seq2_initial_frame_idx: [...],
        ...
        }
    """
    sequences = {}
    for start_idx, end_tasks in start_end_tasks.items():
        start_idx = int(start_idx)
        sequences[start_idx] = []
        previously_completed = set()
        for end_idx, completed_tasks in end_tasks.items():
            end_idx = int(end_idx)
            newly_completed = set(completed_tasks) - previously_completed
            if len(newly_completed) == 1 and len(completed_tasks) > len(previously_completed):
                sequences[start_idx].append((newly_completed.pop(), end_idx))
            else:
                del sequences[start_idx]
                break
            previously_completed.update(completed_tasks)
            if len(previously_completed) == num_tasks_per_rollout:
                break
        else:
            del sequences[start_idx]  # if we did not break, then this is not a valid sequence

    assert (
        len(sequences) > 0
    ), f"No valid sequences found in the start_end_tasks.json with num_tasks_per_rollout={num_tasks_per_rollout}"
    if len(sequences) > num_sequences:
        with temp_seed(42):
            keys = np.random.choice(list(sequences.keys()), size=num_sequences, replace=False)
            return {int(k): sequences[k] for k in keys}
    else:
        # if we have less sequences than requested, return all of them
        logger.info(
            f"Only {len(sequences)} valid sequences found for num_tasks_per_rollout={num_tasks_per_rollout}, returning all of them."
        )
        return sequences


def evaluate_policy(
    model,
    env,
    task_checker,
    start_end_tasks_file,
    val_dataset_dir,
    num_sequences,
    num_tasks_per_sequence,
    eval_log_dir=None,
):
    """
    Run this function to evaluate a model on the CALVIN challenge using visual goals.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        val_dataset_dir: Path to the validation dataset directory containing frames refered to in eval_sequences.
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    # prepare evaluation sequences from offline dataset (validation)
    start_end_tasks = load_start_end_tasks(start_end_tasks_file)
    eval_sequences = get_offline_sequences(
        start_end_tasks=start_end_tasks,
        num_sequences=num_sequences,
        num_tasks_per_rollout=num_tasks_per_sequence,
    )

    eval_log_dir = get_log_dir(eval_log_dir)
    step_to_file = get_step_to_file(val_dataset_dir)
    results = []

    for initial_state_idx, eval_sequence in tqdm(eval_sequences.items(), position=0, leave=True):
        result = evaluate_sequence(model, env, task_checker, initial_state_idx, eval_sequence, step_to_file)
        results.append(result)
        logger.info(f"Completed: {result}/{len(eval_sequence)}\n")

    count = Counter(results)  # type: ignore
    for i in range(1, num_tasks_per_sequence + 1):
        n_success = sum(count[j] for j in reversed(range(i, num_tasks_per_sequence + 1)))
        sr = n_success / len(results)
        logger.info(
            f"{i} / {num_tasks_per_sequence} subtasks: {n_success} / {len(results)} sequences, SR: {sr * 100:.1f}%"
        )
    avg_seq_len = np.mean(results)
    logger.info(f"Average successful sequence length: {avg_seq_len:.1f}")

    return results


def evaluate_sequence(model, env, task_checker, initial_state_idx, eval_sequence, step_to_file):
    """
    Evaluates the model on a single sequence with visual goals
    """
    # get goal-observation for each subtask
    subtasks = []
    for task_name, state_idx in eval_sequence:
        state = get_state_info_from_step(step_to_file=step_to_file, step=state_idx)
        goal_obs, _ = env.reset(robot_obs=state["robot_obs"], scene_obs=state["scene_obs"])
        subtasks.append((task_name, goal_obs))
    # reset the environment to the initial state of the task sequence
    initial_state = get_state_info_from_step(step_to_file=step_to_file, step=initial_state_idx)
    env.reset(robot_obs=initial_state["robot_obs"], scene_obs=initial_state["scene_obs"])
    logger.info(f"Evaluating sequence: {' -> '.join([task[0] for task in subtasks])}")
    success_counter = 0
    for subtask in subtasks:
        success = rollout(model, env, task_checker, subtask)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(model, env, task_checker, subtask):
    """
    Performs a rollout for the given subtask of a sequence.
    subtask: ("<task_name>", <goal_obs>)
    """
    obs = env.get_obs()
    goal = subtask[1]
    task_name = subtask[0]
    model.reset()
    start_info = env.get_info()
    success = False
    for _ in range(EP_LEN):
        action = model.step(obs, goal)
        obs, _, _, _, current_info = env.step(action)
        # check if current step solves a task
        current_task_info = task_checker.get_task_info_for_set(start_info, current_info, {task_name})
        if len(current_task_info) > 0:
            success = True
            break
    return success
