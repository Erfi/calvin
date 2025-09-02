import json
import logging
import multiprocessing
import os
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import Callback, LightningModule, Trainer
from termcolor import colored

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import join_vis_lang, temp_seed
from calvin_agent.rollout.rollout_video import RolloutVideo

log_print = logging.getLogger(__name__)


def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    log_print.info(*args, **kwargs)


def divide_across_ranks(elements, world_size, rank):
    """
    Divide a number across subprocesses in multiprocessing.
    Example: distribute 4 elements in a world of size 3
    rank 0->2, rank 1->1, rank 2->1
    """
    assert rank < world_size
    rest = lambda n, w, i: 1 if n % w > i else 0
    return elements // world_size + rest(elements, world_size, rank)


def sequences_for_rank(num_sequences):
    """
    When using ddp, determine how many sequences every process should evaluate.
    """
    rank = dist.get_rank()
    ws = dist.get_world_size()
    num_seq_per_gpu = divide_across_ranks(num_sequences, ws, rank)
    num_workers = multiprocessing.cpu_count() // ws
    return [
        seq.tolist()
        for seq in np.array_split(get_sequences(num_sequences, num_workers=num_workers), ws)[rank][:num_seq_per_gpu]
    ]


def gather_results(local_results):
    """
    Collect eval results from all processes.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_results
    results = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(results, local_results)
    return list(chain(*results))


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


class RolloutLongHorizonVisualGoal(Callback):
    """
    A class for performing rollouts during validation step.
    This class does not follow the language-based goal sequencing,
    instead it uses visual frames as goals. Hence it required availability of a
    validation dataset and a start_end_tasks.json file like the following:
    {
    "0": {
        "58": [
            "turn_on_led"
        ],
        "160": [
            "open_drawer",
            "turn_on_led"
        ],
        "265": [
            "open_drawer",
            "push_into_drawer",
            "turn_on_led"
        ],
        "361": [
            "lift_pink_block_slider",
            "open_drawer",
            "push_into_drawer",
            "turn_on_led"
        ],
        "489": [
            "lift_blue_block_slider",
            "open_drawer",
            "push_into_drawer",
            "turn_on_led"
        ]
    },
    ...
    }
    This is information about the frame idx at the start of an episode (e.g. 0),
    and what tasks are already completed at each of the end frame idx (e.g. 58, 160, 265, 361, 489).
    """

    def __init__(
        self,
        env_cfg,
        skip_epochs,
        rollout_freq,
        num_videos,
        num_sequences,
        ep_len,
        tasks,
        log_video_to_file,
        save_dir,
        empty_cache,
        debug,
        start_end_tasks_file,
        num_tasks_per_rollout,
    ):
        self.env_cfg = env_cfg
        self.skip_epochs = skip_epochs
        self.rollout_freq = rollout_freq
        self.num_videos = num_videos
        self.num_sequences = num_sequences
        self.ep_len = ep_len
        self.task_checker = hydra.utils.instantiate(tasks)
        self.log_video_to_file = log_video_to_file
        self.save_dir = save_dir
        self.empty_cache = empty_cache
        self.debug = debug
        self.num_tasks_per_rollout = num_tasks_per_rollout
        self.start_end_tasks_file = start_end_tasks_file
        self.env = None
        self.eval_sequences = None
        self.start_end_tasks = None
        self.step_to_file = None

    def load_start_end_tasks(self, start_end_tasks_file):
        start_end_tasks = Path(start_end_tasks_file).expanduser()
        assert start_end_tasks.is_file(), f"{str(start_end_tasks)} not found"
        with open(start_end_tasks) as f:
            tasks = json.load(f)
        return tasks

    def get_offline_sequences(self, start_end_tasks, num_sequences):
        """
        creates a dictionary of
        {
        seq1_initial_frame_idx: [("completed_task1_name", completed_task1_frame_idx), ("completed_task2_name", completed_task2_frame_idx), ...],
        seq2_initial_frame_idx: [...],
        ...
        }
        we are looking for sequences that are completed one after the other.
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
                if len(previously_completed) == self.num_tasks_per_rollout:
                    break
            else:
                del sequences[start_idx]  # if we did not break, then this is not a valid sequence

        assert len(sequences) > 0, "No valid sequences found in the start_end_tasks.json."
        if len(sequences) > num_sequences:
            with temp_seed(42):
                keys = np.random.choice(list(sequences.keys()), size=num_sequences, replace=False)
                return {int(k): sequences[k] for k in keys}
        else:
            # if we have less sequences than requested, return all of them
            log_rank_0(f"Only {len(sequences)} sequences found, returning all of them.")
            return sequences

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.env is None:
            self.device = pl_module.device
            dataset = trainer.val_dataloaders["vis"].dataset  # type: ignore

            from calvin_agent.rollout.rollout import Rollout

            for callback in trainer.callbacks:
                if isinstance(callback, Rollout) and callback.env is not None:
                    self.env = callback.env
                    break
            else:
                self.env = hydra.utils.instantiate(self.env_cfg, dataset, pl_module.device)

        if self.num_videos > 0:
            if dist.is_available() and dist.is_initialized():
                self.num_videos = divide_across_ranks(self.num_videos, dist.get_world_size(), dist.get_rank())
            self.rollout_video = RolloutVideo(
                logger=pl_module.logger,
                empty_cache=self.empty_cache,
                log_to_file=self.log_video_to_file,
                save_dir=self.save_dir,
            )

        if self.step_to_file is None:
            data_dir = trainer.val_dataloaders["vis"].dataset.abs_datasets_dir
            self.step_to_file = self.get_step_to_file(data_dir)

        if self.start_end_tasks is None:
            self.start_end_tasks = self.load_start_end_tasks(self.start_end_tasks_file)

        self.eval_sequences = self.get_offline_sequences(
            start_end_tasks=self.start_end_tasks, num_sequences=self.num_sequences
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args: Any) -> None:
        if pl_module.current_epoch == 0 and self.skip_epochs > 0:
            for i in range(1, self.num_tasks_per_rollout + 1):
                pl_module.log(
                    f"eval_lh/sr_chain_{i}", torch.tensor(0.0, device=pl_module.device), on_step=False, sync_dist=True
                )
            pl_module.log(
                "eval_lh/avg_seq_len", torch.tensor(0.0, device=pl_module.device), on_step=False, sync_dist=True
            )
        elif pl_module.current_epoch >= self.skip_epochs and pl_module.current_epoch % self.rollout_freq == 0:
            results = self.evaluate_policy(pl_module)
            if self.num_videos > 0:
                # log rollout videos
                self.rollout_video.log(pl_module.global_step)

            results = gather_results(results)
            count = Counter(results)  # type: ignore
            print()
            for i in range(1, self.num_tasks_per_rollout + 1):
                n_success = sum(count[j] for j in reversed(range(i, self.num_tasks_per_rollout + 1)))
                sr = n_success / len(results)
                pl_module.log(
                    f"eval_lh/sr_chain_{i}", torch.tensor(sr, device=pl_module.device), on_step=False, sync_dist=True
                )
                log_rank_0(
                    f"{i} / {self.num_tasks_per_rollout} subtasks: {n_success} / {len(results)} sequences, SR: {sr * 100:.1f}%"
                )
            avg_seq_len = np.mean(results)
            pl_module.log(
                "eval_lh/avg_seq_len",
                torch.tensor(avg_seq_len, device=pl_module.device, dtype=torch.float),
                on_step=False,
                sync_dist=True,
            )
            log_rank_0(f"Average successful sequence length: {avg_seq_len:.1f}")
            print()

    def evaluate_policy(self, model):
        """
        Evaluates the policy on the given sequence of tasks.
        """
        results = []
        for i, (initial_state_idx, eval_sequence) in enumerate(self.eval_sequences.items()):
            record = i < self.num_videos
            result = self.evaluate_sequence(model, initial_state_idx, eval_sequence, record, i)
            results.append(result)
            if record:
                self.rollout_video.write_to_tmp()
        return results

    def evaluate_sequence(self, model, initial_state_idx, eval_sequence, record, i):
        """
        Evaluates the model on a single sequence.
        """
        # get goal-observation for each subtask
        subtasks = []
        for task_name, state_idx in eval_sequence:
            state = self.get_state_info_from_step(state_idx)
            goal_obs, _ = self.env.reset(robot_obs=state["robot_obs"], scene_obs=state["scene_obs"])
            subtasks.append((task_name, goal_obs))
        # reset the environment to the initial state of the task sequence
        initial_state = self.get_state_info_from_step(initial_state_idx)
        self.env.reset(robot_obs=initial_state["robot_obs"], scene_obs=initial_state["scene_obs"])

        if self.debug:
            os.makedirs("debug", exist_ok=True)
            initial_obs = self.env.get_obs()
            fig, ax = plt.subplots(
                nrows=1, ncols=self.num_tasks_per_rollout + 1, figsize=(3 * self.num_tasks_per_rollout, 5)
            )
            img = initial_obs["rgb_obs"]["rgb_static"].detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            ax[0].imshow(img)
            ax[0].set_title("initial state")
            ax[0].axis("off")
            for j, (task_name, goal_obs) in enumerate(subtasks, start=1):
                img = goal_obs["rgb_obs"]["rgb_static"].detach().cpu().numpy().squeeze().transpose(1, 2, 0)
                ax[j].imshow(img)
                ax[j].set_title(f"{task_name}")
                ax[j].axis("off")

            fig.savefig(f"debug/rollout_frames_starting_{initial_state_idx}.png")
            plt.close(fig)
        if record:
            caption = " | ".join([task[0] for task in subtasks])
            self.rollout_video.new_video(tag=get_video_tag(i), caption=caption)
        success_counter = 0
        if self.debug:
            print()
            print()
            print(f"Evaluating sequence: {' -> '.join([task[0] for task in subtasks])}")
            print("Subtask: ", end="")
        for subtask in subtasks:
            if record:
                self.rollout_video.new_subtask()
            success = self.rollout(model, subtask, record)
            if record:
                self.rollout_video.draw_outcome(success)
            if success:
                success_counter += 1
            else:
                return success_counter
        return success_counter

    def rollout(self, model, subtask, record):
        """
        Performs a rollout for the given subtask of a sequence.
        subtask: ("<task_name>", <goal_obs>)
        """
        if self.debug:
            print(f"{subtask} ", end="")
        obs = self.env.get_obs()
        goal = subtask[1]
        task_name = subtask[0]
        model.reset()
        start_info = self.env.get_info()
        success = False
        for _ in range(self.ep_len):
            action = model.step(obs, goal)
            obs, _, _, _, current_info = self.env.step(action)
            if self.debug and os.environ.get("DISPLAY") is not None:
                img = self.env.render()
                join_vis_lang(img, task_name)
            if record:
                # update video
                self.rollout_video.update(obs["rgb_obs"]["rgb_static"])
            # check if current step solves a task
            current_task_info = self.task_checker.get_task_info_for_set(start_info, current_info, {task_name})
            if len(current_task_info) > 0:
                success = True
                break
        if self.debug:
            if success:
                print(colored("success", "green"), end=" ")
            else:
                print(colored("fail", "red"), end=" ")
        if record:
            self.rollout_video.add_language_instruction(task_name)
        return success

    def get_file_list(self, data_dir, extension=".npz"):
        """retrieve a list of files inside a folder using glob"""
        dir_path = Path(data_dir).expanduser()
        assert dir_path.is_dir(), f"{data_dir} is not a valid dir path"
        return list(dir_path.glob(f"*{extension}"))

    def get_step_to_file(self, data_dir):
        """Create mapping from step to file"""
        step_to_file = {}
        file_list = self.get_file_list(data_dir)
        for file in file_list:
            step = int(file.stem.split("_")[-1])
            step_to_file[step] = file
        return step_to_file

    def get_state_info_from_step(self, step: int):
        data = np.load(self.step_to_file[step], allow_pickle=True)
        return {
            "robot_obs": data["robot_obs"],
            "scene_obs": data["scene_obs"],
        }
