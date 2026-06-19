"""Launch ZetaBench SAC training on Amazon SageMaker Training Jobs.

This is the *cluster fan-out* entry point. SB3's SAC is off-policy and
single-process, so a multi-GPU/multi-node job does **not** speed up one training
run. The honest, effective use of a fleet is many independent single-GPU jobs:

* ``seeds`` — submit one job per seed (reproducibility / variance across seeds).
* ``hpo``   — a Bayesian ``HyperparameterTuner`` sweep over SAC hyperparameters,
  ``max_parallel_jobs`` running concurrently.

Each job runs the ``sagemaker`` Docker stage (built with
``docker build --target sagemaker``) pushed to ECR. The container's
``docker/sm-entrypoint.sh`` maps SageMaker conventions onto
``experiments/train.py``; the ``hyperparameters`` passed here become Hydra
overrides verbatim (e.g. ``compute=large_gpu`` selects the GPU compute profile).

This module requires the ``cloud`` extra (``pip install -e '.[cloud]'``); it is
run from a SageMaker Studio space / laptop, never inside the training image.

Examples
--------
    # Three seeds, one A10G GPU each, concurrent.
    python experiments/sagemaker_launch.py seeds \
        --image-uri 123456789012.dkr.ecr.us-east-1.amazonaws.com/zeta-bench:sm \
        --role arn:aws:iam::123456789012:role/SageMakerExecutionRole \
        --s3-output s3://my-bucket/zetabench/ \
        --seeds 0 1 2 --total-steps 2000000

    # HPO sweep: 12 trials, 4 at a time.
    python experiments/sagemaker_launch.py hpo \
        --image-uri ...:sm --role ... --s3-output s3://my-bucket/zetabench/ \
        --max-jobs 12 --max-parallel-jobs 4 --total-steps 1000000
"""
from __future__ import annotations

import argparse
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sagemaker.estimator import Estimator


def _wandb_environment() -> dict[str, str]:
    """Forward WandB credentials into the job environment when available.

    The key is read from the *launching* environment (e.g. injected from AWS
    Secrets Manager in Studio) and never committed. Absent a key, runs fall back
    to offline mode via the resolver in :mod:`utils.wandb_setup`.
    """
    env: dict[str, str] = {}
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        env["WANDB_API_KEY"] = api_key
    return env


def _build_estimator(args: argparse.Namespace, hyperparameters: dict[str, object]) -> "Estimator":
    """Construct a SageMaker ``Estimator`` for the ZetaBench training image.

    A single job uses one GPU instance (``instance_count=1``); parallelism comes
    from launching many such jobs, not from distributing one run.
    """
    from sagemaker.estimator import Estimator

    checkpoint_s3_uri = args.checkpoint_s3 or f"{args.s3_output.rstrip('/')}/checkpoints/"

    return Estimator(
        image_uri=args.image_uri,
        role=args.role,
        instance_count=1,
        instance_type=args.instance_type,
        output_path=args.s3_output,
        # Continuously sync /opt/ml/checkpoints to S3 so spot interruptions resume.
        checkpoint_s3_uri=checkpoint_s3_uri,
        use_spot_instances=args.use_spot,
        max_run=args.max_run,
        max_wait=args.max_run + 3600 if args.use_spot else None,
        environment=_wandb_environment(),
        hyperparameters=hyperparameters,
        base_job_name="zetabench-sac",
    )


def _base_hyperparameters(args: argparse.Namespace) -> dict[str, object]:
    """Hydra overrides shared by every launched job."""
    return {
        "agent": "sac",
        "compute": args.compute,
        "total_steps": args.total_steps,
    }


def run_seeds(args: argparse.Namespace) -> None:
    """Submit one independent training job per seed (non-blocking)."""
    for seed in args.seeds:
        hyperparameters = {**_base_hyperparameters(args), "seed": seed}
        estimator = _build_estimator(args, hyperparameters)
        job_name = f"zetabench-sac-seed{seed}-{int(time.time())}"
        print(f"launching {job_name}: seed={seed} instance={args.instance_type}")
        estimator.fit(job_name=job_name, wait=False)
    print(f"submitted {len(args.seeds)} job(s); track them in the SageMaker console / wandb")


def run_hpo(args: argparse.Namespace) -> None:
    """Submit a Bayesian hyperparameter tuning sweep over SAC hyperparameters."""
    from sagemaker.tuner import ContinuousParameter, HyperparameterTuner

    estimator = _build_estimator(args, _base_hyperparameters(args))

    # Hydra dotted paths target configs/agent/sac.yaml fields.
    hyperparameter_ranges = {
        "agent.learning_rate": ContinuousParameter(1e-4, 1e-3),
        "agent.tau": ContinuousParameter(0.005, 0.02),
        "agent.gamma": ContinuousParameter(0.98, 0.999),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        # Mirror the metric WandbLoggingCallback / SB3 prints to stdout; adjust
        # the regex if the logged metric name changes.
        objective_metric_name="eval/success_rate",
        objective_type="Maximize",
        metric_definitions=[
            {"Name": "eval/success_rate", "Regex": r"eval/success_rate[\"']?\s*[:=]\s*([0-9.]+)"},
        ],
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel_jobs,
        base_tuning_job_name="zetabench-sac-hpo",
    )
    print(f"launching HPO: max_jobs={args.max_jobs} parallel={args.max_parallel_jobs}")
    tuner.fit(wait=False)
    print("HPO sweep submitted; track it in the SageMaker console / wandb")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by both subcommands."""
    parser.add_argument("--image-uri", required=True, help="ECR URI of the `sagemaker`-target image")
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--s3-output", required=True, help="S3 prefix for model.tar.gz outputs")
    parser.add_argument("--checkpoint-s3", default=None, help="S3 prefix for spot checkpoints (default: <s3-output>/checkpoints/)")
    parser.add_argument("--instance-type", default="ml.g5.xlarge", help="single-GPU instance type per job")
    parser.add_argument("--compute", default="large_gpu", help="Hydra compute profile (configs/compute/*.yaml)")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="environment steps per job")
    parser.add_argument("--use-spot", action="store_true", help="use managed spot instances")
    parser.add_argument("--max-run", type=int, default=24 * 3600, help="max job runtime in seconds")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    seeds = sub.add_parser("seeds", help="one job per seed")
    _add_common_args(seeds)
    seeds.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds to launch")
    seeds.set_defaults(func=run_seeds)

    hpo = sub.add_parser("hpo", help="hyperparameter tuning sweep")
    _add_common_args(hpo)
    hpo.add_argument("--max-jobs", type=int, default=12, help="total trials")
    hpo.add_argument("--max-parallel-jobs", type=int, default=4, help="concurrent trials")
    hpo.set_defaults(func=run_hpo)

    return parser


def main() -> None:
    """Parse arguments and dispatch to the selected launcher."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
