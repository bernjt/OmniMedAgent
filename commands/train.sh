PYTHONPATH=code torchrun --nproc_per_node=8 -m omni_med_agent.cli.train --config configurations/main.yaml
