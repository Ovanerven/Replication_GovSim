# [Re] Cooperatre or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents

This repository extends the GovSim codebase originally developed by Piatti et al. (2024) for their paper "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents."

Our main changes to the GovSim codebase are mainly to implement our new experiments which we outline in our paper. These are:

- Alternative Universalization: `experiment=fish_baseline_concurrent_universalization`
- Systemic: `experiment=fish_baseline_concurrent_systemic`
- Veil of Ignorance: `experiment=fish_baseline_concurrent_veil_of_ignorance`

To run these experiments, you simply change the experiment argument to the desired experiment name above, and run the python script to complete the GovSim run. 
```bash
cd GovSim  # Replace with the correct path

# Run the Python script to load the model
python3 -m simulation.main experiment=pollution_baseline_concurrent_universalization \ # Replace with experiment of choice
    llm.path=.cache/huggingface/hub/models--microsoft--phi-4/snapshots/f957856cd926f9d681b14153374d755dd97e45ed \ # Replace with path to model
    llm.temperature=0 \
    seed=1 \ # Replace with seed
    group_name=Phi-4-Univ # Replace with model
```

We also added functionality within the pathfinder library to prompt the new models we used: Phi-4, Qwen2.5-7B, and DeepSeek-R1-Distill-Llama-8B. An outline of the changes made can be found in the Summary of Changes at the bottom of this README.

### GovSim Metrics
Additionally, we added some scripts to compute the various metrics we reference in our paper, since we were not able to find where the original authors computed the metrics. The main one is compute_metrics.py, which computes the main GovSim metrics including survival time, survival rate, efficiency, equality, gain, and overusage. The `govsim/govsim_metrics` directory contains these scripts. 

```bash
python compute_metrics.py
```

This will generate CSV files in `metrics/` containing:
- `govsim_metrics.csv`: Raw metrics for all runs, with means for each scenario and experiment at the bottom
- `govsim_metrics_by_scenario.csv`: Aggregated metrics per scenario
- `model_differences.csv`: Comparative analysis between models

```bash
python plots.py
```

This will create plots in `plots/` including:
- Resource plots for each scenario
- Model comparison plots
- Averaged results across scenarios

- `compute_runtime.py`: Analyzes execution times from SLURM output files to compare computational efficiency between models

- `t_test.py`: We also added a script to perform the two-tailed t-test for significance on the effect of each experiment on all the metrics.

Finally, we also provided some of our own environment files: `environment_snellius.yml` and `environment_windows.yml`, which were used to run GovSim on the Dutch national cluster (Snellius) and a windows environment, respectively. Notably, the original environments found in the original GovSim repository were not compatible with our new models, and also did not specify the correct versions for some key libraries, particularly for the visualizaiton app. As such, we recommend using our provided environments.

## Summary of Changes

### Adding New Models to GovSim

All model-related changes take place in `GovSim/pathfinder`, the prompting library used for GovSim.

1. In `GovSim/pathfinder/library/chat.py`:
   - Add new model class pointing to compatible prompting template

2. In `GovSim/pathfinder/library/templates_jinja`:
   - Add compatible chat templates to directory

3. In `GovSim/pathfinder/library/loader.py`:
   - Add logic to load the new model class based on model name
   - Add flash attention if needed

Extra steps were needed for the distill model, which required a greatly increased max_tokens value, due to the model's tendency to reason for much longer than other models. 

4. In `GovSim/pathfinder/library/_gen.py`:
   - Update `max_tokens=16384` 

5. In `GovSim/pathfinder/library/_find.py`:
   - Update `max_tokens=16384`

6. In `GovSim\simulation\utils`
    - Update `max_tokens=16384` in both the gen and find functions

### Adding New Experiments

All experiment-related changes take place in `GovSim/simulation`.

1. In `GovSim/simulation/scenarios/{scenario}/environment/env.py`:
   - Add new prompt function at the top of env.py script
   - Add the new prompt function as a method in the `{scenario}ConcurrentEnv` class

2. In `GovSim/simulation/scenarios/common/environment/concurrent_env.py`:
   - Add prompt injection block for the new experiment, which calls the function defined in env.py when the config is set to run the experiment

3. In `GovSim/simulation/scenarios/{scenario}/conf/experiment`:
   - Add default config settings (set `inject_{new experiment} = false`) in `{scenario}_baseline_concurrent.yaml`
   - Add new experiment config files for each new experiment:
     - Universalization alternative
     - Systemic
     - Veil of ignorance

4. Repeat these steps for all three scenarios

## References

- Piatti et al. (2024). Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents. 
