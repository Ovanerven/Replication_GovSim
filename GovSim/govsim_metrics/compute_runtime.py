import os
import re
from pathlib import Path
from collections import defaultdict

MODEL_PATTERNS = {
    "DeepSeek-R1-Distill-Llama-8B": ["deepseek-r1-distill-llama-8b", "deepseek"],
    "Meta-Llama-2-13B": ["llama-2-13b", "meta-llama--llama-2-13b"],
    "Meta-Llama-2-7B": ["llama-2-7b", "meta-llama--llama-2-7b"],
    "Meta-Llama-3-8B": ["llama-3-8b", "meta-llama--llama-3-8b"],
    "Mistral-7B-Instruct-v0.2": ["mistral-7b", "mistralai--mistral-7b"],
    "Phi-4": ["phi-4"],
    "Qwen2.5-7B": ["qwen2.5-7b", "qwen2.5"]
}

def extract_wall_time(file_path: str) -> float:
    """Extract wall-clock time from a slurm output file and convert to hours."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # First check if job completed successfully
    state_match = re.search(r'State: ([A-Z]+)', content)
    if not state_match or state_match.group(1) != "COMPLETED":
        return 0
        
    # Count number of runs in this file by looking for seed definitions
    num_runs = len(re.findall(r'seed: \d+', content))
    if num_runs == 0:  # Fallback to looking for wandb run initializations
        num_runs = len(re.findall(r'wandb: Tracking run with wandb', content))
    if num_runs == 0:  # If still no runs found, assume it's one run
        num_runs = 1
        
    # Extract runtime from job stats section
    match = re.search(r'Job Wall-clock time: (\d+):(\d+):(\d+)', content)
    if not match:
        return 0
        
    hours, minutes, seconds = map(int, match.groups())
    total_hours = hours + minutes/60 + seconds/3600
    
    # Divide total time by number of runs to get per-run time
    return total_hours / num_runs

def determine_model_and_experiment(content: str) -> tuple[str, str]:
    """Extract model type and experiment type from the config section."""
    try:
        # Find the config section
        config_match = re.search(r'experiment:(.*?)(?=\n\[|\Z)', content, re.DOTALL)
        if not config_match:
            return "unknown", "unknown"
            
        config_text = config_match.group(1).lower()
        
        # Determine model
        model = "unknown"
        for model_name, patterns in MODEL_PATTERNS.items():
            if any(pattern.lower() in content.lower() for pattern in patterns):
                model = model_name
                break
        
        # Determine experiment
        experiment = "default"
        if "inject_universalization: true" in content.lower():
            experiment = "universalization"
        elif "inject_universalization_alternative: true" in content.lower():
            experiment = "universalization_alternative"
        elif "inject_systemic: true" in content.lower():
            experiment = "systemic"
        elif "inject_veilofignorance: true" in content.lower():
            experiment = "veil_of_ignorance"
            
        return model, experiment
    except Exception as e:
        print(f"Error parsing config: {str(e)}")
        return "unknown", "unknown"

def split_into_runs(content: str) -> list[str]:
    """Split a slurm output file into separate runs based on config sections and wandb initializations."""
    # Look for full experiment config sections
    runs = []
    config_sections = re.finditer(r'experiment:.*?(?=experiment:|wandb: Currently logged in|$)', content, re.DOTALL)
    
    for section in config_sections:
        run_content = section.group(0)
        # Find associated wandb initialization if it exists
        wandb_match = re.search(r'wandb: Tracking run.*?(?=wandb: Tracking run|$)', content[section.end():], re.DOTALL)
        if wandb_match:
            run_content += wandb_match.group(0)
        runs.append(run_content)
    
    # If no runs found through config sections, try splitting by wandb initializations
    if not runs:
        wandb_splits = re.split(r'(?=wandb: Tracking run with wandb)', content)
        runs = [split for split in wandb_splits if split.strip()]
    
    # If still no runs found, treat entire content as one run
    if not runs:
        runs = [content]
        
    return runs

def format_hours(hours: float) -> str:
    """Convert hours to a readable time format (HH:MM:SS)."""
    total_seconds = int(hours * 3600)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def analyze_runtimes(slurm_output_dir):
    runtimes = defaultdict(list)
    total_gpu_hours = 0
    
    for filename in os.listdir(slurm_output_dir):
        if not filename.endswith('.out'):
            continue
            
        filepath = os.path.join(slurm_output_dir, Path(filename))
        
        # First check if job completed successfully
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        state_match = re.search(r'State: ([A-Z]+)', content)
        if not state_match or state_match.group(1) != "COMPLETED":
            continue
            
        # Extract total job time
        time_match = re.search(r'Job Wall-clock time: (\d+):(\d+):(\d+)', content)
        if not time_match:
            continue
            
        hours, minutes, seconds = map(int, time_match.groups())
        total_job_hours = hours + minutes/60 + seconds/3600
        
        # Split content into separate runs
        runs = split_into_runs(content)
        num_runs = len(runs)
        
        # Process each run
        for run_content in runs:
            # Determine model and experiment type for this run
            model_name, experiment = determine_model_and_experiment(run_content)
            if model_name == "unknown":
                continue
                
            # Calculate runtime for this individual run
            runtime_hours = total_job_hours / num_runs
            total_gpu_hours += runtime_hours
            
            # Store runtime with combined model+experiment key
            key = f"{model_name} ({experiment})"
            runtimes[key].append(runtime_hours)
    
    # Prepare results for both printing and CSV
    results = []
    model_aggregated_times = defaultdict(list)
    
    print("\nAverage Runtimes per Model/Experiment:")
    print("-" * 50)
    
    for model_exp, times in sorted(runtimes.items()):
        avg_time = sum(times) / len(times)
        total_runs = len(times)
        total_hours = sum(times)
        
        # Extract model name from the combined key
        model_name = model_exp.split(" (")[0]
        model_aggregated_times[model_name].append(avg_time)
        
        # Store results for CSV
        results.append({
            'model_experiment': model_exp,
            'average_runtime': format_hours(avg_time)
        })
        
        # Print formatted results
        print(f"{model_exp}:")
        print(f"  Average runtime: {format_hours(avg_time)} (HH:MM:SS)")
        print(f"  Number of runs: {total_runs}")
        print(f"  Total compute hours: {format_hours(total_hours)} (HH:MM:SS)")
        print()
    
    # Print model-level averages
    print("\nOverall Model Averages (across all experiments):")
    print("-" * 50)
    for model, times in sorted(model_aggregated_times.items()):
        model_mean = sum(times) / len(times)
        print(f"{model}:")
        print(f"  Mean runtime across experiments: {format_hours(model_mean)} (HH:MM:SS)")
        print(f"  Number of experiments: {len(times)}")
        print()
    
    print(f"Total GPU Hours Used: {format_hours(total_gpu_hours)} (HH:MM:SS)")
    
    # Prepare both detailed and aggregated results for CSV
    detailed_results = []
    aggregated_results = []
    
    for model_exp, times in sorted(runtimes.items()):
        avg_time = sum(times) / len(times)
        model_name = model_exp.split(" (")[0]
        model_aggregated_times[model_name].append(avg_time)
        
        detailed_results.append({
            'model_experiment': model_exp,
            'average_runtime': format_hours(avg_time)
        })
    
    # Add model-level averages to results
    for model, times in sorted(model_aggregated_times.items()):
        model_mean = sum(times) / len(times)
        aggregated_results.append({
            'model_experiment': f"{model} (overall mean)",
            'average_runtime': format_hours(model_mean)
        })
    
    # Combine both types of results
    all_results = detailed_results + aggregated_results
    
    # Write results to CSV
    import csv
    csv_file = "utils/metrics/model_runtimes.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model_experiment', 'average_runtime'])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nResults have been saved to {csv_file}")


if __name__ == "__main__":
    slurm_output_dir = r"utils/slurm_outputs"
    analyze_runtimes(slurm_output_dir)