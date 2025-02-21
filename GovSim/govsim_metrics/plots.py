import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
import statsmodels.stats.api as sms
from lifelines import KaplanMeierFitter

def parse_log_env(file_path):
    """Parse a single log_env.json file and extract both before and after resource data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by round and get both before and after resource levels
    rounds = {}
    for entry in data:
        round_num = entry['round']
        if round_num not in rounds:
            rounds[round_num] = {
                'before': entry['resource_in_pool_before_harvesting'],
                'after': entry['resource_in_pool_after_harvesting']
            }
    
    # Convert to lists ordered by round
    max_round = max(rounds.keys())
    times = []
    resources = []
    
    # Create points ensuring no duplicates
    for r in range(max_round + 1):
        if r in rounds:
            if len(times) == 0 or times[-1] != float(r):
                times.append(float(r))
                resources.append(rounds[r]['before'])
            
            times.append(r + 0.83333)
            resources.append(rounds[r]['after'])
    
    return times, resources

def get_model_data(scenario_path, model_name):
    """Get data for all seeds of a specific model."""
    model_path = Path(scenario_path) / model_name
    seed_paths = [p for p in model_path.iterdir() if p.is_dir()]
    
    all_times = []
    all_resources = []
    
    print(f"\nProcessing model: {model_name}")
    print(f"Number of seeds found: {len(seed_paths)}")
    
    for seed_path in seed_paths:
        log_path = seed_path / 'log_env.json'
        if log_path.exists():
            times, resources = parse_log_env(log_path)
            all_times.append(times)
            all_resources.append(resources)
    
    # Find the maximum sequence length
    max_len = max(len(res) for res in all_resources)
    print(f"Max sequence length: {max_len}")
    
    # Pad shorter sequences with 0s instead of NaN
    padded_resources = []
    for resources in all_resources:
        if len(resources) < max_len:
            padding = [0.0] * (max_len - len(resources))
            padded_resources.append(resources + padding)
        else:
            padded_resources.append(resources)
    
    # Use the longest time array
    times_array = max(all_times, key=len)
    
    # Convert to numpy array and calculate mean and std
    resources_array = np.array(padded_resources)
    
    # Calculate mean and std (now including 0s for ended runs)
    mean_resources = np.mean(resources_array, axis=0)
    std_resources = np.std(resources_array, axis=0)
    
    # Find where the sequence truly ends (keep one final zero)
    valid_idx = np.ones_like(mean_resources, dtype=bool)
    for i in range(len(mean_resources)-1, 0, -1):
        if mean_resources[i] == 0 and mean_resources[i-1] == 0:
            valid_idx[i] = False
        else:
            break
    
    times_array = np.array(times_array)[:sum(valid_idx)]
    mean_resources = mean_resources[valid_idx]
    std_resources = std_resources[valid_idx]
    
    return times_array, mean_resources, std_resources

def plot_resources_over_time(base_path, scenario_name, save_path='govsim_metrics/plots'):
    """Plot resources over time for all models in a scenario using plotly and save as HTML."""
    # Create plots directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    scenario_path = Path(base_path) / scenario_name
    models = [p.name for p in scenario_path.iterdir() if p.is_dir()]
    
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'   # cyan
    ]
    
    fig = go.Figure()
    
    for idx, model in enumerate(models):
        times, mean_resources, std_resources = get_model_data(scenario_path, model)
        color = colors[idx % len(colors)]
        
        # Create a unique group identifier
        group_id = f'group_{model}'
        
        # Add main line
        fig.add_trace(go.Scatter(
            x=times,
            y=mean_resources,
            name=model,
            mode='lines',
            line=dict(
                width=2, 
                color=color,
                shape='linear'
            ),
            marker=dict(
                size=0
            ),
            legendgroup=group_id,
            hovertemplate=(
                "Round: %{x}<br>" +
                "Resource: %{y:.1f}<br>" +
                "<extra></extra>"
            ),
            visible=True
        ))
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([mean_resources + std_resources, 
                            (mean_resources - std_resources)[::-1]]),
            fill='toself',
            fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
            line=dict(width=0),
            mode='none',
            legendgroup=group_id,
            showlegend=False,
            name=f'{model} std',
            hoverinfo='skip',
            visible=True
        ))
    
    fig.update_layout(
        title=f'Resource Levels Over Time - {scenario_name}',
        xaxis_title='Round',
        yaxis_title='Resources in Pool',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        margin=dict(r=200),
        yaxis=dict(
            range=[0, None]
        )
    )
    
    # Save as HTML file
    filename = f"{scenario_name}_interactive.html"
    fig.write_html(os.path.join(save_path, filename))
    print(f"Saved interactive plot to {os.path.join(save_path, filename)}")

def plot_resources_over_time_matplotlib(base_path, scenario_name, selected_models=None, save_path='govsim_metrics/plots'):
    """Plot resources over time for selected models using matplotlib."""
    scenario_path = Path(base_path) / scenario_name
    all_models = [p.name for p in scenario_path.iterdir() if p.is_dir()]
    
    # If no models specified, use all models
    models = selected_models if selected_models else all_models
    
    # Create plots directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Define colors (matching plotly colors)
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'   # cyan
    ]
    
    plt.figure(figsize=(12, 6))
    
    for idx, model in enumerate(models):
        times, mean_resources, std_resources = get_model_data(scenario_path, model)
        color = colors[idx % len(colors)]
        
        # Plot main line
        plt.plot(times, mean_resources, color=color, label=model, linewidth=2)
        
        # Plot confidence bands
        plt.fill_between(times, 
                        mean_resources - std_resources,
                        mean_resources + std_resources,
                        color=color,
                        alpha=0.2)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Round')
    plt.ylabel('Resources in Pool')
    plt.title(f'Resource Levels Over Time - {scenario_name}')
    plt.ylim(bottom=0)  # Set minimum y-axis value to 0
    plt.xlim(0, 12)  # Set x-axis from 0 to 12
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    filename = f"{scenario_name}_{'_'.join(models) if len(models) <= 3 else 'multiple_models'}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

def compute_survival_months_stats(df):
    """Compute survival statistics from resource data."""
    # Get all run columns (excluding x and round)
    run_cols = [col for col in df.columns if col not in ['x', 'round']]
    
    durations = []
    events = []
    
    # Calculate survival times for each run
    for run in run_cols:
        event_time = None
        for i, r in df[run].items():
            if r == 0.0 or np.isnan(r):
                event_time = df.loc[i, 'round'] + 1
                break
        events.append(0 if event_time is None else 1)
        if event_time is None:
            event_time = df['round'].max() + 1
        durations.append(event_time)
    
    # Calculate statistics
    mean_survival = np.mean(durations)
    ci = sms.DescrStatsW(durations).tconfint_mean()
    ci_width = (ci[1] - ci[0]) / 2
    max_survival = max(durations)
    
    return mean_survival, ci_width, ci[0], ci[1], max_survival

def compute_additional_metrics(base_path, scenario_name, save_path='govsim_metrics/plots'):
    """Compute and save additional metrics plots."""
    scenario_path = Path(base_path) / scenario_name
    models = [p.name for p in scenario_path.iterdir() if p.is_dir()]
    
    # Create scenario subfolder
    scenario_save_path = Path(save_path) / scenario_name
    scenario_save_path.mkdir(parents=True, exist_ok=True)
    
    for model in models:
        print(f"\nProcessing additional metrics for model: {model}")
        model_path = Path(scenario_path) / model
        seed_paths = [p for p in model_path.iterdir() if p.is_dir()]
        
        # Get model data first - we'll need this for multiple plots
        times, mean_resources, std_resources = get_model_data(scenario_path, model)
        
        # Prepare data for survival analysis
        durations = []
        events = []
        
        for seed_path in seed_paths:
            log_path = seed_path / 'log_env.json'
            if log_path.exists():
                times_seed, resources_seed = parse_log_env(log_path)
                
                # Find when resources hit 0 or end of simulation
                event_time = None
                for i, r in enumerate(resources_seed):
                    if r == 0.0:
                        event_time = times_seed[i]
                        break
                
                events.append(1 if event_time is not None else 0)
                durations.append(event_time if event_time is not None else max(times_seed))
        
        # 1. Kaplan-Meier Survival Analysis
        kmf = KaplanMeierFitter()
        kmf.fit(durations, events, label=model)
        
        fig_kaplan = go.Figure()
        
        # Add KM estimate
        fig_kaplan.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[model],
                mode='lines',
                name=model,
                line=dict(width=2)
            )
        )
        
        # Add confidence intervals if they exist
        if hasattr(kmf, 'confidence_interval_'):
            ci_columns = kmf.confidence_interval_.columns
            print(f"CI columns available: {ci_columns}")
            
            ci_lower = kmf.confidence_interval_.iloc[:, 0]
            ci_upper = kmf.confidence_interval_.iloc[:, 1]
            
            fig_kaplan.add_trace(
                go.Scatter(
                    x=kmf.confidence_interval_.index,
                    y=ci_upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            fig_kaplan.add_trace(
                go.Scatter(
                    x=kmf.confidence_interval_.index,
                    y=ci_lower,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(0,0,255,0.2)',
                    line=dict(width=0),
                    showlegend=False
                )
            )
        
        fig_kaplan.update_layout(
            title=f'Survival Analysis - {model}',
            xaxis_title='Round',
            yaxis_title='Survival probability',
            yaxis_range=[0, 1.05],
            height=500,
            showlegend=True
        )
        
        # Save Kaplan-Meier plot
        fig_kaplan.write_html(str(scenario_save_path / f"kaplan_meier_{model}.html"))
        
        # 2. Percentage of collapse over time
        perc_collapsed = [1 if r <= 0 else 0 for r in mean_resources]
        
        fig_collapse = go.Figure()
        fig_collapse.add_trace(
            go.Scatter(
                x=times,
                y=perc_collapsed,
                mode='lines',
                name=model,
                line=dict(width=2)
            )
        )
        
        fig_collapse.update_layout(
            title=f'Percentage of Collapse Over Time - {model}',
            xaxis_title='Round',
            yaxis_title='Percentage Collapsed',
            yaxis_range=[0, 1.05],
            height=500,
            showlegend=True
        )
        
        # Save collapse plot
        fig_collapse.write_html(str(scenario_save_path / f"collapse_percentage_{model}.html"))
        
        # 3. Compute Gini coefficient over time
        gini_coeffs = []
        for resources in mean_resources:
            if resources > 0:
                # For single value, Gini coefficient is 0
                gini_coeffs.append(0)
            else:
                gini_coeffs.append(np.nan)
        
        # Plot Gini coefficients
        fig_gini = go.Figure()
        fig_gini.add_trace(
            go.Scatter(
                x=times,
                y=gini_coeffs,
                mode='lines',
                name=model,
                line=dict(width=2)
            )
        )
        
        fig_gini.update_layout(
            title=f'Gini Coefficient Over Time - {model}',
            xaxis_title='Round',
            yaxis_title='Gini Coefficient',
            yaxis_range=[0, 1],
            height=500,
            showlegend=True
        )
        
        # Save Gini plot
        fig_gini.write_html(str(scenario_save_path / f"gini_coefficient_{model}.html"))

# Define model groupings with path names and display names
MODEL_GROUPS = {
    'Base Models': {
        'paths': [
            'Meta-Llama-2-7B',
            'Meta-Llama-2-13B',
            'Meta-Llama-3-8B',
            'Mistral-7B-Instruct-v0.2'
        ],
        'display_names': {
            'Meta-Llama-2-7B': 'Llama-2-7B',
            'Meta-Llama-2-13B': 'Llama-2-13B',
            'Meta-Llama-3-8B': 'Llama-3-8B',
            'Mistral-7B-Instruct-v0.2': 'Mistral-7B'
        }
    },

    'Base Models and Phi-4': {
        'paths': [
            'Meta-Llama-2-7B',
            'Meta-Llama-2-13B',
            'Meta-Llama-3-8B',
            'Mistral-7B-Instruct-v0.2',
            'Phi-4',
            'DeepSeek-R1-Distill-Llama-8B'
        ],
        'display_names': {
            'Meta-Llama-2-7B': 'Llama-2-7B',
            'Meta-Llama-2-13B': 'Llama-2-13B',
            'Meta-Llama-3-8B': 'Llama-3-8B',
            'Mistral-7B-Instruct-v0.2': 'Mistral-7B',
            'Phi-4': 'Phi-4',
            'DeepSeek-R1-Distill-Llama-8B': 'DeepSeek Distill Llama-3-8B',
        }
    },

    'Phi 4': {
        'paths': [
            'Phi-4',
            'Systemic_Phi-4',
            'Univ_alt_Phi-4',
            'Univ_Phi-4',
            'VeilOfIgnorance_Phi-4'
        ],
        'display_names': {
            'Phi-4': 'Base',
            'Systemic_Phi-4': 'Systemic',
            'Univ_alt_Phi-4': 'Univ-alt',
            'Univ_Phi-4': 'Univ',
            'VeilOfIgnorance_Phi-4': 'VoI'
        }
    },
    'Meta Llama 3 8B': {
        'paths': [
            'Meta-Llama-3-8B',
            'Systemic_Meta-Llama-3-8B',
            'Univ_alt_Meta-Llama-3-8B',
            'Univ_Meta-Llama-3-8B',
            'VeilOfIgnorance_Meta-Llama-3-8B'
        ],
        'display_names': {
            'Meta-Llama-3-8B': 'Base',
            'Systemic_Meta-Llama-3-8B': 'Systemic',
            'Univ_alt_Meta-Llama-3-8B': 'Univ-alt',
            'Univ_Meta-Llama-3-8B': 'Univ',
            'VeilOfIgnorance_Meta-Llama-3-8B': 'VoI'
        }
    },
    'Meta Llama 2': {
        'paths': [
            'Meta-Llama-2-7B',
            'Meta-Llama-2-13B',
            'Univ_Meta-Llama-2-7B'
        ],
        'display_names': {
            'Meta-Llama-2-7B': 'Llama-2-7B',
            'Meta-Llama-2-13B': 'Llama-2-13B',
            'Univ_Meta-Llama-2-7B': 'Univ-7B'
        }
    },
    'Mistral 7B Instruct v0.2': {
        'paths': [
            'Mistral-7B-Instruct-v0.2',
            'Systemic_Mistral-7B-Instruct-v0.2',
            'Univ_alt_Mistral-7B-Instruct-v0.2',
            'Univ_Mistral-7B-Instruct-v0.2',
            'VeilOfIgnorance_Mistral-7B-Instruct-v0.2'
        ],
        'display_names': {
            'Mistral-7B-Instruct-v0.2': 'Base',
            'Systemic_Mistral-7B-Instruct-v0.2': 'Systemic',
            'Univ_alt_Mistral-7B-Instruct-v0.2': 'Univ-alt',
            'Univ_Mistral-7B-Instruct-v0.2': 'Univ',
            'VeilOfIgnorance_Mistral-7B-Instruct-v0.2': 'VoI'
        }
    },
    'Qwen 2.5 7B': {
        'paths': [
            'Qwen2.5-7B',
            'DeepSeek-R1-Distill-Qwen-7B'
        ],
        'display_names': {
            'Qwen2.5-7B': 'Qwen-7B',
            'DeepSeek-R1-Distill-Qwen-7B': 'DeepSeek-Qwen'
        }
    },
    'DeepSeekR1 Distill Llama 8B': {
        'paths': [
            'DeepSeek-R1-Distill-Llama-8B',
            'Univ_DeepSeek-R1-Distill-Llama-8B',
            'Univ_alt_DeepSeek-R1-Distill-Llama-8B',
            'Systemic_DeepSeek-R1-Distill-Llama-8B',
            'VeilOfIgnorance_DeepSeek-R1-Distill-Llama-8B'
        ],
        'display_names': {
            'DeepSeek-R1-Distill-Llama-8B': 'Base',
            'Univ_DeepSeek-R1-Distill-Llama-8B': 'Univ',
            'Univ_alt_DeepSeek-R1-Distill-Llama-8B': 'Univ-alt',
            'Systemic_DeepSeek-R1-Distill-Llama-8B': 'Systemic',
            'VeilOfIgnorance_DeepSeek-R1-Distill-Llama-8B': 'VoI'
        }
    }
}

def plot_model_group_comparison_matplotlib(base_path, scenario_name, model_group, group_name, save_path='govsim_metrics/plots'):
    """Plot comparison of different variants of a base model using matplotlib."""
    scenario_path = Path(base_path) / scenario_name
    
    # Create scenario-specific directory structure
    save_dir = Path(save_path) / 'resource_plots' / scenario_name.split('_')[0]
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 3))
    
    # Store data for overlap checking
    all_data = {}
    
    # First pass: collect all data
    for model in model_group['paths']:
        try:
            times, mean_resources, std_resources = get_model_data(scenario_path, model)
            display_name = model_group['display_names'][model]
            all_data[display_name] = mean_resources.tolist()  # Convert to list for comparison
        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")
    
    # Find overlapping lines
    overlaps = {}
    for name1 in all_data:
        for name2 in all_data:
            if name1 < name2:  # Check each pair only once
                if all_data[name1] == all_data[name2]:
                    overlaps[name1] = name2
    
    # Second pass: actual plotting
    for model in model_group['paths']:
        try:
            times, mean_resources, std_resources = get_model_data(scenario_path, model)
            display_name = model_group['display_names'][model]
            
            # Modify label if there's an overlap
            if display_name in overlaps:
                display_name = f"{display_name} (identical to {overlaps[display_name]})"
            elif display_name in overlaps.values():
                continue  # Skip plotting the overlapping line
            
            line = plt.plot(times, mean_resources, 
                    label=display_name, 
                    linewidth=2)
            
            color = line[0].get_color()
            
            plt.fill_between(times,
                           mean_resources - std_resources,
                           mean_resources + std_resources,
                           color=color,
                           alpha=0.2)
            
        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Resources in Pool', fontsize=14)
    plt.ylim(bottom=0)
    
    # Increase font size for tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust the plot layout to make room for legend
    plt.subplots_adjust(right=0.85)
    
    # Position legend in the empty space we created with larger font
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    
    plt.savefig(
        save_dir / f"resources_{group_name}_comparison.pdf",
        format='pdf',
        bbox_inches='tight'
    )
    plt.close()

def plot_model_group_comparison_averaged_scenarios(base_path, scenarios, model_group, group_name, save_path='govsim_metrics/plots'):
    """Plot comparison of different variants of a base model averaged across all scenarios."""
    
    # Create directory for averaged plots
    save_dir = Path(save_path) / 'resource_plots' / 'averaged_scenarios'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store aggregated data for each model
    aggregated_data = {}
    
    # First collect all data across scenarios
    for model in model_group['paths']:
        all_times = []
        all_means = []
        all_stds = []
        
        for scenario in scenarios:
            try:
                scenario_path = Path(base_path) / scenario
                times, mean_resources, std_resources = get_model_data(scenario_path, model)
                all_times.append(times)
                all_means.append(mean_resources)
                all_stds.append(std_resources)
            except Exception as e:
                print(f"Error processing model {model} in scenario {scenario}: {str(e)}")
                continue
        
        if all_means:  # Only process if we have data
            # Use shortest length to ensure alignment
            min_length = min(len(m) for m in all_means)
            
            # Trim all arrays to minimum length
            aligned_means = [m[:min_length] for m in all_means]
            aligned_stds = [s[:min_length] for s in all_stds]
            times = all_times[0][:min_length]  # Use times from first scenario
            
            # Calculate average mean and propagate uncertainties
            mean_across_scenarios = np.mean(aligned_means, axis=0)
            # Combine standard deviations using error propagation
            combined_std = np.sqrt(np.mean([s**2 for s in aligned_stds], axis=0))
            
            display_name = model_group['display_names'][model]
            aggregated_data[display_name] = {
                'times': times,
                'mean': mean_across_scenarios,
                'std': combined_std
            }
    
    # Create the plot
    plt.figure(figsize=(12, 3))
    
    # Plot the data
    for display_name, data in aggregated_data.items():
        line = plt.plot(data['times'], data['mean'],
                       label=display_name,
                       linewidth=2)
        
        color = line[0].get_color()
        plt.fill_between(data['times'],
                        data['mean'] - data['std'],
                        data['mean'] + data['std'],
                        color=color,
                        alpha=0.2)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Resources in Pool', fontsize=14)
    plt.ylim(bottom=0)
    
    # Increase font size for tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust the plot layout to make room for legend
    plt.subplots_adjust(right=0.85)
    
    # Position legend in the empty space we created with larger font
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    
    plt.savefig(
        save_dir / f"resources_{group_name}_comparison_averaged.pdf",
        format='pdf',
        bbox_inches='tight'
    )
    plt.close()

# Main execution
base_path = r"simulation\results"
scenarios = ["fishing_v6.4_final", "sheep_v6.4_final", "pollution_v6.4_final"]

for scenario in scenarios:
    print(f"\nProcessing scenario: {scenario}")
    
    for group_name, model_group in MODEL_GROUPS.items():
        print(f"Processing model group: {group_name}")
        plot_model_group_comparison_matplotlib(base_path, scenario, model_group, group_name)

for group_name, model_group in MODEL_GROUPS.items():
    print(f"\nProcessing averaged scenarios for model group: {group_name}")
    plot_model_group_comparison_averaged_scenarios(base_path, scenarios, model_group, group_name)