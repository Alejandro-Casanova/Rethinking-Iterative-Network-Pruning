import pandas as pd
import matplotlib
import json
from pathlib import Path
from typing import Literal
from itertools import chain

current_dir = Path.cwd()
results_dir = current_dir / "my_runs"

def time_string_to_minutes(time_str: str) -> float:
    # print(time_str)
    # Split the time string into hours, minutes, and seconds.decimals
    hours, minutes, seconds = time_str.split(':')
    
    # Convert each part to float
    hours = float(hours)
    minutes = float(minutes)
    seconds = float(seconds)
    
    # Convert hours and seconds to minutes
    total_minutes = (hours * 60) + minutes + (seconds / 60)
    
    return total_minutes

def plot_results(
      filter: str = None
):
  
  search_str = "**/results.json" if filter is None else f'**/{filter}/**/results.json'
  files = list(results_dir.glob(search_str))

  data = {}
  final_data = []
  fixed_params = {'avg':{}}

  for file in files:
    with open(file) as user_file:
      file_contents = user_file.read()
      parsed_json = json.loads(file_contents)

      final_data.append(parsed_json)
      # data[file.parent.stem] = parsed_json
  
  #print(final_data)

  # for dataset_key, dataset_value in data.items():
  #   # Get fixed params
  #   fixed_params[dataset_key] = {}
  #   for key, value in dataset_value.items():
  #     if key == "data": continue
  #     fixed_params[dataset_key][key] = value
  #   # Get data in the right format
  #   for i, datapoint in enumerate(dataset_value["data"]):
  #     if len(final_data) <= i:
  #       final_data.append({
  #         param: datapoint[param],
  #         dataset_key: datapoint[error_var],
  #         "avg": datapoint[error_var] / len(data)
  #       })
  #     else:
  #       final_data[i][dataset_key] = datapoint[error_var]
  #       final_data[i]["avg"] += (datapoint[error_var] / len(data))

  # print(json.dumps(final_data, sort_keys=True, indent=4))
  # print(json.dumps(fixed_params, sort_keys=True, indent=4))
  # exit()

  # Save to dictionary   
  #ith open(f"{error_var}_{param}_{select}", "w") as fp:
  #  json.dump(final_data , fp) 

  df = pd.DataFrame(final_data)
  # print(df)
  df['runtime'] = df['runtime'].apply(time_string_to_minutes) # Convert runtime string to float (minutes)
  df = df.sort_values(by='prune_rate')
  print(df)

  # Group by the 'Category' column and calculate the mean of each group
  grouped_df = df.groupby('prune_rate')

  # Calculate mean and std over different seeds
  df_mean = grouped_df.mean().reset_index()
  df_std = grouped_df.std().reset_index()
  print(df_mean)
  print(df_std)

  # Values to include in pivot table
  values_to_include = ['runtime']

  # Pivot table with prune_rate as columns and prune_iterations as rows, averaging all columns
  pivot_table_mean = df.pivot_table(index='prune_iterations', columns='prune_rate', values=values_to_include, aggfunc='mean').round(2)
  pivot_table_std = df.pivot_table(index='prune_iterations', columns='prune_rate', values=values_to_include, aggfunc='std').round(2)

  # Combine the mean and standard deviation tables into a single table
  pivot_table_combined = pd.DataFrame()

  # Iterating through columns to combine mean and std
  for value in values_to_include:
    for prune_rate in pivot_table_mean[value].columns:
        mean_str = pivot_table_mean[value][prune_rate].map('{:.2f}'.format)
        std_str = pivot_table_std[value][prune_rate].map('{:.2f}'.format)
        combined = mean_str + " ± " + std_str
        pivot_table_combined[(value, prune_rate)] = combined

  # Set the index to prune_iterations
  pivot_table_combined.index = pivot_table_mean.index

  # Define a function to highlight the maximum value in a DataFrame column
  def highlight_max(s: pd.Series):
      aux = s.apply(str.split, args="±")
      aux.apply(list.reverse)
      aux = aux.apply(list.pop)
      aux = aux.apply(float)
      is_max = aux == aux.max()
      return ['font-weight: bold; color: red;' if v else '' for v in is_max]
  
  # TODO function: highlight second max

  # Apply the highlight_max function to each column
  styled_pivot_table = pivot_table_combined.style.apply(highlight_max, axis=0)

  # Save the styled DataFrame to an HTML file
  styled_pivot_table.to_html('styled_pivot_table.html')

  return

  y_vars = list(data.keys()) 
  y_vars.sort()
  y_vars.append("avg")
  y_vars.insert(0, param)
  df=df.reindex(columns=y_vars)
  y_vars.pop(0)
  y_vars_main_plot = y_vars.copy()
  y_vars_main_plot.pop(len(y_vars)-1)
  # print(y_vars_main_plot)
  print(df)

  for key, val in fixed_params.items():
    idx = df.idxmin("rows", skipna=False)[key]
    min = df.at[idx, param]
    val[param] = min
    # print(f"Index: {idx}, Val: {min}")
  idx = df.idxmin("rows", skipna=False)['avg']
  min = df.at[idx, param]
  fixed_params['avg'][param] = min
  
  df2 = pd.DataFrame(fixed_params)
  df2.sort_index(axis=1, inplace=True)
  print(df2)

  title = ["" for i in range(len(data)+1)]
  title[0] = f"{error_var} for tuning of {param} ({select})"
  title[1] = f"Average {error_var}"
  df.plot(x=param, y=y_vars, logx=True,
          subplots=[y_vars_main_plot, ["avg"]],
          title=title)



if __name__ == "__main__":

  plot_results('one-shot')
