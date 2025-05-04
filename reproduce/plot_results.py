import argparse
import glob
import os
import matplotlib
import pandas as pd
# import matplotlib
import json
from pathlib import Path
from typing import Literal
from itertools import chain
from matplotlib import colors, pyplot as plt

current_dir = Path.cwd()
results_dir = os.path.normpath(os.path.join(current_dir, "reproduce/my_runs"))

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
    filter_list: list = None, # Select filter from results directory, after date. E.g. 'one-shot', 'dynamic-iterative'...
    plot_variables_list: list = ['acc_drop'],
    highlight_min: bool = False, # Highlight min values, instead of max
    latex_out: bool = False,
    output_path: str = None
):
    if isinstance(filter_list, str):
        print("Single String passed instead of list... But don't worry, I can handle it ;)")
        filter_list = [filter_list]
    if isinstance(plot_variables_list, str):
        print("Single String passed instead of list... But don't worry, I can handle it ;)")
        plot_variables_list = [plot_variables_list]

    files = []
    for filter in filter_list:
        search_str = "**/results.json" if filter is None else f'**/{filter}/*/*/results.json'
        search_path = os.path.join(results_dir, search_str)
        search_path = os.path.normpath(search_path)
        print(f"Search path: {search_path}")
        files += list(glob.glob(search_path, recursive=True))

    if len(files) == 0:
        raise Exception("No results parsed!")
    else:
        print(f"Files found: {len(files)}")
    
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
    # print(df)

    # Drop columns I can not handle for now
    if 'latency_pruned' in df.columns:
        df = df.drop(columns=['latency_pruned', 'latency_original', 'latency_delta'])
    print(df)

    # Group by the 'Category' column and calculate the mean of each group
    grouped_df = df.groupby('prune_rate')

    # Calculate mean and std over different seeds
    df_mean = grouped_df.mean().reset_index()
    df_std = grouped_df.std().reset_index()
    print(df_mean)
    print(df_std)

    if not df_mean["seed"].eq(2.0).all(axis=0):
        raise Exception("Average of seeds should be 2, to ensure no samples were left out.")

    # Values to include in pivot table
    values_to_include = plot_variables_list

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
    def highlight(highlight_min: bool = False):
        def highlight_fun(s: pd.Series):
            aux = s.apply(str.split, args="±")
            aux.apply(list.reverse)
            aux = aux.apply(list.pop)
            aux = aux.apply(float)
            if highlight_min:
                is_highlighted = aux == aux.min()
                is_highlighted_2 = aux == aux.nsmallest(2).iloc[-1]
            else:
                is_highlighted = aux == aux.max()
                is_highlighted_2 = aux == aux.nlargest(2).iloc[-1]
            
            if latex_out:
                first_value_highlighted = ['cellcolor:[HTML]{f08080}; textbf:--rwrap;' if v else '' for v in is_highlighted]
                second_value_highlighted = ['cellcolor:[HTML]{fdfd96}; textbf:--rwrap;' if v else '' for v in is_highlighted_2]
            else:
                first_value_highlighted = ['font-weight: bold; color: red;' if v else '' for v in is_highlighted]
                second_value_highlighted = ['font-weight: bold; color: yellow;' if v else '' for v in is_highlighted_2]
            return [item1 if item1 != '' else item2 for item1, item2 in zip(first_value_highlighted, second_value_highlighted)]
        return highlight_fun
    
   
    def background_grad_fun(s: pd.Series, cmap='PuBu', low=0, high=0):
        """Background gradient"""
        a = s.apply(str.split, args="±")
        a.apply(list.reverse)
        a = a.apply(list.pop)
        a = a.apply(float)
        range = a.max() - a.min()
        norm = colors.Normalize(a.min() - (range * low),
                            a.max() + (range * high))
        normed = norm(a.values)
        c = [colors.rgb2hex(x) for x in matplotlib.colormaps.get_cmap(cmap)(normed)]
        # c_text = [colors.rgb2hex(x) for x in matplotlib.colormaps.get_cmap('gray')(normed)]
        c_text = ["#FFFFFF" if x > 0.5 else "#000000" for x in normed]
        return [f"background-color: {bg_color}; color: {text_color};" for bg_color, text_color in zip(c, c_text)]


    # Apply the highlight_max function to each column
    # styled_pivot_table = pivot_table_combined.style.apply(
    #     highlight(highlight_min),
    #     axis=0
    # )

    styled_pivot_table = pivot_table_combined.style.apply(
        background_grad_fun,
        cmap='Blues'
    )

    if latex_out:
        latex_str = styled_pivot_table.to_latex(
            clines="all;data",
            label="blablabla",
            caption="BLABLABLA",
            multirow_align="c",
            convert_css=True,
            position_float="centering",
            multicol_align="|c|",
            hrules=True,
            # index=False,
            # formatters={"name": str.upper},
            # float_format="{:.1f}".format,
        )  

        # Clean up header
        for plot_var in plot_variables_list:
            latex_str = latex_str.replace(f"(\'{plot_var}\',", "")
        latex_str = latex_str.replace(")", "")
        latex_str = latex_str.replace("prune_iterations", "p\\_iter")

        print(latex_str)
        return
    
    # Save the styled DataFrame to an HTML file
    if output_path is None:
        output_path = 'styled_pivot_table.html'
    styled_pivot_table.to_html(output_path)

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

    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument("-f", "--experiment-filters", type=str, default="one-shot,dynamic-iterative")
    parser.add_argument("-plt", "--plot-variables", type=str, default="acc_drop")
    parser.add_argument("-ltx", "--latex-out", action="store_true", default=False)
    parser.add_argument("-hmn", "--highlight-min", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    args = parser.parse_args()

    if args.experiment_filters is not None:
        selection_filters = args.experiment_filters.split(",")
    else:
        # one-shot, dynamic-iterative, dynamic-iterative-flops, inverse-comparison
        selection_filters = ['one-shot', 'dynamic-iterative'] 

    if args.plot_variables is not None:
        plot_variables_list = args.plot_variables.split(",")
    else:
        plot_variables_list = ['real_prune_ratio']

    plot_results(
        filter_list=selection_filters, 
        plot_variables_list=plot_variables_list, 
        highlight_min=args.highlight_min,
        latex_out=args.latex_out
    )
