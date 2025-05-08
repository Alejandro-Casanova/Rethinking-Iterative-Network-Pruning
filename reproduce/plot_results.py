import argparse
import glob
import os
import matplotlib
import numpy as np
import pandas as pd
# import matplotlib
import json
from pathlib import Path
from typing import Literal
from itertools import chain
from matplotlib import colors as pycolors, pyplot as plt

variable_labels_dict: dict = {
    'acc_drop': 'ΔExactitud (%)',
    'real_prune_ratio': 'Ratio de poda real (%)',
    'prune_rate': 'Ratio de poda (%)',
    'prune_iterations': 'Iteraciones de poda',
    'runtime': 'Tiempo de ejecución (min)',
    'latency_original': 'Latencia original (ms)',
    'latency_pruned': 'Latencia pruned (ms)',
    'latency_delta': 'Latencia delta (ms)',
    'speed_up': 'Aceleración',
    'target_speed_up': 'Aceleración objetivo',
}

variables_short_names_dict: dict = {
    'acc_drop': 'Exactitud',
    'real_prune_ratio': 'Ratio de poda real',
    'prune_rate': 'Ratio de poda',
    'prune_iterations': 'Iteraciones de poda',
    'runtime': 'Tiempo de ejecución',
    'latency_original': 'Latencia original',
    'latency_pruned': 'Latencia pruned',
    'latency_delta': 'Latencia delta',
    'speed_up': 'Aceleración',
    'target_speed_up': 'Aceleración objetivo',
} 

current_dir = Path.cwd()
results_dir = os.path.normpath(os.path.join(current_dir, "reproduce/my_runs"))

pd.set_option('display.max_columns', None)

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
    output_path: str = None,
    force_speedup: bool = False,
    interactive_plot: bool = False,
    only_plot: bool = False,
    image_format: Literal['eps', 'svg'] = 'eps',
):
    if isinstance(filter_list, str):
        print("Single String passed instead of list... But don't worry, I can handle it ;)")
        filter_list = [filter_list]
    if isinstance(plot_variables_list, str):
        print("Single String passed instead of list... But don't worry, I can handle it ;)")
        plot_variables_list = [plot_variables_list]

    if output_path is None:
        raise Exception("Output path not specified!")
    else:
        # Check output path is a directory
        if not os.path.isdir(output_path):
            raise Exception("Output path is a file! Please specify a directory name instead.")
        output_path = os.path.normpath(output_path)
        # Create path if it does not exist
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

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
    # print(df)

    # Group by the 'Category' column and calculate the mean of each group
    grouped_df = df.groupby('prune_rate')

    # Calculate mean and std over different seeds
    df_mean = grouped_df.mean().reset_index()
    df_std = grouped_df.std().reset_index()
    print(df_mean)
    # print(df_std)

    if force_speedup:
        # Fill all target_speed_up NaN values in df with the mean target_speed_up for the same prune_rate
        # First iterate over rows
        for index, row in df.iterrows():
            if pd.isna(row['target_speed_up']):
                # Get the mean target_speed_up for the same prune_rate
                mean_target_speed_up = df_mean.loc[df_mean['prune_rate'] == row['prune_rate'], 'target_speed_up'].values[0]
                # Fill the NaN value with the mean
                df.at[index, 'target_speed_up'] = mean_target_speed_up
    # print(df.head(10))

    if not df_mean["seed"].eq(2.0).all(axis=0):
        raise Exception("Average of seeds should be 2, to ensure no samples were left out.")

    # Values to include in pivot table
    values_to_include = plot_variables_list

    # Pivot table with prune_rate as columns and prune_iterations as rows, averaging all columns
    columns_selector = "target_speed_up" if force_speedup else "prune_rate"
    pivot_table_mean = df.pivot_table(index='prune_iterations', columns=columns_selector, values=values_to_include, aggfunc='mean').round(2)
    pivot_table_std = df.pivot_table(index='prune_iterations', columns=columns_selector, values=values_to_include, aggfunc='std').round(2)
    print(pivot_table_mean)

    if len(values_to_include) == 1:
        x_values = pivot_table_mean[values_to_include[0]].columns
        x_label = variable_labels_dict[columns_selector]
        x_label_short = variables_short_names_dict[columns_selector]
        y_label = variable_labels_dict[values_to_include[0]]
        y_label_short = variables_short_names_dict[values_to_include[0]]
    elif len(values_to_include) == 2:
        x_values = pivot_table_mean[values_to_include[1]]
        x_label = variable_labels_dict[values_to_include[1]]
        x_label_short = variables_short_names_dict[values_to_include[1]]
        y_label = variable_labels_dict[values_to_include[0]]
        y_label_short = variables_short_names_dict[values_to_include[0]]
    else:
        raise Exception("More than 2 variables to plot. Please select only 1 or 2 variables.")
    
    filename = f"{y_label_short} vs {x_label_short}"
    output_path = os.path.join(output_path, filename)

    # Plot line chart with pivot_table_mean and add std as error bars
    
    plt.figure(figsize=(10, 6))
    cmap = matplotlib.colormaps.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, len(pivot_table_mean[values_to_include[0]].index)))

    for color, row in zip(colors, pivot_table_mean[values_to_include[0]].index):
        plt.errorbar(
            x_values.loc[row] if len(values_to_include) == 2 else x_values,
            pivot_table_mean[values_to_include[0]].loc[row],
            xerr=pivot_table_std[values_to_include[1]].loc[row] if len(values_to_include) == 2 else None,
            yerr=pivot_table_std[values_to_include[0]].loc[row],
            marker='o',
            label=f'{row}',
            capsize=5,
            color=color
        )
    plt.title(f'{y_label_short} vs {x_label_short}')
    plt.xlabel(f'{x_label}')
    plt.ylabel(y_label)
    plt.legend(title=variables_short_names_dict["prune_iterations"])
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_path + f".{image_format}", format=image_format)

    if interactive_plot:
        plt.show()

    if only_plot:
        return
    
    if len(values_to_include) > 1:
        return # If more than 1 variable to plot, stop here

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
                first_value_highlighted = ['cellcolor:[HTML]{0xF08080}; textbf:--rwrap;' if v else '' for v in is_highlighted]
                second_value_highlighted = ['cellcolor:[HTML]{0xFDFD96}; textbf:--rwrap;' if v else '' for v in is_highlighted_2]
            else:
                first_value_highlighted = ['font-weight: bold; color: red;' if v else '' for v in is_highlighted]
                second_value_highlighted = ['font-weight: bold; color: yellow;' if v else '' for v in is_highlighted_2]
            return [item1 if item1 != '' else item2 for item1, item2 in zip(first_value_highlighted, second_value_highlighted)]
        return highlight_fun
    
   
    def background_grad_fun(s: pd.Series, cmap='Blues', low=0, high=0):
        """Background gradient"""
        a = s.apply(str.split, args="±")
        a.apply(list.reverse)
        a = a.apply(list.pop)
        a = a.apply(float)
        range = a.max() - a.min()
        norm = pycolors.Normalize(a.min() - (range * low),
                            a.max() + (range * high))
        normed = norm(a.values)
        c = [pycolors.rgb2hex(x) for x in matplotlib.colormaps.get_cmap(cmap)(normed)]
        # c_text = [pycolors.rgb2hex(x) for x in matplotlib.colormaps.get_cmap('gray')(normed)]
        c_text = ["#FFFFFF" if x > 0.5 else "#000000" for x in normed]
        return [f"background-color: {bg_color}; color: {text_color};" for bg_color, text_color in zip(c, c_text)]

    styled_pivot_table = pivot_table_combined.style.apply(
        background_grad_fun,
        cmap='Blues'
    )
    # print(styled_pivot_table)
    if latex_out:

        latex_str = styled_pivot_table.to_latex(
            clines="skip-last;data",
            label="placeholder",
            caption="placeholder",
            convert_css=True,
            position_float="centering",
            multicol_align="|c|",
            hrules=True,
            position="h",
            column_format="ccccccc",
            # index=False,
            # formatters={"name": str.upper},
            # float_format="{:.1f}".format,
        )  

        # Clean up header
        for plot_var in plot_variables_list:
            latex_str = latex_str.replace(f"(\'{plot_var}\',", "")
        latex_str = latex_str.replace(")", "")
        latex_str = latex_str.replace("prune_iterations", "it.")

        # Replace header with template
        latex_str = latex_str.split("\\midrule")[1] # Split string at "\midrule"
        # Get the string to replace it with from latex_example_header.tex
        template_file_path = "reproduce/latex_example_header_prune_rate.tex" if not force_speedup else "reproduce/latex_example_header_speed_up.tex"
        with open(template_file_path, "r", encoding='utf-8') as f:
            header_str = f.read()
        latex_str = header_str + latex_str # Add header to the string
        # Insert closing brace after "\end{tabular}""
        latex_str = latex_str.replace("\\end{tabular}", "\\end{tabular}}")

        with open(output_path + '.tex', "w", encoding="utf-8") as fp:
            fp.write(latex_str)
        return
    
    # Apply the highlight_max function to each column
    # styled_pivot_table = pivot_table_combined.style.apply(
    #     highlight(highlight_min),
    #     axis=0
    # )

    # Save the styled DataFrame to an HTML file
    styled_pivot_table.to_html(output_path + ".html")

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
    parser.add_argument("-f", "--experiment-filters", type=str, default="one-shot,dynamic-iterative", help="Experiment filters to select from results directory. E.g. 'one-shot,dynamic-iterative'")
    parser.add_argument("-o", "--output-path", type=str, default=None, help="Output path for the plot and table (must be a directory)")
    parser.add_argument("-plt", "--plot-variables", type=str, default="acc_drop", help="Variables to plot (just 1 or 2). Comma separated. E.g. 'acc_drop,real_prune_ratio'")
    parser.add_argument("-ltx", "--latex-out", action="store_true", default=False, help="Output latex table instead of html")
    parser.add_argument("-hmn", "--highlight-min", action="store_true", default=False, help="Highlight min values in table (invert gradient)")
    parser.add_argument("-i", "--interactive", action="store_true", default=False, help="Activate interactive plot mode")
    parser.add_argument("-fs", "--force-speedup", action="store_true", default=False, help="Force target_speed_up to be the x axis constant (experiment 2)")
    parser.add_argument("-op", "--only-plot", action="store_true", default=False, help="Only plot the results graph, no table.")
    parser.add_argument("-if", "--image-format", type=str, default="eps", help="Image format to save the plot (default: eps)")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Activate verbose mode")

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
        latex_out=args.latex_out,
        output_path=os.path.normpath(args.output_path) if args.output_path is not None else None,
        force_speedup=args.force_speedup,
        interactive_plot=args.interactive,
        only_plot=args.only_plot,
        # image_format=args.image_format, # Not used yet
    )
