import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.express as px
import requests
import solara as sl
from requests.auth import HTTPBasicAuth

# For hitting the API
base_url = os.environ["DOMINO_KUBECOST_URL"]
assets_url = sl.reactive(f"{base_url}/assets")
allocations_url = sl.reactive(f"{base_url}/allocation")
username = os.environ["DOMINO_KUBECOST_USERNAME"]
pwd = os.environ["DOMINO_KUBECOST_PASSWORD"]
auth = sl.reactive(HTTPBasicAuth(username, pwd))

# For interacting with the different scopes
breakdown_options = ["Execution Type", "Top Projects", "User", "Organization"]
breakdown_to_param = {
    # "Execution Type": "dominodatalab_com_workload_type",
    "Top Projects": "dominodatalab_com_project_name",
    "User": "dominodatalab_com_starting_user_username",
    "Organization": "dominodatalab_com_organization_name",
}
# breakdown_choice = sl.reactive(breakdown_options[0])

# For granular aggregations
window_options = ["Last 30 days", "Last 15 days", "Last week", "Today"]
window_to_param = {
    "Last 30 days": "30d",
    "Last 15 days": "15d",
    "Last week": "lastweek",
    "Today": "today",
}
window_choice = sl.reactive(window_options[0])

# TODO: This should be replaced with real values
EXECUTION_COST_MAX = os.getenv("DOMINO_EXECUTION_COST_MAX", None)
PROJECT_MAX_SPEND = os.getenv("DOMINO_PROJECT_MAX_SPEND", 8)
ORG_MAX_SPEND = os.getenv("DOMINO_ORG_MAX_SPEND", 500)

BREAKDOWN_SPEND_MAP = {"Top Projects": PROJECT_MAX_SPEND, "Organization": ORG_MAX_SPEND}
# If the user changes the global filter by clicking on a bar in the breakdown chart
# (lefthand chart), we want to change the breakdown to something else
GLOBAL_FILTER_CHANGE_MAP = {
    "Organization": "Top Projects",
    "Top Projects": "User",
    "User": "Top Projects",
    "Execution Type": "User",
}


def get_all_organizations() -> List[str]:
    params = {
        "window": "30d",
        "aggregate": "label:dominodatalab_com_organization_name",
        "accumulate": True,
    }
    orgs_res = requests.get(allocations_url.value, params=params, auth=auth.value)
    orgs = orgs_res.json()["data"][0].keys()
    return [org for org in orgs if not org.startswith("__")]


ALL_ORGS = [""] + get_all_organizations()
filtered_label = sl.reactive("")
filtered_value = sl.reactive("")


def set_global_filters(click_data: Dict) -> None:
    filtered_label.set(click_data["seriesName"])  # The chart they clicked
    filtered_value.set(click_data["name"])  # The bar within the chart they clicked
    # breakdown_choice.set(GLOBAL_FILTER_CHANGE_MAP[breakdown_choice.value])


def clear_filters() -> None:
    filtered_label.set("")
    filtered_value.set("")


def set_filter(params: Dict) -> None:
    if filtered_value.value and filtered_label.value:
        param_label = breakdown_to_param[filtered_label.value]
        params["filter"] = f'label[{param_label}]:"{filtered_value.value}"'


def _format_datetime(dt_str: str) -> str:
    datetime_object = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
    return datetime_object.strftime("%m/%d %I:%M %p")


def get_cost_per_breakdown(breakdown_for: str) -> Dict[str, float]:
    params = {
        "window": window_to_param[window_choice.value],
        # "aggregate": f"label:{breakdown_to_param[breakdown_choice.value]}",
        "aggregate": f"label:{breakdown_for}",
        "accumulate": True,
    }
    set_filter(params)

    res = requests.get(allocations_url.value, params=params, auth=auth.value)
    data = res.json()["data"][0]
    return {
        key: round(data[key]["totalCost"], 2)
        for key in data
        if not key.startswith("__")
    }


def get_overall_cost() -> Dict[str, float]:
    params = {
        "window": window_to_param[window_choice.value],
        "aggregate": "category",
        "accumulate": True,
    }
    set_filter(params)

    res = requests.get(assets_url.value, params=params, auth=auth.value)

    data = res.json()["data"][0]
    data.keys()

    return {key: round(data[key]["totalCost"], 2) for key in data}


def _to_date(date_string: str) -> str:
    """Converts minute-level date string to day level

    ex:
       _to_date(2023-04-28T15:05:00Z) -> 2023-04-28
    """
    dt = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%d")


def get_daily_cost() -> pd.DataFrame:
    window = window_to_param[window_choice.value]
    params = {
        "window": window,
        # "aggregate": "category",
    }
    set_filter(params)
    # TODO: Use the assets route to join to the aloc route, because the aloc route
    #  doesn't support filters...

    # res = requests.get(assets_url.value, params=params, auth=auth.value)
    res = requests.get(allocations_url.value, params=params, auth=auth.value)
    data = res.json()["data"]
    # May not have all historical days
    alocs = [day for day in data if day]
    # Route returns data non-cumulatively. We make it cumulative by summing over the
    # returned windows (could be days, hours, weeks etc)
    daily_costs = defaultdict(dict)

    cpu_costs = ["cpuCost", "cpuCostAdjustment", "gpuCost", "gpuCostAdjustment"]
    storage_costs = ["pvCost", "pvCostAdjustment", "ramCost", "ramCostAdjustment"]

    costs = {"Compute": cpu_costs, "Storage": storage_costs}
    # Gets the overall cost per day
    for aloc in alocs:
        for key, values in aloc.items():
            start = values["start"]
            for cost_type, cost_keys in costs.items():
                if cost_type not in daily_costs[start]:
                    daily_costs[start][cost_type] = 0.0
                daily_costs[start][cost_type] += round(
                    sum(values[key] for key in cost_keys), 2
                )
    # Cumulative sum over the daily costs
    df = pd.DataFrame(daily_costs).T.sort_index()
    df["Compute"] = df["Compute"].cumsum()
    df["Storage"] = df["Storage"].cumsum()
    # Unless we are looking at today granularity, rollup values to the day level
    # (they are returned at the 5min level)
    if window != "today":
        df.index = df.index.map(_to_date)
        df = df.groupby(level=0).max()
    return df


def get_execution_cost_table() -> pd.DataFrame:
    # TODO: Break down further by execution id
    # label:dominodatalab_com_execution_id
    params = {
        "window": window_to_param[window_choice.value],
        "aggregate": (
            "label:dominodatalab_com_workload_id,"  # TODO: workload_id or execution_id
            "label:dominodatalab_com_workload_type,"
            "label:dominodatalab_com_starting_user_username,"
            "label:dominodatalab_com_project_id"
        ),
        "accumulate": True,
    }
    set_filter(params)

    res = requests.get(allocations_url.value, params=params, auth=auth.value)
    aloc_data = res.json()["data"][0]

    exec_data = []

    cpu_cost_key = ["cpuCost", "gpuCost"]
    gpu_cost_key = ["cpuCostAdjustment", "gpuCostAdjustment"]
    storage_cost_keys = ["pvCost", "ramCost", "pvCostAdjustment", "ramCostAdjustment"]

    keys = list(aloc_data.keys())
    keys = [key for key in keys if not key.startswith("__")]
    for key in keys:
        workload_id, workload_type, uname, project_id = key.split("/")
        # workload_id="foobar"
        # workload_type, uname, project_id = key.split("/")
        key_data = aloc_data[key]
        cpu_cost = round(sum([key_data[k] for k in cpu_cost_key]), 2)
        gpu_cost = round(sum([key_data[k] for k in gpu_cost_key]), 2)
        compute_cost = round(cpu_cost + gpu_cost, 2)
        storage_cost = round(sum([key_data[k] for k in storage_cost_keys]), 2)
        waste = f"{((1-key_data['totalEfficiency'])*100)}%"
        exec_data.append(
            {
                "TYPE": workload_type,
                "USER": uname,
                "START": key_data["start"],
                "END": key_data["end"],
                "CPU_COST": f"${cpu_cost}",
                "GPU_COST": f"${gpu_cost}",
                "COMPUTE_COST": f"${compute_cost}",
                "COMPUTE_WASTE": waste,
                "STORAGE_COST": f"${storage_cost}",
                "WORKLOAD_ID": workload_id,
                "PROJECT_ID": project_id,
            }
        )
    df = pd.DataFrame(exec_data)
    for col in ["START", "END"]:
        if col in df.columns:
            df[col] = df[col].apply(_format_datetime)
    return df


@sl.component()
def Executions() -> None:
    df = get_execution_cost_table()
    sl.DataFrame(df)


@sl.component()
def DailyCostBreakdown() -> None:
    df = get_daily_cost()
    fig = px.bar(
        df,
        labels={
            "index": "Date",
            "value": "Cost ($)",
        },
        title="Overall Cost (Cumulative)",
        color_discrete_sequence=px.colors.qualitative.D3,
    )
    # Horizontal line indicating the "max" spend by the company
    exec_cost = EXECUTION_COST_MAX or (
        (df["Compute"].max() + df["Storage"].max()) * 0.8
    )
    fig.add_shape(
        type="line",
        x0=df.index.min(),
        x1=df.index.max(),
        y0=exec_cost,
        y1=exec_cost,
        line=dict(
            color="red",
            width=3,
            dash="dash",
        ),
    )
    sl.FigurePlotly(fig)


@sl.component()
def SingleCost(name: str, cost: float) -> None:
    with sl.Column():
        cost_ = f"## ${cost}" if name == "Total" else f"#### ${cost}"
        sl.Markdown(cost_)
        name_ = f"### {name}" if name == "Total" else name
        sl.Markdown(name_)


@sl.component()
def TopLevelCosts() -> None:
    costs = get_overall_cost()
    # with sl.Columns([2, 1, 1, 1]):
    with sl.Row(justify="space-around"):
        # with sl.Card():
        SingleCost("Total", round(sum(list(costs.values())), 2))
        for name, cost in costs.items():
            # with sl.Card():
            SingleCost(name, cost)


@sl.component()
def OverallCosts() -> None:
    with sl.Column():
        with sl.Card():
            TopLevelCosts()
        with sl.Card():
            DailyCostBreakdown()


@sl.component()
def CostBreakdown() -> None:
    # with sl.Row(gap="1px", justify="space-around"):
    with sl.Card("Cost Usage"):
        with sl.Columns([1, 1, 1]):
            for name, breakdown_choice_ in breakdown_to_param.items():
                # with sl.Card(f"Cost Usage - {name}", margin=10):
                # sl.Select(label="", value=breakdown_choice, values=breakdown_options)
                costs = get_cost_per_breakdown(breakdown_choice_)
                cost_values = list(costs.values())
                max_spend = BREAKDOWN_SPEND_MAP.get(name, 1e1000)
                overflow_values = [v - max_spend for v in cost_values]
                overflow_values = [max(v, 0) for v in overflow_values]
                option = {
                    "title": {"text": name},
                    "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                    "legend": {},
                    "grid": {
                        "left": "3%",
                        "right": "4%",
                        "bottom": "3%",
                        "containLabel": True,
                    },
                    "xAxis": {"type": "value", "boundaryGap": [0, 0.01]},
                    "yAxis": {"type": "category", "data": list(costs.keys())},
                    "series": [
                        {
                            "type": "bar",
                            "data": cost_values,
                            "stack": "y",
                            "name": name,
                        },
                        {
                            "type": "bar",
                            "data": overflow_values,
                            "stack": "y",
                            "color": "red",
                            "name": name,
                        },
                    ],
                }
                sl.FigureEcharts(option, on_click=set_global_filters)


@sl.component()
def Page() -> None:
    sl.Title("Cost Analysis")
    sl.Markdown(
        "# Domino Cost Management Report",
        style="display: inline-block; margin: 0 auto;",
    )
    with sl.Column(style="width:15%"):
        with sl.Row():
            sl.Select(label="Window", value=window_choice, values=window_options)
            if filtered_label.value and filtered_value.value:
                sl.Button(
                    f"{filtered_label.value}: {filtered_value.value} x",
                    on_click=clear_filters,
                )
            # sl.Select(label="Organization", value=filtered_value, values=ALL_ORGS)
    # with sl.Columns([2, 3]):
    #     CostBreakdown()
    #     OverallCosts()
    with sl.Column():
        OverallCosts()
        CostBreakdown()
        if filtered_value.value:
            with sl.Card("Executions"):
                Executions()
