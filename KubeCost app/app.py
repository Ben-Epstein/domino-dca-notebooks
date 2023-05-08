import os
from datetime import datetime
from typing import Dict

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
breakdown_options = ["Execution Type", "Top Projects", "User"]
breakdown_to_param = {
    "Execution Type": "dominodatalab_com_workload_type",
    "Top Projects": "dominodatalab_com_project_name",
    "User": "dominodatalab_com_starting_user_username",
}
breakdown_choice = sl.reactive(breakdown_options[0])

# For granular aggregations
window_options = ["Last 30 days", "Last 15 days", "Last week", "Today"]
window_to_param = {
    "Last 30 days": "30d",
    "Last 15 days": "15d",
    "Last week": "lastweek",
    "Today": "today",
}
window_choice = sl.reactive(window_options[0])


def _format_datetime(dt_str: str) -> str:
    datetime_object = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
    return datetime_object.strftime("%m/%d %I:%M %p")


def get_cost_per_breakdown() -> Dict[str, float]:
    params = {
        "window": window_to_param[window_choice.value],
        "aggregate": f"label:{breakdown_to_param[breakdown_choice.value]}",
        "accumulate": True,
    }

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

    res = requests.get(assets_url.value, params=params, auth=auth.value)

    data = res.json()["data"][0]
    data.keys()

    return {key: round(data[key]["totalCost"], 2) for key in data}


def get_daily_cost() -> pd.DataFrame:
    params = {"window": window_to_param[window_choice.value], "aggregate": "category"}

    res = requests.get(assets_url.value, params=params, auth=auth.value)
    data = res.json()["data"]
    # May not have all historical days
    data = [day for day in data if day]

    day_costs = {
        day["Network"]["start"]: {key: round(day[key]["totalCost"], 2) for key in day}
        for day in data
    }
    return pd.DataFrame(day_costs).T


def get_execution_cost_table() -> pd.DataFrame:
    params = {
        "window": window_to_param[window_choice.value],
        "aggregate": "label:dominodatalab_com_workload_type,label:dominodatalab_com_starting_user_username,label:dominodatalab_com_project_id",
        # TODO: How do I get all the project IDs (and names)
        #     "filter": 'label[dominodatalab_com_project_id]:"645277142a8f2d12e2e13fca"',
        "accumulate": True,
    }

    res = requests.get(allocations_url.value, params=params, auth=auth.value)
    aloc_data = res.json()["data"][0]

    exec_data = []

    cpu_cost_key = ["cpuCost", "gpuCost"]
    gpu_cost_key = ["cpuCostAdjustment", "gpuCostAdjustment"]
    storage_cost_keys = ["pvCost", "ramCost", "pvCostAdjustment", "ramCostAdjustment"]

    keys = list(aloc_data.keys())
    keys = [key for key in keys if not key.startswith("__")]
    for key in keys:
        workload_type, username, project_id = key.split("/")
        key_data = aloc_data[key]
        cpu_cost = round(sum([key_data[k] for k in cpu_cost_key]), 2)
        gpu_cost = round(sum([key_data[k] for k in gpu_cost_key]), 2)
        compute_cost = round(cpu_cost + gpu_cost, 2)
        storage_cost = round(sum([key_data[k] for k in storage_cost_keys]), 2)
        waste = f"{((1-key_data['totalEfficiency'])*100)}%"
        exec_data.append(
            {
                "TYPE": workload_type,
                "USER": username,
                "START": key_data["start"],
                "END": key_data["end"],
                "CPU_COST": f"${cpu_cost}",
                "GPU_COST": f"${gpu_cost}",
                "COMPUTE_COST": f"${compute_cost}",
                "COMPUTE_WASTE": waste,
                "STORAGE_COST": f"${storage_cost}",
                "ID": project_id,
            }
        )
    df = pd.DataFrame(exec_data)
    df["START"] = df["START"].apply(_format_datetime)
    df["END"] = df["END"].apply(_format_datetime)
    return df


@sl.component()
def Executions() -> None:
    # sl.Markdown("## Executions")
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
        color_discrete_sequence=px.colors.qualitative.Set3,
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
        with sl.Card("Executions"):
            Executions()


@sl.component()
def CostBreakdown() -> None:
    with sl.Card("Cost Usage - Breakdown"):
        sl.Select(label="", value=breakdown_choice, values=breakdown_options)
        costs = get_cost_per_breakdown()
        option = {
            # "title": {
            #     "text": 'Executions'
            # },
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {"type": "value", "boundaryGap": [0, 0.01]},
            "yAxis": {"type": "category", "data": list(costs.keys())},
            "series": [
                {"type": "bar", "data": list(costs.values())},
            ],
        }
        sl.FigureEcharts(option)


@sl.component()
def Page() -> None:
    sl.Title("Cost Analysis")
    sl.Markdown(
        "# Domino Cost Management Report",
        style="display: inline-block; margin: 0 auto;",
    )
    with sl.Column(style="width:15%"):
        sl.Select(label="Window", value=window_choice, values=window_options)
    with sl.Columns([2, 3]):
        CostBreakdown()
        OverallCosts()
