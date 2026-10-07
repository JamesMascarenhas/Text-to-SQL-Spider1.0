"""
task 1 figures built from the summary csv files so they always match the latest scoring
runs on the mac after summarizeRuns.py

figures
  size curve      ex and em against parameter count for the coder family and qwen3 with and without thinking
                  repeat seeds get averaged with a bar from the lowest to the highest seed
                  the em panel next to the ex panel shows how flat em stays while ex climbs
  cost tradeoff   ex against mean output tokens per question for every configuration
                  including the 4 bit model and self consistency once those exist

writes png files into outputs/figures

how to run
    python src/plotResults.py
"""

import math

import matplotlib

# draws straight to files so it works without a screen
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from spiderUtils import outDir  # noqa: E402


summaryDir = outDir / "summary"
figuresDir = outDir / "figures"

# total parameters in billions
# check every one against its hugging face model card and cite the cards in the report
paramsBillions = {
    "Qwen2.5-Coder-0.5B-Instruct": 0.49,
    "Qwen2.5-Coder-1.5B-Instruct": 1.54,
    "Qwen2.5-Coder-3B-Instruct": 3.09,
    "Qwen2.5-Coder-7B-Instruct": 7.61,
    "Qwen2.5-Coder-7B-Instruct-AWQ": 7.61,
    "Qwen3-1.7B": 1.7,
    "Qwen3-4B": 4.0,
    "Qwen3-8B": 8.2,
}

seriesStyles = {
    "Qwen2.5-Coder": {"colour": "#4C72B0", "marker": "o"},
    "Qwen3 no thinking": {"colour": "#DD8452", "marker": "s"},
    "Qwen3 thinking": {"colour": "#55A868", "marker": "^"},
}


# loading

def loadRunTable() -> pd.DataFrame:
    accuracy = pd.read_csv(summaryDir / "accuracy.csv")
    efficiency = pd.read_csv(summaryDir / "efficiency.csv")
    settings = pd.read_csv(summaryDir / "settings.csv")

    runTable = accuracy.merge(efficiency, on = "run").merge(settings, on = "run")

    modelNames = []
    for modelId in runTable["model"]:
        modelNames.append(modelId.split("/")[-1])
    runTable["modelName"] = modelNames

    # voted runs are named by their strategy in the summary labels
    isVoted = []
    for runLabel in runTable["run"]:
        isVoted.append("self consistency" in runLabel)
    runTable["isVoted"] = isVoted

    # quantization is written as json in the csv so none shows up with quotes around it
    isQuantized = []
    for quantization in runTable["quantization"]:
        isQuantized.append(str(quantization).strip('"') != "none")
    runTable["isQuantized"] = isQuantized

    return runTable


def seriesName(tableRow) -> str:
    if tableRow["modelName"].startswith("Qwen2.5-Coder"):
        return "Qwen2.5-Coder"
    if tableRow["thinking"] == "on":
        return "Qwen3 thinking"
    return "Qwen3 no thinking"


def shortLabel(tableRow) -> str:
    # compact point labels for the cost figure
    label = tableRow["modelName"].replace("Qwen2.5-Coder-", "Coder ").replace("-Instruct", "")
    label = label.replace("-AWQ", " AWQ").replace("Qwen3-", "Qwen3 ")

    if tableRow["thinking"] == "on":
        label = label + " think"

    if tableRow["isVoted"]:
        votedPart = tableRow["run"].split("self consistency")[-1].strip()
        label = label + " SC " + votedPart

    return label


def groupSeeds(runTable: pd.DataFrame) -> pd.DataFrame:
    """
    one point per configuration so repeat seeds dont crowd the plot
    the mean is the point and the min and max become the error bar
    """
    groupRows = []

    groupKeys = ["modelName", "thinking", "isVoted", "isQuantized", "run_group"]
    runTable = runTable.copy()

    # voted runs stay separate from each other since n differs
    runGroups = []
    for _, tableRow in runTable.iterrows():
        if tableRow["isVoted"]:
            runGroups.append(tableRow["run"])
        else:
            runGroups.append("single")
    runTable["run_group"] = runGroups

    for keyValues, groupTable in runTable.groupby(groupKeys):
        firstRow = groupTable.iloc[0]

        groupRows.append({
            "modelName": firstRow["modelName"],
            "thinking": firstRow["thinking"],
            "isVoted": firstRow["isVoted"],
            "isQuantized": firstRow["isQuantized"],
            "run": firstRow["run"],
            "nSeeds": len(groupTable),
            "exMean": groupTable["ex_all"].mean(),
            "exMin": groupTable["ex_all"].min(),
            "exMax": groupTable["ex_all"].max(),
            "emMean": groupTable["em_all"].mean(),
            "emMin": groupTable["em_all"].min(),
            "emMax": groupTable["em_all"].max(),
            "tokensMean": groupTable["output_tokens_mean"].mean(),
        })

    return pd.DataFrame(groupRows)


# figures

def plotSizeCurves(groups: pd.DataFrame):
    # only plain runs belong on a size curve so the 4 bit model and voted runs stay off it
    plainGroups = groups[(groups["isVoted"] == False) & (groups["isQuantized"] == False)].copy()  # noqa: E712

    paramValues = []
    seriesValues = []
    for _, groupRow in plainGroups.iterrows():
        paramValues.append(paramsBillions.get(groupRow["modelName"], float("nan")))
        seriesValues.append(seriesName(groupRow))
    plainGroups["params"] = paramValues
    plainGroups["series"] = seriesValues

    figure, axes = plt.subplots(1, 2, figsize = (10, 4), sharey = True)

    for axis, metric, title in [(axes[0], "ex", "Execution accuracy (EX)"), (axes[1], "em", "Exact set match (EM)")]:
        for series, style in seriesStyles.items():
            seriesGroups = plainGroups[plainGroups["series"] == series].sort_values("params")
            if len(seriesGroups) == 0:
                continue

            means = seriesGroups[f"{metric}Mean"]
            lowerBars = means - seriesGroups[f"{metric}Min"]
            upperBars = seriesGroups[f"{metric}Max"] - means

            axis.errorbar(
                seriesGroups["params"], means,
                yerr = [lowerBars, upperBars],
                color = style["colour"], marker = style["marker"], capsize = 3, label = series,
            )

        axis.set_xscale("log")

        # plain size ticks read better than powers of ten
        sizeTicks = [0.5, 1, 2, 4, 8]
        axis.set_xticks(sizeTicks)
        sizeTickLabels = []
        for tick in sizeTicks:
            sizeTickLabels.append(f"{tick:g}")
        axis.set_xticklabels(sizeTickLabels)
        axis.minorticks_off()

        axis.set_xlabel("parameters in billions (log scale)")
        axis.set_title(title, fontsize = 11)
        axis.grid(alpha = 0.3)

    axes[0].set_ylabel("accuracy on Spider dev")
    axes[0].legend(fontsize = 8)
    figure.suptitle("Accuracy by model size (thinking points are seed means with min to max bars)", fontsize = 11)
    figure.tight_layout()
    figure.savefig(figuresDir / "size_curves.png", dpi = 200)
    plt.close(figure)


def plotCostTradeoff(groups: pd.DataFrame):
    figure, axis = plt.subplots(figsize = (7, 5))

    # a model and its 4 bit twin land almost on top of each other so their labels go above and below
    # whichever of the two scored higher gets the label above
    twinOffsets = {}
    for _, groupRow in groups.iterrows():
        if not groupRow["isQuantized"]:
            continue
        baseName = groupRow["modelName"].replace("-AWQ", "")
        baseRows = groups[(groups["modelName"] == baseName) & (groups["isVoted"] == False)]  # noqa: E712
        if len(baseRows) == 0:
            continue
        baseEx = baseRows.iloc[0]["exMean"]
        if groupRow["exMean"] >= baseEx:
            twinOffsets[groupRow["modelName"]] = (5, 6)
            twinOffsets[baseName] = (5, -12)
        else:
            twinOffsets[groupRow["modelName"]] = (5, -12)
            twinOffsets[baseName] = (5, 6)

    # points that sit close together take turns putting their label above and below
    # so pairs like 4b and 8b thinking or the two self consistency runs dont print on top of each other
    crowdedOffsets = {}
    placedPoints = []
    for _, groupRow in groups.sort_values("tokensMean").iterrows():
        xPosition = math.log10(groupRow["tokensMean"])
        yPosition = groupRow["exMean"]

        neighbourCount = 0
        for placedX, placedY in placedPoints:
            if abs(placedX - xPosition) < 0.25 and abs(placedY - yPosition) < 0.02:
                neighbourCount = neighbourCount + 1

        if neighbourCount % 2 == 1:
            crowdedOffsets[groupRow["run"]] = (5, -12)

        placedPoints.append((xPosition, yPosition))

    for _, groupRow in groups.iterrows():
        style = seriesStyles[seriesName(groupRow)]

        # voted and 4 bit runs get hollow markers so they read as variants of a base model
        if groupRow["isVoted"] or groupRow["isQuantized"]:
            faceColour = "none"
        else:
            faceColour = style["colour"]

        axis.scatter(
            groupRow["tokensMean"], groupRow["exMean"],
            marker = style["marker"], s = 60, edgecolors = style["colour"], facecolors = faceColour,
        )
        if groupRow["modelName"] in twinOffsets and not groupRow["isVoted"]:
            labelOffset = twinOffsets[groupRow["modelName"]]
        elif groupRow["run"] in crowdedOffsets:
            labelOffset = crowdedOffsets[groupRow["run"]]
        else:
            labelOffset = (5, 4)

        axis.annotate(shortLabel(groupRow), (groupRow["tokensMean"], groupRow["exMean"]),
                      textcoords = "offset points", xytext = labelOffset, fontsize = 7)

    # legend entries drawn once per series
    for series, style in seriesStyles.items():
        axis.scatter([], [], marker = style["marker"], color = style["colour"], label = series)
    axis.scatter([], [], marker = "o", facecolors = "none", edgecolors = "grey", label = "4 bit or self consistency")

    axis.set_xscale("log")

    # room on the right so the longest labels arent cut off
    axis.set_xlim(groups["tokensMean"].min() * 0.7, groups["tokensMean"].max() * 3)

    tokenTicks = [30, 100, 300, 1000, 3000, 10000]
    visibleTicks = []
    for tick in tokenTicks:
        if groups["tokensMean"].min() * 0.7 <= tick <= groups["tokensMean"].max() * 3:
            visibleTicks.append(tick)
    axis.set_xticks(visibleTicks)
    tokenTickLabels = []
    for tick in visibleTicks:
        tokenTickLabels.append(f"{tick:g}")
    axis.set_xticklabels(tokenTickLabels)
    axis.minorticks_off()

    axis.set_xlabel("mean output tokens per question (log scale)")
    axis.set_ylabel("execution accuracy (EX)")
    axis.set_title("Accuracy against generation cost", fontsize = 11)
    axis.grid(alpha = 0.3)
    axis.legend(fontsize = 8, loc = "lower right")
    figure.tight_layout()
    figure.savefig(figuresDir / "cost_tradeoff.png", dpi = 200)
    plt.close(figure)


def main():
    figuresDir.mkdir(parents = True, exist_ok = True)

    runTable = loadRunTable()
    groups = groupSeeds(runTable)

    plotSizeCurves(groups)
    plotCostTradeoff(groups)

    # the numbers behind both figures so the report text can quote them exactly
    groups.to_csv(figuresDir / "figure_points.csv", index = False)

    print(f"wrote size_curves.png and cost_tradeoff.png and figure_points.csv into {figuresDir}")


if __name__ == "__main__":
    main()