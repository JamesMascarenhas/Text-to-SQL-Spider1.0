"""
task 1 figures built from the summary csv files so they always match the latest scoring
runs on the mac after summarizeRuns.py

figures
  size curve      ex and em against parameter count for the coder family and qwen3 with and without thinking
                  repeat seeds get averaged with a bar from the lowest to the highest seed
                  the em panel next to the ex panel shows how flat em stays while ex climbs
  cost tradeoff   ex against mean output tokens per question for every configuration
                  including the 4 bit model and self consistency once those exist
  by difficulty   ex at each spider difficulty for the coder sizes with the 4 bit model
                  and for every qwen3 size with thinking off and on
  thinking return what thinking adds at each size and difficulty with a 95 percent interval
                  and how many ex points each thousand extra tokens buys
  em against ex   overall em against overall ex for every configuration to show the two metrics rank models differently
  compute ladder  task 3 view of qwen3 4b by difficulty going from no thinking to thinking to voting
                  with the best of 5 upper bound marked over each group

writes png files into outputs/figures
needs selfConsistency.py to have run first for the compute ladder

how to run
    python src/plotResults.py
"""

import math

import matplotlib

# draws straight to files so it works without a screen
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from spiderUtils import difficultyLevels, outDir  # noqa: E402


summaryDir = outDir / "summary"
figuresDir = outDir / "figures"
selfConsistencyDir = outDir / "selfConsistency"

difficultyNames = {"easy": "easy", "medium": "medium", "hard": "hard", "extra": "extra hard", "all": "all"}

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

# variants keep their base models shape so the model type still reads
# but get their own colour so 4 bit and voting dont blur together
variantStyles = {
    "Qwen2.5-Coder 4 bit AWQ": {"colour": "#B8860B", "marker": "o"},
    "Qwen3-4B thinking self consistency": {"colour": "#8172B3", "marker": "^"},
}


def pointStyle(tableRow) -> dict:
    if tableRow["isQuantized"]:
        return variantStyles["Qwen2.5-Coder 4 bit AWQ"]
    if tableRow["isVoted"]:
        return variantStyles["Qwen3-4B thinking self consistency"]
    return seriesStyles[seriesName(tableRow)]


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
    # colour and shape already say which family and mode a point is so the label only needs the size
    # the voted runs share one size so they get their sample count instead
    if tableRow["isVoted"]:
        votedPart = tableRow["run"].split("self consistency")[-1].strip()
        return "SC " + votedPart

    label = tableRow["modelName"].replace("Qwen2.5-Coder-", "").replace("-Instruct", "")
    label = label.replace("-AWQ", " AWQ").replace("Qwen3-", "")
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

        # by difficulty means for the difficulty figure with seeds averaged the same way
        for level in difficultyLevels:
            groupRows[-1][f"ex_{level}Mean"] = groupTable[f"ex_{level}"].mean()

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
    figure.suptitle("Accuracy by model size", fontsize = 11)
    figure.tight_layout()
    figure.savefig(figuresDir / "size_curves.png", dpi = 200)
    plt.close(figure)


contextGrey = "#C4C4C4"


def plotCostTradeoff(groups: pd.DataFrame, fileName: str, includeVoted: bool, focusKeys):
    """
    ex against mean output tokens for every configuration
    task 1 leaves the voted runs out since self consistency only gets introduced in task 3
    task 3 adds them back and passes focusKeys so only the points its comparisons use keep their colour
    and everything else turns grey as context
    focusKeys is None or a set of legend series and point label pairs
    """
    if not includeVoted:
        groups = groups[groups["isVoted"] == False]  # noqa: E712

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
        for placedX, placedY, _ in placedPoints:
            if abs(placedX - xPosition) < 0.25 and abs(placedY - yPosition) < 0.02:
                neighbourCount = neighbourCount + 1

        if neighbourCount % 2 == 1:
            crowdedOffsets[groupRow["run"]] = (5, -12)

        placedPoints.append((xPosition, yPosition, groupRow["run"]))

    # the left point of a crowded pair puts its label on its left so it doesnt run into its neighbour
    leftLabelled = set()
    for placedX, placedY, placedRun in placedPoints:
        if placedRun in crowdedOffsets:
            continue
        for otherX, otherY, otherRun in placedPoints:
            if otherX > placedX and otherX - placedX < 0.25 and abs(otherY - placedY) < 0.02:
                leftLabelled.add(placedRun)

    for _, groupRow in groups.iterrows():
        style = pointStyle(groupRow)

        pointKey = (legendSeriesFor(groupRow), shortLabel(groupRow))
        if focusKeys is None or pointKey in focusKeys:
            pointColour = style["colour"]
            textColour = "black"
        else:
            pointColour = contextGrey
            textColour = "#9A9A9A"

        axis.scatter(
            groupRow["tokensMean"], groupRow["exMean"],
            marker = style["marker"], s = 60, color = pointColour,
        )
        if groupRow["modelName"] in twinOffsets and not groupRow["isVoted"]:
            labelOffset = twinOffsets[groupRow["modelName"]]
        elif groupRow["run"] in crowdedOffsets:
            labelOffset = crowdedOffsets[groupRow["run"]]
        elif groupRow["run"] in leftLabelled:
            labelOffset = (-5, 4)
        else:
            labelOffset = (5, 4)

        if labelOffset[0] < 0:
            alignment = "right"
        else:
            alignment = "left"

        axis.annotate(shortLabel(groupRow), (groupRow["tokensMean"], groupRow["exMean"]),
                      textcoords = "offset points", xytext = labelOffset, fontsize = 7, ha = alignment,
                      color = textColour)

    # legend entries drawn once per series
    # each variant sits right under its base model in the legend
    legendOrder = [
        ("Qwen2.5-Coder", seriesStyles["Qwen2.5-Coder"]),
        ("Qwen2.5-Coder 4 bit AWQ", variantStyles["Qwen2.5-Coder 4 bit AWQ"]),
        ("Qwen3 no thinking", seriesStyles["Qwen3 no thinking"]),
        ("Qwen3 thinking", seriesStyles["Qwen3 thinking"]),
        ("Qwen3-4B thinking self consistency", variantStyles["Qwen3-4B thinking self consistency"]),
    ]
    focusSeries = set()
    if focusKeys is not None:
        for seriesLabel, _ in focusKeys:
            focusSeries.add(seriesLabel)

    for legendLabel, style in legendOrder:
        if legendLabel == "Qwen3-4B thinking self consistency" and not includeVoted:
            continue
        if focusKeys is not None and legendLabel not in focusSeries:
            continue
        axis.scatter([], [], marker = style["marker"], color = style["colour"], label = legendLabel)

    if focusKeys is not None:
        axis.scatter([], [], marker = "o", color = contextGrey, label = "other Task 1 configurations")

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
    figure.savefig(figuresDir / fileName, dpi = 200)
    plt.close(figure)


def drawGroupedBars(axis, groupLabels: list, barSpecs: list, barWidth: float):
    """
    one cluster of bars per group label with the bars side by side inside each cluster
    each bar spec is a legend label then a colour then one value per group label
    """
    nBars = len(barSpecs)

    for barPosition, (barLabel, barColour, barValues) in enumerate(barSpecs):
        # centres the cluster on the group tick
        offset = (barPosition - (nBars - 1) / 2) * barWidth

        xPositions = []
        for groupPosition in range(len(groupLabels)):
            xPositions.append(groupPosition + offset)

        axis.bar(xPositions, barValues, width = barWidth, color = barColour, label = barLabel)

    axis.set_xticks(range(len(groupLabels)))
    axis.set_xticklabels(groupLabels)


def findGroup(groups: pd.DataFrame, modelName: str, thinking: str):
    # plain run groups only so the 4 bit twin and voted runs never get picked by accident
    matches = groups[
        (groups["modelName"] == modelName)
        & (groups["thinking"] == thinking)
        & (groups["isVoted"] == False)  # noqa: E712
    ]
    return matches.iloc[0]


def difficultyValues(groupRow) -> list:
    values = []
    for level in difficultyLevels:
        values.append(groupRow[f"ex_{level}Mean"])
    return values


def plotByDifficulty(groups: pd.DataFrame):
    groupLabels = []
    for level in difficultyLevels:
        groupLabels.append(difficultyNames[level])

    # darker shade means a bigger model so size reads the same way in both panels
    coderSpecs = [
        ("0.5B", "#B7CDE8", "Qwen2.5-Coder-0.5B-Instruct"),
        ("1.5B", "#86A9D6", "Qwen2.5-Coder-1.5B-Instruct"),
        ("3B", "#4C72B0", "Qwen2.5-Coder-3B-Instruct"),
        ("7B", "#24406E", "Qwen2.5-Coder-7B-Instruct"),
        ("7B AWQ", "#B8860B", "Qwen2.5-Coder-7B-Instruct-AWQ"),
    ]
    coderBars = []
    for barLabel, barColour, modelName in coderSpecs:
        coderBars.append((barLabel, barColour, difficultyValues(findGroup(groups, modelName, "none"))))

    # each size gets its off and on bars next to each other so the thinking jump is easy to read
    qwenSpecs = [
        ("1.7B no thinking", "#F2C2A0", "Qwen3-1.7B", "off"),
        ("1.7B thinking", "#A8D5B2", "Qwen3-1.7B", "on"),
        ("4B no thinking", "#E59866", "Qwen3-4B", "off"),
        ("4B thinking", "#55A868", "Qwen3-4B", "on"),
        ("8B no thinking", "#C0612B", "Qwen3-8B", "off"),
        ("8B thinking", "#2E6B3C", "Qwen3-8B", "on"),
    ]
    qwenBars = []
    for barLabel, barColour, modelName, thinking in qwenSpecs:
        qwenBars.append((barLabel, barColour, difficultyValues(findGroup(groups, modelName, thinking))))

    figure, axes = plt.subplots(1, 2, figsize = (11, 4.6), sharey = True)

    drawGroupedBars(axes[0], groupLabels, coderBars, barWidth = 0.16)
    axes[0].set_title("Qwen2.5-Coder by size with the 4 bit model", fontsize = 10)

    drawGroupedBars(axes[1], groupLabels, qwenBars, barWidth = 0.13)
    axes[1].set_title("Qwen3 by size with thinking off and on", fontsize = 10)

    for axis, nColumns in [(axes[0], 5), (axes[1], 3)]:
        axis.set_ylim(0, 1)
        axis.set_xlabel("Spider difficulty")
        axis.grid(axis = "y", alpha = 0.3)
        axis.set_axisbelow(True)
        # legend under the bars since the easy bars reach almost to the top
        axis.legend(fontsize = 8, ncol = nColumns, loc = "upper center", bbox_to_anchor = (0.5, -0.17), frameon = False)

    axes[0].set_ylabel("execution accuracy (EX)")
    figure.suptitle("Execution accuracy by difficulty", fontsize = 11)
    figure.tight_layout()
    figure.savefig(figuresDir / "accuracy_by_difficulty.png", dpi = 200)
    plt.close(figure)


def plotThinkingReturn():
    thinkingTable = pd.read_csv(summaryDir / "thinking.csv")

    levels = difficultyLevels + ["all"]
    groupLabels = []
    for level in levels:
        groupLabels.append(difficultyNames[level])

    sizeColours = [("Qwen3-1.7B", "1.7B", "#A8D5B2"), ("Qwen3-4B", "4B", "#55A868"), ("Qwen3-8B", "8B", "#2E6B3C")]
    barWidth = 0.24

    figure, axes = plt.subplots(1, 2, figsize = (11, 4.2))

    panels = [
        (axes[0], "gain", "gain_ci_low", "gain_ci_high", 100, "EX gain from thinking (points)", "What thinking adds"),
        (axes[1], "points_per_1k_tokens", "points_per_1k_ci_low", "points_per_1k_ci_high", 1,
         "EX points per 1,000 extra output tokens", "What each extra token buys"),
    ]

    for axis, valueColumn, lowColumn, highColumn, scale, yLabel, title in panels:
        for barPosition, (modelName, sizeLabel, barColour) in enumerate(sizeColours):
            offset = (barPosition - 1) * barWidth

            xPositions = []
            values = []
            lowerBars = []
            upperBars = []
            for groupPosition, level in enumerate(levels):
                tableRow = thinkingTable[(thinkingTable["model"] == modelName) & (thinkingTable["difficulty"] == level)].iloc[0]
                value = tableRow[valueColumn] * scale
                xPositions.append(groupPosition + offset)
                values.append(value)
                lowerBars.append(value - tableRow[lowColumn] * scale)
                upperBars.append(tableRow[highColumn] * scale - value)

            axis.bar(xPositions, values, width = barWidth, color = barColour, label = sizeLabel,
                     yerr = [lowerBars, upperBars], capsize = 2, error_kw = {"elinewidth": 0.8})

        # a thin line at zero so intervals that cross it stand out
        axis.axhline(0, color = "black", linewidth = 0.8)
        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels(groupLabels)
        axis.set_xlabel("Spider difficulty")
        axis.set_ylabel(yLabel)
        axis.set_title(title, fontsize = 10)
        axis.grid(axis = "y", alpha = 0.3)
        axis.set_axisbelow(True)

    axes[0].legend(fontsize = 8, title = "Qwen3 size", title_fontsize = 8)
    figure.suptitle("Return on thinking by difficulty (95 percent intervals)", fontsize = 11)
    figure.tight_layout()
    figure.savefig(figuresDir / "thinking_return.png", dpi = 200)
    plt.close(figure)


def plotComputeLadder(runTable: pd.DataFrame):
    levels = difficultyLevels + ["all"]
    groupLabels = []
    for level in levels:
        groupLabels.append(difficultyNames[level])

    fourB = runTable[runTable["modelName"] == "Qwen3-4B"]

    # the single thinking run is seed 42 since thats the baseline the task 3 test was fixed against
    ladderRows = [
        ("no thinking", "#E59866", fourB[fourB["thinking"] == "off"].iloc[0]),
        ("thinking, one sample (seed 42)", "#55A868",
         fourB[(fourB["thinking"] == "on") & (fourB["isVoted"] == False) & (fourB["seed"] == 42)].iloc[0]),  # noqa: E712
        ("thinking, vote over 3", "#B3A9D6", fourB[fourB["run"].str.contains("N=3")].iloc[0]),
        ("thinking, vote over 5", "#8172B3", fourB[fourB["run"].str.contains("N=5")].iloc[0]),
    ]

    barSpecs = []
    for barLabel, barColour, tableRow in ladderRows:
        values = []
        for level in levels:
            values.append(tableRow[f"ex_{level}"])
        barSpecs.append((barLabel, barColour, values))

    figure, axis = plt.subplots(figsize = (8, 4.6))
    barWidth = 0.19
    drawGroupedBars(axis, groupLabels, barSpecs, barWidth = barWidth)

    # best of 5 is the share of questions where at least one of the 5 samples was right
    # no vote can beat it so it shows how much room voting had
    byDifficulty = pd.read_csv(selfConsistencyDir / "by_difficulty.csv")
    for groupPosition, level in enumerate(levels):
        upperBound = byDifficulty[byDifficulty["difficulty"] == level].iloc[0]["upper_bound_any_right"]
        clusterHalfWidth = 2 * barWidth
        if groupPosition == 0:
            boundLabel = "best of 5 (upper bound)"
        else:
            boundLabel = None
        axis.hlines(upperBound, groupPosition - clusterHalfWidth, groupPosition + clusterHalfWidth,
                    colors = "black", linestyles = "dashed", linewidth = 1, label = boundLabel)

    axis.set_ylim(0, 1)
    axis.set_xlabel("Spider difficulty")
    axis.set_ylabel("execution accuracy (EX)")
    axis.set_title("Qwen3-4B as test time compute is added", fontsize = 11)
    axis.grid(axis = "y", alpha = 0.3)
    axis.set_axisbelow(True)

    # matplotlib lists the dashed bound first so it gets moved behind the bars to follow the ladder order
    handles, labels = axis.get_legend_handles_labels()
    boundPosition = labels.index("best of 5 (upper bound)")
    boundHandle = handles.pop(boundPosition)
    boundLabel = labels.pop(boundPosition)
    handles.append(boundHandle)
    labels.append(boundLabel)
    axis.legend(handles, labels, fontsize = 8, ncol = 3, loc = "upper center", bbox_to_anchor = (0.5, -0.14), frameon = False)
    figure.tight_layout()
    figure.savefig(figuresDir / "compute_ladder.png", dpi = 200)
    plt.close(figure)


def legendSeriesFor(groupRow) -> str:
    if groupRow["isQuantized"]:
        return "Qwen2.5-Coder 4 bit AWQ"
    if groupRow["isVoted"]:
        return "Qwen3-4B thinking self consistency"
    return seriesName(groupRow)


def plotEmAgainstExSquare(groups: pd.DataFrame):
    """
    overall em against overall ex for every task 1 configuration on one shared range for both axes
    so a step right means the same as a step up and the gap between the metrics reads straight off the plot
    voted runs stay out since self consistency only gets introduced in task 3
    """
    plainGroups = groups[groups["isVoted"] == False]  # noqa: E712

    figure, axis = plt.subplots(figsize = (7, 7))

    axisLow = 0.15
    axisHigh = 0.85

    # the 7b pair and the two bigger thinking runs sit close together so their labels go to set sides
    # keyed by legend series then point label
    labelNudges = {
        ("Qwen2.5-Coder 4 bit AWQ", "7B AWQ"): (6, 3, "left"),
        ("Qwen2.5-Coder", "7B"): (6, -10, "left"),
        ("Qwen3 thinking", "8B"): (-7, 3, "right"),
        ("Qwen3 thinking", "4B"): (0, -13, "center"),
    }

    for _, groupRow in plainGroups.iterrows():
        style = pointStyle(groupRow)
        pointPosition = (groupRow["emMean"], groupRow["exMean"])
        pointLabel = shortLabel(groupRow)
        labelKey = (legendSeriesFor(groupRow), pointLabel)

        axis.scatter(pointPosition[0], pointPosition[1], marker = style["marker"], s = 55, color = style["colour"])

        offsetX, offsetY, alignment = labelNudges.get(labelKey, (6, 3, "left"))
        axis.annotate(pointLabel, pointPosition, textcoords = "offset points", xytext = (offsetX, offsetY),
                      fontsize = 7, ha = alignment)

    for legendLabel, style in [
        ("Qwen2.5-Coder", seriesStyles["Qwen2.5-Coder"]),
        ("Qwen2.5-Coder 4 bit AWQ", variantStyles["Qwen2.5-Coder 4 bit AWQ"]),
        ("Qwen3 no thinking", seriesStyles["Qwen3 no thinking"]),
        ("Qwen3 thinking", seriesStyles["Qwen3 thinking"]),
    ]:
        axis.scatter([], [], marker = style["marker"], color = style["colour"], label = legendLabel)

    axis.set_xlim(axisLow, axisHigh)
    axis.set_ylim(axisLow, axisHigh)
    # equal scales so a step right means the same as a step up
    axis.set_aspect("equal")
    axis.set_xlabel("exact set match (EM)")
    axis.set_ylabel("execution accuracy (EX)")
    axis.set_title("Exact set match against execution accuracy", fontsize = 11)
    axis.grid(alpha = 0.3)
    # the bottom right is empty since no configuration scores high em with low ex
    axis.legend(fontsize = 8, loc = "lower right")
    figure.tight_layout()
    figure.savefig(figuresDir / "em_against_ex_square.png", dpi = 200)
    plt.close(figure)


def main():
    figuresDir.mkdir(parents = True, exist_ok = True)

    runTable = loadRunTable()
    groups = groupSeeds(runTable)

    plotSizeCurves(groups)
    plotCostTradeoff(groups, "cost_tradeoff.png", includeVoted = False, focusKeys = None)

    # the task 3 version keeps everything as grey context and colours only the points its tests compare
    votingFocus = {
        ("Qwen3 thinking", "4B"),
        ("Qwen3-4B thinking self consistency", "SC N=3"),
        ("Qwen3-4B thinking self consistency", "SC N=5"),
        ("Qwen2.5-Coder", "7B"),
    }
    plotCostTradeoff(groups, "cost_tradeoff_voting.png", includeVoted = True, focusKeys = votingFocus)
    plotByDifficulty(groups)
    plotThinkingReturn()
    plotComputeLadder(runTable)
    plotEmAgainstExSquare(groups)

    # the numbers behind both figures so the report text can quote them exactly
    groups.to_csv(figuresDir / "figure_points.csv", index = False)

    print(f"wrote size_curves.png cost_tradeoff.png cost_tradeoff_voting.png accuracy_by_difficulty.png thinking_return.png compute_ladder.png em_against_ex_square.png and figure_points.csv into {figuresDir}")


if __name__ == "__main__":
    main()