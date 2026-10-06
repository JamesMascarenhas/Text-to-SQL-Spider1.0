"""
pulls every scored full run together into the tables the report needs
reads the scored files and summary files that scoring.py writes plus the config files from colab
only looks at runs whose names start with full- so pilots never sneak in
works with whatever has been scored so far so it can be rerun as more runs finish
runs on the mac and only needs the standard library

tables it makes
  accuracy        em and ex and ex with gold values by difficulty plus a 95 percent interval on overall ex
  errors          every wrong answer put in exactly one bucket so the ex gap can be explained
  components      the papers component matching f1 over all examples
  efficiency      time and tokens per example plus truncation checks
  mcnemar         paired significance tests for the comparisons the report makes with holm correction
  seeds           mean and spread across seeds wherever there are repeat runs
  settings        the run settings straight from each config for the methodology

writes csv files and a summary.md into outputs/summary

how to run
    python src/summarizeRuns.py
"""

import csv
import json
import math
import statistics
from pathlib import Path

from spiderUtils import difficultyLevels, outDir, readJsonl


runsDir = outDir / "runs"
summaryDir = outDir / "summary"

# rows show up in this order and anything not listed goes at the end alphabetically
preferredRunOrder = [
    "full-qwen25coder-0.5b",
    "full-qwen25coder-1.5b",
    "full-qwen25coder-3b",
    "full-qwen25coder-7b",
    "full-qwen25coder-7b-awq",
    "full-qwen3-1.7b-nothink",
    "full-qwen3-1.7b-think",
    "full-qwen3-1.7b-think-seed66",
    "full-qwen3-1.7b-think-seed73",
    "full-qwen3-4b-nothink",
    "full-qwen3-4b-think",
    "full-qwen3-4b-think-seed66",
    "full-qwen3-4b-think-seed73",
    "full-qwen3-4b-think-seed137",
    "full-qwen3-4b-think-seed255",
    "full-qwen3-4b-think-sc3",
    "full-qwen3-4b-think-sc5",
    "full-qwen3-8b-nothink",
    "full-qwen3-8b-think",
]

# the comparisons the report actually makes
# each one is a research question family then a question then the two runs it compares
# it gets skipped if either run isnt scored yet
# holm correction runs inside each family so rq1 and rq3 dont inflate each others p values
plannedComparisons = [
    ("rq1", "size within coder 1.5b vs 7b", "full-qwen25coder-1.5b", "full-qwen25coder-7b"),
    ("rq1", "size within coder 0.5b vs 1.5b", "full-qwen25coder-0.5b", "full-qwen25coder-1.5b"),
    ("rq1", "size within coder 1.5b vs 3b", "full-qwen25coder-1.5b", "full-qwen25coder-3b"),
    ("rq1", "size within coder 3b vs 7b", "full-qwen25coder-3b", "full-qwen25coder-7b"),
    ("rq1", "thinking on vs off qwen3 1.7b", "full-qwen3-1.7b-nothink", "full-qwen3-1.7b-think"),
    ("rq1", "thinking on vs off qwen3 4b", "full-qwen3-4b-nothink", "full-qwen3-4b-think"),
    ("rq1", "thinking on vs off qwen3 8b", "full-qwen3-8b-nothink", "full-qwen3-8b-think"),
    ("rq1", "reasoning 4b vs coder 7b", "full-qwen25coder-7b", "full-qwen3-4b-think"),
    ("rq1", "quantization 7b bf16 vs awq", "full-qwen25coder-7b", "full-qwen25coder-7b-awq"),
    ("rq1", "qwen3 size no thinking 1.7b vs 4b", "full-qwen3-1.7b-nothink", "full-qwen3-4b-nothink"),
    ("rq1", "qwen3 size no thinking 4b vs 8b", "full-qwen3-4b-nothink", "full-qwen3-8b-nothink"),
    ("rq1", "qwen3 size thinking 1.7b vs 4b", "full-qwen3-1.7b-think", "full-qwen3-4b-think"),
    ("rq1", "qwen3 size thinking 4b vs 8b", "full-qwen3-4b-think", "full-qwen3-8b-think"),
    ("rq3", "self consistency 4b N3 vs single", "full-qwen3-4b-think", "full-qwen3-4b-think-sc3"),
    ("rq3", "self consistency 4b N5 vs single", "full-qwen3-4b-think", "full-qwen3-4b-think-sc5"),
    ("rq3", "self consistency 4b N5 vs coder 7b", "full-qwen25coder-7b", "full-qwen3-4b-think-sc5"),
]


# loading

def findScoredRuns() -> list:
    # a run counts as scored once its summary file exists since scoring.py writes that last
    runNames = []
    for summaryPath in runsDir.glob("full-*_summary.json"):
        runName = summaryPath.name.replace("_summary.json", "")

        # no config means the run was copied over before colab finished it so its numbers dont count yet
        if not (runsDir / f"{runName}_config.json").exists():
            print(f"skipping {runName} since it has no config file so it probably wasnt finished when it was copied")
            continue

        runNames.append(runName)

    orderedNames = []
    for runName in preferredRunOrder:
        if runName in runNames:
            orderedNames.append(runName)

    leftoverNames = sorted(set(runNames) - set(orderedNames))

    return orderedNames + leftoverNames


def loadRun(runName: str) -> dict:
    with open(runsDir / f"{runName}_summary.json") as summaryFile:
        runSummary = json.load(summaryFile)

    with open(runsDir / f"{runName}_config.json") as configFile:
        runConfig = json.load(configFile)

    scoredRows = readJsonl(runsDir / f"{runName}_scored.jsonl")

    # keyed by dev_idx so paired tests line up the same question across runs
    rowsById = {}
    for row in scoredRows:
        rowsById[row["dev_idx"]] = row

    return {
        "name": runName,
        "label": buildLabel(runConfig),
        "summary": runSummary,
        "config": runConfig,
        "rows": scoredRows,
        "rowsById": rowsById,
    }


def buildLabel(runConfig: dict) -> str:
    # a readable name for tables built from the config so it cant drift from what actually ran
    modelName = runConfig["model"].split("/")[-1]
    label = modelName

    if runConfig["thinking"] == "on":
        label = label + " think"
    elif runConfig["thinking"] == "off":
        label = label + " no think"

    # voted runs get named by their strategy instead of a seed
    if "strategy" in runConfig:
        label = label + f" self consistency N={runConfig['strategy']['n_samples']}"
        return label

    seed = runConfig["decoding"]["seed"]
    if runConfig["thinking"] == "on" and seed != 42:
        label = label + f" seed {seed}"

    return label


# small stats helpers

def wilsonInterval(nCorrect: int, nTotal: int, confidence: float = 0.95) -> tuple:
    """
    95 percent interval for an accuracy
    wilson instead of the plain normal one because it behaves better near 0 and 1
    """
    # exact z for the interval instead of the rounded 1.96 so it matches standard stats packages
    z = statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)

    if nTotal == 0:
        return float("nan"), float("nan")

    proportion = nCorrect / nTotal
    denominator = 1 + z * z / nTotal
    centre = (proportion + z * z / (2 * nTotal)) / denominator
    halfWidth = z * math.sqrt(proportion * (1 - proportion) / nTotal + z * z / (4 * nTotal * nTotal)) / denominator

    return centre - halfWidth, centre + halfWidth


def exactMcNemar(onlyFirstRight: int, onlySecondRight: int) -> float:
    """
    exact mcnemar test as a two sided binomial test on the questions where the two runs disagree
    questions both got right or both got wrong say nothing about which run is better so they drop out
    exact instead of the chi squared version since some comparisons have few disagreements
    """
    nDisagree = onlyFirstRight + onlySecondRight

    if nDisagree == 0:
        return 1.0

    smallerCount = min(onlyFirstRight, onlySecondRight)

    tailProbability = 0.0
    for k in range(smallerCount + 1):
        tailProbability = tailProbability + math.comb(nDisagree, k) * 0.5 ** nDisagree

    return min(1.0, 2 * tailProbability)


def holmAdjust(pValues: list) -> list:
    """
    holm correction since running several tests at once makes a lucky small p value more likely
    sorts the p values and scales each by how many tests are left then keeps them from going back down
    """
    nTests = len(pValues)

    def pValueAt(position):
        return pValues[position]

    order = sorted(range(nTests), key = pValueAt)

    adjusted = [0.0] * nTests
    runningMax = 0.0

    for rank, position in enumerate(order):
        scaled = (nTests - rank) * pValues[position]
        runningMax = max(runningMax, scaled)
        adjusted[position] = min(1.0, runningMax)

    return adjusted


# tables

def buildAccuracyTable(runs: list) -> list:
    tableRows = []

    for run in runs:
        scoresByDifficulty = run["summary"]["scores_by_difficulty"]
        nAll = scoresByDifficulty["all"]["n"]
        nExCorrect = round(scoresByDifficulty["all"]["ex"] * nAll)
        intervalLow, intervalHigh = wilsonInterval(nExCorrect, nAll)

        tableRow = {"run": run["label"]}
        for metric in ["em", "ex", "ex_gold_values"]:
            for level in difficultyLevels + ["all"]:
                tableRow[f"{metric}_{level}"] = scoresByDifficulty[level][metric]

        tableRow["ex_all_ci_low"] = intervalLow
        tableRow["ex_all_ci_high"] = intervalHigh
        tableRows.append(tableRow)

    return tableRows


def classifyError(row: dict) -> str:
    """
    puts each example in exactly one bucket checked in this order
      correct      ex says right
      no_sql       nothing could be extracted
      timeout      the query never finished within the scorers time limit
      crash        the query errors on the database
      value_only   runs and is wrong but becomes right once gold values go in
      structural   runs and is wrong even with gold values so the structure itself is off
    value_only leans on the lenient gold value check so its an upper bound
    """
    if row["ex"] == 1:
        return "correct"

    if row.get("pred_sql") is None:
        return "no_sql"

    # older scored files dont have this field and none of them had timeouts
    if row.get("timed_out", False):
        return "timeout"

    if row["exec_error"] is not None:
        return "crash"

    if row["ex_gold_values"] == 1:
        return "value_only"

    return "structural"


def buildErrorTable(runs: list) -> list:
    buckets = ["correct", "no_sql", "timeout", "crash", "value_only", "structural"]
    tableRows = []

    for run in runs:
        counts = {}
        for bucket in buckets:
            counts[bucket] = 0

        for row in run["rows"]:
            bucket = classifyError(row)
            counts[bucket] = counts[bucket] + 1

        nWrong = len(run["rows"]) - counts["correct"]

        tableRow = {"run": run["label"], "n": len(run["rows"]), "wrong": nWrong}
        for bucket in buckets:
            tableRow[bucket] = counts[bucket]

        # shares of the wrong answers since thats what says which kind of fix could help most
        for bucket in buckets[1:]:
            if nWrong > 0:
                tableRow[f"{bucket}_share_of_wrong"] = counts[bucket] / nWrong
            else:
                tableRow[f"{bucket}_share_of_wrong"] = float("nan")

        tableRows.append(tableRow)

    return tableRows


def buildComponentTable(runs: list) -> list:
    tableRows = []

    for run in runs:
        f1ByComponent = run["summary"]["component_matching"]["f1"]

        tableRow = {"run": run["label"]}
        for componentName, scoresByLevel in f1ByComponent.items():
            tableRow[componentName] = scoresByLevel["all"]

        tableRows.append(tableRow)

    return tableRows


def buildEfficiencyTable(runs: list) -> list:
    tableRows = []

    for run in runs:
        rows = run["rows"]

        secondsPerExample = []
        outputTokens = []
        reasoningTokens = []
        hitLength = 0
        notClosed = 0

        for row in rows:
            secondsPerExample.append(row["amortized_sec"])
            outputTokens.append(row["n_output_tokens"])

            # stopped by the token limit instead of finishing on its own
            if row["finish_reason"] == "length":
                hitLength = hitLength + 1

            if "n_reasoning_tokens" in row:
                reasoningTokens.append(row["n_reasoning_tokens"])
                if not row["think_closed"]:
                    notClosed = notClosed + 1

        tableRow = {
            "run": run["label"],
            # the mean of the amortized times is the same as total generation time over the number of examples
            "sec_per_example_mean": statistics.mean(secondsPerExample),
            "total_generation_min": sum(secondsPerExample) / 60,
            "output_tokens_mean": statistics.mean(outputTokens),
            "output_tokens_median": statistics.median(outputTokens),
            "output_tokens_max": max(outputTokens),
            "hit_token_limit": hitLength,
        }

        if len(reasoningTokens) > 0:
            tableRow["reasoning_tokens_mean"] = statistics.mean(reasoningTokens)
            tableRow["reasoning_tokens_median"] = statistics.median(reasoningTokens)
            tableRow["reasoning_tokens_max"] = max(reasoningTokens)
            tableRow["thinking_not_finished"] = notClosed
        else:
            tableRow["reasoning_tokens_mean"] = ""
            tableRow["reasoning_tokens_median"] = ""
            tableRow["reasoning_tokens_max"] = ""
            tableRow["thinking_not_finished"] = ""

        tableRows.append(tableRow)

    return tableRows


def buildMcNemarTable(runsByName: dict) -> list:
    tableRows = []

    for family, question, firstName, secondName in plannedComparisons:
        if firstName not in runsByName or secondName not in runsByName:
            continue

        firstRun = runsByName[firstName]
        secondRun = runsByName[secondName]

        # only questions both runs answered since a resumed or partial run might be missing some
        sharedIds = sorted(set(firstRun["rowsById"]) & set(secondRun["rowsById"]))

        for metric in ["ex", "em"]:
            onlyFirstRight = 0
            onlySecondRight = 0
            firstTotal = 0
            secondTotal = 0

            for devIdx in sharedIds:
                firstRight = firstRun["rowsById"][devIdx][metric]
                secondRight = secondRun["rowsById"][devIdx][metric]
                firstTotal = firstTotal + firstRight
                secondTotal = secondTotal + secondRight

                if firstRight == 1 and secondRight == 0:
                    onlyFirstRight = onlyFirstRight + 1
                elif firstRight == 0 and secondRight == 1:
                    onlySecondRight = onlySecondRight + 1

            tableRows.append({
                "family": family,
                "question": question,
                "metric": metric,
                "first": firstRun["label"],
                "second": secondRun["label"],
                "n": len(sharedIds),
                "first_acc": firstTotal / len(sharedIds),
                "second_acc": secondTotal / len(sharedIds),
                "difference": (secondTotal - firstTotal) / len(sharedIds),
                "only_first_right": onlyFirstRight,
                "only_second_right": onlySecondRight,
                "p_value": exactMcNemar(onlyFirstRight, onlySecondRight),
            })

    # holm runs separately for every research question and metric pair since each one is its own family of tests
    familyNames = []
    for tableRow in tableRows:
        if tableRow["family"] not in familyNames:
            familyNames.append(tableRow["family"])

    for family in familyNames:
        for metric in ["ex", "em"]:
            familyRows = []
            for tableRow in tableRows:
                if tableRow["family"] == family and tableRow["metric"] == metric:
                    familyRows.append(tableRow)

            pValues = []
            for tableRow in familyRows:
                pValues.append(tableRow["p_value"])

            adjusted = holmAdjust(pValues)
            for tableRow, adjustedP in zip(familyRows, adjusted):
                tableRow["p_holm"] = adjustedP

    return tableRows


def buildSeedTable(runs: list) -> list:
    # groups runs that are the same model and thinking mode and only differ by seed
    groups = {}
    for run in runs:
        # voted runs combine several seeds so they arent another sample of the same thing
        if "strategy" in run["config"]:
            continue

        groupKey = (run["config"]["model"], run["config"]["thinking"])
        if groupKey not in groups:
            groups[groupKey] = []
        groups[groupKey].append(run)

    tableRows = []

    for (modelId, thinking), groupRuns in groups.items():
        # a single run has no spread to report
        if len(groupRuns) < 2:
            continue

        seeds = []
        for run in groupRuns:
            seeds.append(run["config"]["decoding"]["seed"])

        seedTexts = []
        for seed in sorted(seeds):
            seedTexts.append(str(seed))

        tableRow = {
            "model": modelId.split("/")[-1],
            "thinking": thinking,
            "seeds": " ".join(seedTexts),
            "n_runs": len(groupRuns),
        }

        for metric in ["em", "ex", "ex_gold_values"]:
            values = []
            for run in groupRuns:
                values.append(run["summary"]["scores_by_difficulty"]["all"][metric])

            tableRow[f"{metric}_mean"] = statistics.mean(values)
            tableRow[f"{metric}_sd"] = statistics.stdev(values)
            tableRow[f"{metric}_min"] = min(values)
            tableRow[f"{metric}_max"] = max(values)

        tableRows.append(tableRow)

    return tableRows


def buildSettingsTable(runs: list) -> list:
    tableRows = []

    for run in runs:
        runConfig = run["config"]
        decoding = runConfig["decoding"]
        lastSession = runConfig["sessions"][-1]

        # older configs were written before dtype_used existed and those runs used exactly what was asked for
        dtypeUsed = runConfig.get("dtype_used", f"{runConfig['dtype']} as requested")

        tableRows.append({
            "run": run["label"],
            "model": runConfig["model"],
            "revision": runConfig["revision"],
            "thinking": runConfig["thinking"],
            "quantization": json.dumps(runConfig["quantization"]),
            "dtype_used": dtypeUsed,
            "temperature": decoding["temperature"],
            "top_p": decoding["top_p"],
            "top_k": decoding["top_k"],
            "max_new_tokens": decoding["max_new_tokens"],
            "seed": decoding["seed"],
            "max_model_len": runConfig["max_model_len"],
            "chunk": runConfig["chunk"],
            "gpu": lastSession["gpu"],
            "vllm": lastSession["vllm"],
            "n_sessions": len(runConfig["sessions"]),
        })

    return tableRows


# writing

def formatCell(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"

    return str(value)


def writeCsv(tableRows: list, path: Path):
    if len(tableRows) == 0:
        return

    # union of keys in first seen order since some rows have extra columns
    fieldNames = []
    for tableRow in tableRows:
        for key in tableRow:
            if key not in fieldNames:
                fieldNames.append(key)

    with open(path, "w", newline = "") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames = fieldNames)
        writer.writeheader()
        for tableRow in tableRows:
            writer.writerow(tableRow)


def toMarkdown(tableRows: list, columns: list) -> str:
    if len(tableRows) == 0:
        return "_no runs available for this table yet_\n"

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))

    for tableRow in tableRows:
        cells = []
        for column in columns:
            cells.append(formatCell(tableRow.get(column, "")))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main():
    runNames = findScoredRuns()
    if len(runNames) == 0:
        print("no scored full runs found so score something first")
        return

    runs = []
    for runName in runNames:
        runs.append(loadRun(runName))

    runsByName = {}
    for run in runs:
        runsByName[run["name"]] = run

    print(f"found {len(runs)} scored runs")
    for run in runs:
        print(f"  {run['name']}  as  {run['label']}")

    accuracyRows = buildAccuracyTable(runs)
    errorRows = buildErrorTable(runs)
    componentRows = buildComponentTable(runs)
    efficiencyRows = buildEfficiencyTable(runs)
    mcnemarRows = buildMcNemarTable(runsByName)
    seedRows = buildSeedTable(runs)
    settingsRows = buildSettingsTable(runs)

    summaryDir.mkdir(parents = True, exist_ok = True)
    writeCsv(accuracyRows, summaryDir / "accuracy.csv")
    writeCsv(errorRows, summaryDir / "errors.csv")
    writeCsv(componentRows, summaryDir / "components.csv")
    writeCsv(efficiencyRows, summaryDir / "efficiency.csv")
    writeCsv(mcnemarRows, summaryDir / "mcnemar.csv")
    writeCsv(seedRows, summaryDir / "seeds.csv")
    writeCsv(settingsRows, summaryDir / "settings.csv")

    # one readable file with every table in it
    sections = []

    sections.append("## Accuracy overall\n")
    sections.append(toMarkdown(accuracyRows, ["run", "em_all", "ex_all", "ex_all_ci_low", "ex_all_ci_high", "ex_gold_values_all"]))

    for metric in ["ex", "em"]:
        sections.append(f"\n## {metric.upper()} by difficulty\n")
        columns = ["run"]
        for level in difficultyLevels + ["all"]:
            columns.append(f"{metric}_{level}")
        sections.append(toMarkdown(accuracyRows, columns))

    sections.append("\n## Error breakdown\n")
    sections.append(toMarkdown(errorRows, [
        "run", "wrong", "no_sql", "timeout", "crash", "value_only", "structural",
        "crash_share_of_wrong", "value_only_share_of_wrong", "structural_share_of_wrong",
    ]))

    sections.append("\n## Component matching F1 over all examples\n")
    if len(componentRows) > 0:
        componentColumns = list(componentRows[0].keys())
        sections.append(toMarkdown(componentRows, componentColumns))

    sections.append("\n## Efficiency\n")
    sections.append(toMarkdown(efficiencyRows, [
        "run", "sec_per_example_mean", "total_generation_min", "output_tokens_mean", "output_tokens_median",
        "output_tokens_max", "reasoning_tokens_mean", "reasoning_tokens_max", "hit_token_limit", "thinking_not_finished",
    ]))

    sections.append("\n## McNemar tests\n")
    sections.append(toMarkdown(mcnemarRows, [
        "family", "question", "metric", "first_acc", "second_acc", "difference",
        "only_first_right", "only_second_right", "p_value", "p_holm",
    ]))

    sections.append("\n## Seed variation\n")
    sections.append(toMarkdown(seedRows, [
        "model", "thinking", "seeds", "ex_mean", "ex_sd", "ex_min", "ex_max", "em_mean", "em_sd",
    ]))

    sections.append("\n## Run settings\n")
    sections.append(toMarkdown(settingsRows, [
        "run", "revision", "quantization", "dtype_used", "temperature", "top_p", "top_k",
        "max_new_tokens", "seed", "max_model_len", "gpu", "vllm", "n_sessions",
    ]))

    markdownPath = summaryDir / "summary.md"
    markdownPath.write_text("\n".join(sections))

    print(f"wrote csv files and {markdownPath}")


if __name__ == "__main__":
    main()