"""
task 2 which asks how features of the reasoning traces relate to text to sql performance
reads the scored files of thinking runs and turns every trace into a handful of measurable features
then relates each feature to correctness and to difficulty and checks whether it says anything beyond trace length
runs on the mac after scoring

features
  logLength       log of the number of reasoning tokens since effort and the models own uncertainty show up as length
  branchPer1k     how often the trace says alternatively per 1000 words which is the slides explore another branch operator
  sqlDrafts       how many uppercase SELECT keywords the trace writes which is how much sql it drafts and redrafts
  waitPer1k       how often it says wait per 1000 words which is the slides reconsider operator and serves as a contrast
  answerInTrace   whether the final sql appears word for word in the trace as a surface check on faithfulness

correctness is execution accuracy since em misjudges these models as task 1 showed

statistics
  descriptives    median and interquartile range by correctness and difficulty
  mann whitney    correct vs wrong per feature with a rank biserial effect size since the features are heavily skewed
  kruskal wallis  across the four difficulty levels plus spearman against difficulty as an ordered scale
  auroc           how well each feature alone separates wrong from right overall and within each difficulty
  logistic        correctness on difficulty and each feature with and without log length
                  standard errors are clustered by gold query since spider paraphrase pairs share one
  faithfulness    share of answers found in the trace by correctness and difficulty with a fisher exact test
  replication     the key numbers again on the other seeds and on other model sizes

writes csv files plus figures and a traceAnalysis.md into outputs/traceAnalysis

how to run
    python src/analyzeTraces.py
    python src/analyzeTraces.py --primary full-qwen3-4b-think --replicate full-qwen3-4b-think-seed66 full-qwen3-1.7b-think
"""

import argparse
import math
import re

import matplotlib

# draws straight to files so it works without a screen
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, spearmanr  # noqa: E402

from spiderUtils import difficultyLevels, outDir, readJsonl  # noqa: E402


runsDir = outDir / "runs"
analysisDir = outDir / "traceAnalysis"

defaultPrimary = "full-qwen3-4b-think"
# every thinking run that could exist and anything not scored yet just gets skipped
defaultReplications = [
    "full-qwen3-4b-think-seed66",
    "full-qwen3-4b-think-seed73",
    "full-qwen3-4b-think-seed137",
    "full-qwen3-4b-think-seed255",
    "full-qwen3-1.7b-think",
    "full-qwen3-1.7b-think-seed66",
    "full-qwen3-1.7b-think-seed73",
    "full-qwen3-8b-think",
    "full-qwen3-8b-think-seed66",
    "full-qwen3-8b-think-seed73",
]

# the features that get the full set of tests and the labels used in tables and plots
mainFeatures = ["logLength", "branchPer1k", "sqlDrafts", "waitPer1k"]
featureLabels = {
    "logLength": "log reasoning tokens",
    "branchPer1k": "alternatively per 1000 words",
    "sqlDrafts": "SELECT keywords drafted",
    "waitPer1k": "wait per 1000 words",
}

difficultyRanks = {"easy": 0, "medium": 1, "hard": 2, "extra": 3}

wordPattern = re.compile(r"[a-z_][a-z0-9_]*")
branchPattern = re.compile(r"\balternatively\b")
waitPattern = re.compile(r"\bwait\b")

# uppercase only since drafted sql is written in capitals and lowercase select is usually prose like i need to select
draftPattern = re.compile(r"\bSELECT\b")


# features

def normalizeSql(sqlText: str) -> str:
    # makes small formatting differences not matter when looking for the answer inside the trace
    normalized = sqlText.lower()
    normalized = normalized.replace('"', "'")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*([(),=<>])\s*", r"\1", normalized)
    normalized = normalized.strip()
    normalized = normalized.rstrip(";")

    return normalized


def extractFeatures(row: dict) -> dict:
    # the decoded trace starts with the think tag as plain text so its dropped before counting words
    traceText = row["reasoning_text"].replace("<think>", " ")
    lowerText = traceText.lower()

    nWords = len(wordPattern.findall(lowerText))
    nWords = max(1, nWords)

    predSql = row.get("pred_sql")
    if predSql:
        answerInTrace = int(normalizeSql(predSql) in normalizeSql(traceText))
    else:
        answerInTrace = 0

    return {
        "dev_idx": row["dev_idx"],
        "difficulty": row["difficulty"],
        "difficultyRank": difficultyRanks[row["difficulty"]],
        "correct": row["ex"],
        "reasoningTokens": row["n_reasoning_tokens"],
        "logLength": math.log(max(1, row["n_reasoning_tokens"])),
        "branchPer1k": 1000 * len(branchPattern.findall(lowerText)) / nWords,
        "sqlDrafts": len(draftPattern.findall(traceText)),
        "waitPer1k": 1000 * len(waitPattern.findall(lowerText)) / nWords,
        "answerInTrace": answerInTrace,
        "thinkClosed": int(row["think_closed"]),
        # paraphrase pairs share a gold query so they get grouped together for the clustered errors
        "goldGroup": row["db_id"] + " " + normalizeSql(row["gold"]),
    }


def loadFeatures(runName: str):
    scoredPath = runsDir / f"{runName}_scored.jsonl"
    if not scoredPath.exists():
        return None

    rows = readJsonl(scoredPath)

    # only thinking runs have traces to analyze
    if "reasoning_text" not in rows[0]:
        print(f"skipping {runName} since it has no reasoning traces")
        return None

    featureRows = []
    for row in rows:
        featureRows.append(extractFeatures(row))

    features = pd.DataFrame(featureRows)
    features["goldGroupCode"] = pd.factorize(features["goldGroup"])[0]

    return features


# statistics helpers

def wrongVsRightAuroc(features: pd.DataFrame, featureName: str) -> tuple:
    """
    chance that a randomly picked wrong answer has a higher value than a randomly picked right one
    0.5 means the feature cant tell them apart and 1 means higher always means wrong
    computed from the mann whitney u statistic so ties count as half
    """
    wrongValues = features.loc[features["correct"] == 0, featureName]
    rightValues = features.loc[features["correct"] == 1, featureName]

    if len(wrongValues) == 0 or len(rightValues) == 0:
        return float("nan"), len(wrongValues), len(rightValues)

    uStatistic = mannwhitneyu(wrongValues, rightValues).statistic
    auroc = uStatistic / (len(wrongValues) * len(rightValues))

    return auroc, len(wrongValues), len(rightValues)


def fitClusteredLogit(features: pd.DataFrame, formula: str):
    # clustered by gold query so paraphrase pairs dont count as independent evidence
    model = smf.logit(formula, features)
    fitted = model.fit(disp = 0, cov_type = "cluster", cov_kwds = {"groups": features["goldGroupCode"]})

    return fitted


# tables

def buildDescriptives(features: pd.DataFrame) -> pd.DataFrame:
    descriptiveRows = []

    for featureName in mainFeatures + ["reasoningTokens"]:
        for level in difficultyLevels + ["all"]:
            if level == "all":
                levelFeatures = features
            else:
                levelFeatures = features[features["difficulty"] == level]

            for correctValue, correctLabel in [(1, "right"), (0, "wrong")]:
                values = levelFeatures.loc[levelFeatures["correct"] == correctValue, featureName]

                if len(values) == 0:
                    continue

                descriptiveRows.append({
                    "feature": featureName,
                    "difficulty": level,
                    "group": correctLabel,
                    "n": len(values),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "mean": values.mean(),
                })

    return pd.DataFrame(descriptiveRows)


def buildTests(features: pd.DataFrame) -> pd.DataFrame:
    testRows = []

    for featureName in mainFeatures:
        wrongValues = features.loc[features["correct"] == 0, featureName]
        rightValues = features.loc[features["correct"] == 1, featureName]
        mannWhitney = mannwhitneyu(wrongValues, rightValues)

        auroc, _, _ = wrongVsRightAuroc(features, featureName)

        # one group per difficulty level for kruskal wallis
        valuesByLevel = []
        for level in difficultyLevels:
            valuesByLevel.append(features.loc[features["difficulty"] == level, featureName])
        kruskalResult = kruskal(*valuesByLevel)

        spearmanResult = spearmanr(features["difficultyRank"], features[featureName])
        lengthCorrelation = spearmanr(features["logLength"], features[featureName])

        testRows.append({
            "feature": featureName,
            "mannwhitney_u": mannWhitney.statistic,
            "mannwhitney_p": mannWhitney.pvalue,
            # rank biserial runs from -1 to 1 and positive means wrong answers tend to have higher values
            "rank_biserial": 2 * auroc - 1,
            "kruskal_h": kruskalResult.statistic,
            "kruskal_p": kruskalResult.pvalue,
            "spearman_with_difficulty": spearmanResult.statistic,
            "spearman_difficulty_p": spearmanResult.pvalue,
            "spearman_with_length": lengthCorrelation.statistic,
        })

    return pd.DataFrame(testRows)


def buildAurocTable(features: pd.DataFrame) -> pd.DataFrame:
    aurocRows = []

    for featureName in mainFeatures:
        tableRow = {"feature": featureName}

        for level in difficultyLevels + ["all"]:
            if level == "all":
                levelFeatures = features
            else:
                levelFeatures = features[features["difficulty"] == level]

            auroc, nWrong, nRight = wrongVsRightAuroc(levelFeatures, featureName)
            tableRow[f"auroc_{level}"] = auroc
            tableRow[f"n_wrong_{level}"] = nWrong

        aurocRows.append(tableRow)

    return pd.DataFrame(aurocRows)


def buildRegressionTable(features: pd.DataFrame) -> pd.DataFrame:
    """
    each feature gets two models
      without length   correct ~ difficulty + feature which says whether it matters at all once difficulty is fixed
      with length      correct ~ difficulty + log length + feature which says whether it adds anything beyond length
    odds ratios are per one unit of the feature and for log length one unit means about 2.7 times longer
    """
    regressionRows = []

    for featureName in mainFeatures:
        formulas = [("without length", f"correct ~ C(difficulty) + {featureName}")]

        if featureName != "logLength":
            formulas.append(("with length", f"correct ~ C(difficulty) + logLength + {featureName}"))

        for modelName, formula in formulas:
            fitted = fitClusteredLogit(features, formula)
            interval = fitted.conf_int().loc[featureName]

            regressionRows.append({
                "feature": featureName,
                "model": modelName,
                "coef": fitted.params[featureName],
                "odds_ratio": math.exp(fitted.params[featureName]),
                "or_ci_low": math.exp(interval[0]),
                "or_ci_high": math.exp(interval[1]),
                "p_clustered": fitted.pvalues[featureName],
                "n": int(fitted.nobs),
                "n_gold_groups": features["goldGroupCode"].nunique(),
            })

    return pd.DataFrame(regressionRows)


def buildFaithfulnessTable(features: pd.DataFrame) -> tuple:
    faithfulnessRows = []

    for level in difficultyLevels + ["all"]:
        if level == "all":
            levelFeatures = features
        else:
            levelFeatures = features[features["difficulty"] == level]

        faithfulnessRows.append({
            "difficulty": level,
            "n": len(levelFeatures),
            "answer_in_trace_share": levelFeatures["answerInTrace"].mean(),
            "share_right_when_in_trace": levelFeatures.loc[levelFeatures["answerInTrace"] == 1, "correct"].mean(),
            "share_right_when_not_in_trace": levelFeatures.loc[levelFeatures["answerInTrace"] == 0, "correct"].mean(),
        })

    # 2 by 2 of in trace vs correct and fisher since one cell can be small
    contingency = pd.crosstab(features["answerInTrace"], features["correct"])
    fisherResult = fisher_exact(contingency.values)

    return pd.DataFrame(faithfulnessRows), fisherResult.pvalue


def summarizeForReplication(runName: str, features: pd.DataFrame) -> dict:
    # the handful of numbers that decide whether the main pattern holds in another sample
    lengthFit = fitClusteredLogit(features, "correct ~ C(difficulty) + logLength")

    summaryRow = {
        "run": runName,
        "n": len(features),
        "ex": features["correct"].mean(),
        "thinking_not_finished": int((features["thinkClosed"] == 0).sum()),
        "median_tokens_right": features.loc[features["correct"] == 1, "reasoningTokens"].median(),
        "median_tokens_wrong": features.loc[features["correct"] == 0, "reasoningTokens"].median(),
        "auroc_length": wrongVsRightAuroc(features, "logLength")[0],
        "length_odds_ratio": math.exp(lengthFit.params["logLength"]),
        "length_p": lengthFit.pvalues["logLength"],
        "answer_in_trace_share": features["answerInTrace"].mean(),
    }

    # does each content feature add anything beyond length here too
    # the sign matters as much as the p value since positive means more of it goes with being right at the same length
    for featureName in ["branchPer1k", "sqlDrafts", "waitPer1k"]:
        fitted = fitClusteredLogit(features, f"correct ~ C(difficulty) + logLength + {featureName}")
        summaryRow[f"{featureName}_coef_beyond_length"] = fitted.params[featureName]
        summaryRow[f"{featureName}_p_beyond_length"] = fitted.pvalues[featureName]

    return summaryRow


# figures

def plotFeatureBoxes(features: pd.DataFrame, runName: str):
    # one panel per feature with right and wrong side by side at every difficulty level
    figure, axes = plt.subplots(1, len(mainFeatures), figsize = (4.2 * len(mainFeatures), 4))

    for axis, featureName in zip(axes, mainFeatures):
        boxData = []
        boxPositions = []
        boxColours = []

        for levelPosition, level in enumerate(difficultyLevels):
            levelFeatures = features[features["difficulty"] == level]

            for offset, correctValue, colour in [(-0.18, 1, "#4C72B0"), (0.18, 0, "#DD8452")]:
                values = levelFeatures.loc[levelFeatures["correct"] == correctValue, featureName]
                boxData.append(values)
                boxPositions.append(levelPosition + offset)
                boxColours.append(colour)

        boxes = axis.boxplot(boxData, positions = boxPositions, widths = 0.3, patch_artist = True, showfliers = False)
        for patch, colour in zip(boxes["boxes"], boxColours):
            patch.set_facecolor(colour)

        axis.set_xticks(range(len(difficultyLevels)))
        axis.set_xticklabels(difficultyLevels)
        axis.set_title(featureLabels[featureName], fontsize = 10)

    # a manual legend since the boxes were drawn in pairs
    axes[0].plot([], [], color = "#4C72B0", linewidth = 8, label = "right")
    axes[0].plot([], [], color = "#DD8452", linewidth = 8, label = "wrong")
    axes[0].legend(loc = "upper left", fontsize = 8)

    figure.suptitle(f"Trace features by difficulty and correctness ({runName}, outliers hidden)", fontsize = 11)
    figure.tight_layout()
    figure.savefig(analysisDir / f"boxplots_{runName}.png", dpi = 200)
    plt.close(figure)


def plotAccuracyByLength(features: pd.DataFrame, runName: str):
    # quartiles of length within each difficulty level so harder questions arent just shoved into the long bins
    figure, axis = plt.subplots(figsize = (6, 4))

    for level in difficultyLevels:
        levelFeatures = features[features["difficulty"] == level].copy()
        levelFeatures["lengthQuartile"] = pd.qcut(levelFeatures["reasoningTokens"], 4, labels = False, duplicates = "drop")

        quartileMedians = []
        quartileAccuracies = []
        for quartile, quartileFeatures in levelFeatures.groupby("lengthQuartile"):
            quartileMedians.append(quartileFeatures["reasoningTokens"].median())
            quartileAccuracies.append(quartileFeatures["correct"].mean())

        axis.plot(quartileMedians, quartileAccuracies, marker = "o", label = level)

    axis.set_xscale("log")
    axis.set_xlabel("median reasoning tokens in the length quartile (log scale)")
    axis.set_ylabel("execution accuracy")
    axis.set_title(f"Accuracy by trace length within each difficulty ({runName})", fontsize = 10)
    axis.legend(title = "difficulty", fontsize = 8)
    figure.tight_layout()
    figure.savefig(analysisDir / f"accuracy_by_length_{runName}.png", dpi = 200)
    plt.close(figure)


# writing

def toMarkdown(table: pd.DataFrame) -> str:
    if len(table) == 0:
        return "_nothing to show_\n"

    columns = list(table.columns)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))

    for _, tableRow in table.iterrows():
        cells = []
        for column in columns:
            value = tableRow[column]
            if isinstance(value, float):
                # tiny p values stay readable in scientific notation
                if value != 0 and abs(value) < 0.001:
                    cells.append(f"{value:.1e}")
                else:
                    cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", default = defaultPrimary)
    parser.add_argument("--replicate", nargs = "*", default = defaultReplications)
    args = parser.parse_args()

    analysisDir.mkdir(parents = True, exist_ok = True)

    primaryFeatures = loadFeatures(args.primary)
    if primaryFeatures is None:
        print(f"no scored traces found for {args.primary}")
        return

    primaryFeatures.to_csv(analysisDir / f"features_{args.primary}.csv", index = False)

    descriptives = buildDescriptives(primaryFeatures)
    tests = buildTests(primaryFeatures)
    aurocTable = buildAurocTable(primaryFeatures)
    regression = buildRegressionTable(primaryFeatures)
    faithfulness, fisherP = buildFaithfulnessTable(primaryFeatures)

    descriptives.to_csv(analysisDir / "descriptives.csv", index = False)
    tests.to_csv(analysisDir / "tests.csv", index = False)
    aurocTable.to_csv(analysisDir / "auroc.csv", index = False)
    regression.to_csv(analysisDir / "regression.csv", index = False)
    faithfulness.to_csv(analysisDir / "faithfulness.csv", index = False)

    plotFeatureBoxes(primaryFeatures, args.primary)
    plotAccuracyByLength(primaryFeatures, args.primary)

    # the primary run goes first in the replication table so everything sits side by side
    replicationRows = [summarizeForReplication(args.primary, primaryFeatures)]
    for runName in args.replicate:
        runFeatures = loadFeatures(runName)
        if runFeatures is None:
            print(f"skipping {runName} since it isnt scored yet")
            continue
        replicationRows.append(summarizeForReplication(runName, runFeatures))

    replication = pd.DataFrame(replicationRows)
    replication.to_csv(analysisDir / "replication.csv", index = False)

    sections = []
    sections.append(f"# Trace analysis for {args.primary}\n")
    sections.append(f"{len(primaryFeatures)} traces and {primaryFeatures['goldGroupCode'].nunique()} distinct gold queries\n")

    sections.append("\n## Descriptives over all difficulties\n")
    sections.append(toMarkdown(descriptives[descriptives["difficulty"] == "all"]))
    sections.append("\n## Descriptives by difficulty\n")
    sections.append(toMarkdown(descriptives[descriptives["difficulty"] != "all"]))
    sections.append("\n## Tests\n")
    sections.append(toMarkdown(tests))
    sections.append("\n## AUROC for telling wrong from right\n")
    sections.append(toMarkdown(aurocTable))
    sections.append("\n## Logistic regression with errors clustered by gold query\n")
    sections.append(toMarkdown(regression))
    sections.append("\n## Answer found in trace\n")
    sections.append(toMarkdown(faithfulness))
    sections.append(f"\nfisher exact p for answer in trace vs correct {fisherP:.3g}\n")
    sections.append("\n## Replication across seeds and sizes\n")
    sections.append(toMarkdown(replication))

    (analysisDir / "traceAnalysis.md").write_text("\n".join(sections))
    print(f"wrote tables figures and {analysisDir / 'traceAnalysis.md'}")


if __name__ == "__main__":
    main()