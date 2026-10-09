"""
turns the result files in outputs into latex tables for the report
every number in a report table comes straight from these files so nothing gets typed in by hand
each table is written as its own tabular into report/tables and main.tex wraps it with the caption
captions and labels live in main.tex so they can be edited without touching this script
runs on the mac after summarizeRuns analyzeTraces and selfConsistency

how to run
    python report/buildTables.py
"""

import csv
import json
import sys
from pathlib import Path

reportDir = Path(__file__).resolve().parent
projectRoot = reportDir.parent
outDir = projectRoot / "outputs"
summaryDir = outDir / "summary"
traceDir = outDir / "traceAnalysis"
consistencyDir = outDir / "selfConsistency"
runsDir = outDir / "runs"
tablesDir = reportDir / "tables"

# the prompt wording comes from the same place the experiments used
sys.path.insert(0, str(projectRoot / "src"))
from spiderUtils import systemPrompt, userTemplate  # noqa: E402

difficultyLevels = ["easy", "medium", "hard", "extra"]
levelHeaders = ["Easy", "Medium", "Hard", "Extra", "All"]

# the eleven task 1 configurations in the order every table shows them
# first is the label summarizeRuns uses and second is the short name the report uses
taskOneConfigs = [
    ("Qwen2.5-Coder-0.5B-Instruct", "Coder 0.5B"),
    ("Qwen2.5-Coder-1.5B-Instruct", "Coder 1.5B"),
    ("Qwen2.5-Coder-3B-Instruct", "Coder 3B"),
    ("Qwen2.5-Coder-7B-Instruct", "Coder 7B"),
    ("Qwen2.5-Coder-7B-Instruct-AWQ", "Coder 7B AWQ"),
    ("Qwen3-1.7B no think", "Qwen3-1.7B off"),
    ("Qwen3-1.7B think", "Qwen3-1.7B on"),
    ("Qwen3-4B no think", "Qwen3-4B off"),
    ("Qwen3-4B think", "Qwen3-4B on"),
    ("Qwen3-8B no think", "Qwen3-8B off"),
    ("Qwen3-8B think", "Qwen3-8B on"),
]

# a rule goes under the last coder row so the two families read as separate blocks
lastCoderLabel = "Qwen2.5-Coder-7B-Instruct-AWQ"


# reading

def readCsv(path: Path) -> list:
    with open(path, newline = "") as csvFile:
        reader = csv.DictReader(csvFile)
        rows = []
        for row in reader:
            rows.append(row)

    return rows


def indexBy(rows: list, keyName: str) -> dict:
    indexed = {}
    for row in rows:
        indexed[row[keyName]] = row

    return indexed


def readJsonl(path: Path) -> list:
    rows = []
    with open(path) as jsonlFile:
        for line in jsonlFile:
            if line.strip() == "":
                continue
            rows.append(json.loads(line))

    return rows


# formatting

def percent(value) -> str:
    # accuracies are stored as shares and the report shows them as percentages
    return f"{100 * float(value):.1f}"


def signedPoints(value) -> str:
    # a real minus sign in latex since a hyphen looks too short next to numbers
    points = 100 * float(value)
    if points < 0:
        return f"$-${abs(points):.1f}"
    return f"+{points:.1f}"


def plainPoints(value) -> str:
    points = 100 * float(value)
    if points < 0:
        return f"$-${abs(points):.1f}"
    return f"{points:.1f}"


def interval(low, high) -> str:
    return f"[{plainPoints(low)}, {plainPoints(high)}]"


def pValue(value) -> str:
    p = float(value)
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def scientificP(value) -> str:
    # tiny p values read better as powers of ten than as a row of zeros
    p = float(value)
    if p >= 0.001:
        return f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def plainNumber(value, decimals: int = 1) -> str:
    # whole numbers stay whole so a median of 2 doesnt print as 2.0
    number = float(value)
    if number == int(number):
        return f"{int(number):,}"
    return f"{number:,.{decimals}f}"


def latexEscape(text: str) -> str:
    # sql and questions can hold characters that mean something special to latex
    # one character at a time so a replacement never gets escaped a second time
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    escaped = ""
    for character in text:
        if character in replacements:
            escaped = escaped + replacements[character]
        else:
            escaped = escaped + character

    return escaped


def writeTable(fileName: str, lines: list):
    tablesDir.mkdir(parents = True, exist_ok = True)
    (tablesDir / fileName).write_text("\n".join(lines) + "\n")
    print(f"wrote {tablesDir / fileName}")


# task 1 tables

def buildModelsTable():
    # parameter counts are copied from the hugging face model cards since the run files dont hold them
    cardFacts = [
        ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen2.5-Coder-0.5B-Instruct", "Coder", "0.49B", "0.36B", "standard", "greedy", "512", "bf16"),
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen2.5-Coder-1.5B-Instruct", "Coder", "1.54B", "1.31B", "standard", "greedy", "512", "bf16"),
        ("Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen2.5-Coder-3B-Instruct", "Coder", "3.09B", "2.77B", "standard", "greedy", "512", "bf16"),
        ("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B-Instruct", "Coder", "7.61B", "6.53B", "standard", "greedy", "512", "bf16"),
        ("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", "Qwen2.5-Coder-7B-Instruct-AWQ", "Coder", "7.61B", "6.53B", "standard", "greedy", "512", "4 bit weights, fp16"),
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B think", "Qwen3", "1.7B", "1.4B", "off / on", "greedy / sampled", "512 / 8,192", "bf16"),
        ("Qwen/Qwen3-4B", "Qwen3-4B think", "Qwen3", "4.0B", "3.6B", "off / on", "greedy / sampled", "512 / 8,192", "bf16"),
        ("Qwen/Qwen3-8B", "Qwen3-8B think", "Qwen3", "8.2B", "6.95B", "off / on", "greedy / sampled", "512 / 8,192", "bf16"),
    ]
    settingsByRun = indexBy(readCsv(summaryDir / "settings.csv"), "run")

    lines = [
        "\\begin{tabular}{@{}llllllll l@{}}",
        "\\toprule",
        "Model (Hugging Face ID) & Family & Params & Non-emb. & Modes & Decoding & Max new tok. & Precision & Rev. \\\\",
        "\\midrule",
    ]
    for modelId, runLabel, family, params, nonEmbedding, modes, decoding, maxTokens, precision in cardFacts:
        revision = settingsByRun[runLabel]["revision"][:7]
        lines.append(
            f"\\texttt{{{latexEscape(modelId)}}} & {family} & {params} & {nonEmbedding} & {modes} & {decoding} & {maxTokens} & {precision} & \\texttt{{{revision}}} \\\\"
        )
        if runLabel == lastCoderLabel:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("models.tex", lines)


def buildInferenceTable():
    # these settings are fixed by the code so they come from the settings file to make sure they match
    settingsByRun = indexBy(readCsv(summaryDir / "settings.csv"), "run")
    greedy = settingsByRun["Qwen3-4B no think"]
    sampled = settingsByRun["Qwen3-4B think"]

    lines = [
        "\\begin{tabular}{@{}lll@{}}",
        "\\toprule",
        "Setting & Coder and Qwen3 no thinking & Qwen3 thinking \\\\",
        "\\midrule",
        "Decoding & greedy & sampling \\\\",
        f"Temperature / top-p / top-k & {float(greedy['temperature']):g} / {float(greedy['top_p']):.1f} / off & {float(sampled['temperature']):g} / {float(sampled['top_p']):g} / {sampled['top_k']} \\\\",
        f"Max new tokens & {int(greedy['max_new_tokens']):,} & {int(sampled['max_new_tokens']):,} \\\\",
        "Reasoning & off & on (\\texttt{enable\\_thinking}) \\\\",
        "Seeds & 42 & 42, 66, 73 (plus 137, 255 for 4B) \\\\",
        "Context window & 16,384 (10,240 for 8B) & 16,384 (10,240 for 8B) \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    writeTable("inference.tex", lines)


def buildAccuracyTable():
    accuracyByRun = indexBy(readCsv(summaryDir / "accuracy.csv"), "run")

    # seed means only exist for the thinking runs
    seedRows = readCsv(summaryDir / "seeds.csv")
    seedByModel = {}
    for seedRow in seedRows:
        seedByModel[seedRow["model"]] = seedRow

    lines = [
        "\\begin{tabular}{@{}l rrrrr rrrrr c r r@{}}",
        "\\toprule",
        " & \\multicolumn{5}{c}{EM} & \\multicolumn{5}{c}{EX} & EX all & EX gold & EX seed \\\\",
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
        "Configuration & Easy & Med. & Hard & Extra & All & Easy & Med. & Hard & Extra & All & 95\\% CI & values & mean $\\pm$ SD \\\\",
        "\\midrule",
    ]

    for runLabel, shortName in taskOneConfigs:
        row = accuracyByRun[runLabel]
        cells = [shortName]
        for metric in ["em", "ex"]:
            for level in difficultyLevels + ["all"]:
                cells.append(percent(row[f"{metric}_{level}"]))
        cells.append(interval(row["ex_all_ci_low"], row["ex_all_ci_high"]))
        cells.append(percent(row["ex_gold_values_all"]))

        seedCell = ""
        # only the thinking rows have seeds and the no think labels also end in think
        isThinkingRun = runLabel.endswith(" think") and not runLabel.endswith(" no think")
        if isThinkingRun:
            modelName = runLabel.replace(" think", "")
            seedRow = seedByModel[modelName]
            seedCell = f"{percent(seedRow['ex_mean'])} $\\pm$ {percent(seedRow['ex_sd'])}"
        cells.append(seedCell)

        lines.append(" & ".join(cells) + " \\\\")
        if runLabel == lastCoderLabel:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("accuracy.tex", lines)


# the planned comparisons in the order the report discusses them with their short names
comparisonOrder = [
    ("size within coder 0.5b vs 1.5b", "Coder 0.5B $\\to$ 1.5B"),
    ("size within coder 1.5b vs 3b", "Coder 1.5B $\\to$ 3B"),
    ("size within coder 3b vs 7b", "Coder 3B $\\to$ 7B"),
    ("size within coder 1.5b vs 7b", "Coder 1.5B $\\to$ 7B"),
    ("thinking on vs off qwen3 1.7b", "Qwen3-1.7B off $\\to$ on"),
    ("thinking on vs off qwen3 4b", "Qwen3-4B off $\\to$ on"),
    ("thinking on vs off qwen3 8b", "Qwen3-8B off $\\to$ on"),
    ("qwen3 size no thinking 1.7b vs 4b", "Qwen3 off, 1.7B $\\to$ 4B"),
    ("qwen3 size no thinking 4b vs 8b", "Qwen3 off, 4B $\\to$ 8B"),
    ("qwen3 size thinking 1.7b vs 4b", "Qwen3 on, 1.7B $\\to$ 4B"),
    ("qwen3 size thinking 4b vs 8b", "Qwen3 on, 4B $\\to$ 8B"),
    ("reasoning 4b vs coder 7b", "Coder 7B $\\to$ Qwen3-4B on"),
    ("quantization 7b bf16 vs awq", "Coder 7B bf16 $\\to$ AWQ"),
]


def buildMcNemarTable():
    rows = readCsv(summaryDir / "mcnemar.csv")
    rowByQuestionAndMetric = {}
    for row in rows:
        rowByQuestionAndMetric[(row["question"], row["metric"])] = row

    # the same comparisons rerun with every possibly lucky ex pass counted as wrong
    pessimisticRows = readCsv(summaryDir / "mcnemar_pessimistic.csv")
    for row in pessimisticRows:
        rowByQuestionAndMetric[(row["question"], row["metric"])] = row

    lines = [
        "\\begin{tabular}{@{}l r c r r r r@{}}",
        "\\toprule",
        " & \\multicolumn{4}{c}{EX} & \\multicolumn{2}{c}{EM} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
        "Comparison (first $\\to$ second) & Difference [95\\% CI] & Only first / only second & Holm $p$ & Pessimistic Holm $p$ & Difference [95\\% CI] & Holm $p$ \\\\",
        "\\midrule",
    ]
    for question, shortName in comparisonOrder:
        exRow = rowByQuestionAndMetric[(question, "ex")]
        emRow = rowByQuestionAndMetric[(question, "em")]
        pessimisticRow = rowByQuestionAndMetric[(question, "ex_pessimistic")]
        cells = [
            shortName,
            f"{signedPoints(exRow['difference'])} {interval(exRow['difference_ci_low'], exRow['difference_ci_high'])}",
            f"{exRow['only_first_right']} / {exRow['only_second_right']}",
            pValue(exRow["p_holm"]),
            pValue(pessimisticRow["p_holm"]),
            f"{signedPoints(emRow['difference'])} {interval(emRow['difference_ci_low'], emRow['difference_ci_high'])}",
            pValue(emRow["p_holm"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("mcnemar.tex", lines)


def buildEfficiencyTable():
    efficiencyByRun = indexBy(readCsv(summaryDir / "efficiency.csv"), "run")

    lines = [
        "\\begin{tabular}{@{}l rrrr@{}}",
        "\\toprule",
        "  & Seconds per & \\multicolumn{2}{c}{Output tokens per question} & Reached \\\\ Seconds per & \\multicolumn{2}{c}{Output tokens per question} & Reached \\\\",
        "\\cmidrule(lr){3-4}",
        "Configuration & question & Mean & Median & token limit \\\\",
        "\\midrule",
    ]
    for runLabel, shortName in taskOneConfigs:
        row = efficiencyByRun[runLabel]
        seconds = float(row["sec_per_example_mean"])
        if seconds < 1:
            secondsText = f"{seconds:.3f}"
        else:
            secondsText = f"{seconds:.2f}"
        cells = [
            shortName,
            secondsText,
            plainNumber(round(float(row["output_tokens_mean"]))),
            plainNumber(row["output_tokens_median"]),
            row["hit_token_limit"],
        ]
        lines.append(" & ".join(cells) + " \\\\")
        if runLabel == lastCoderLabel:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("efficiency.tex", lines)


def buildThinkingTable():
    rows = readCsv(summaryDir / "thinking.csv")
    rowBySizeAndLevel = {}
    for row in rows:
        rowBySizeAndLevel[(row["model"], row["difficulty"])] = row

    sizes = [("Qwen3-1.7B", "1.7B"), ("Qwen3-4B", "4B"), ("Qwen3-8B", "8B")]
    levels = difficultyLevels + ["all"]

    lines = [
        "\\begin{tabular}{@{}l ccccc@{}}",
        "\\toprule",
        "Size & " + " & ".join(levelHeaders) + " \\\\",
        "\\midrule",
        "\\multicolumn{6}{@{}l}{\\textit{EX gain from thinking, points [95\\% CI]}} \\\\",
    ]
    for modelName, shortName in sizes:
        cells = [shortName]
        for level in levels:
            row = rowBySizeAndLevel[(modelName, level)]
            cells.append(f"{signedPoints(row['gain'])} {interval(row['gain_ci_low'], row['gain_ci_high'])}")
        lines.append(" & ".join(cells) + " \\\\")

    # two separate panels so every cell holds a single number and nothing reads like a fraction
    lines.append("\\midrule")
    lines.append("\\multicolumn{6}{@{}l}{\\textit{Extra output tokens per question}} \\\\")
    for modelName, shortName in sizes:
        cells = [shortName]
        for level in levels:
            row = rowBySizeAndLevel[(modelName, level)]
            cells.append(plainNumber(round(float(row["extra_tokens"]))))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{6}{@{}l}{\\textit{EX points gained per 1,000 extra tokens}} \\\\")
    for modelName, shortName in sizes:
        cells = [shortName]
        for level in levels:
            row = rowBySizeAndLevel[(modelName, level)]
            cells.append(f"{float(row['points_per_1k_tokens']):.1f}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("thinking.tex", lines)


def buildEmParserTable():
    parserByRun = indexBy(readCsv(summaryDir / "em_parser.csv"), "run")

    lines = [
        "\\begin{tabular}{@{}l rrrrr@{}}",
        "\\toprule",
        " & & Cannot be & Of those, & Of those, & EX-correct answers \\\\",
        "Configuration & Predictions & parsed (\\%) & run & EX right & lost to the parser (\\%) \\\\",
        "\\midrule",
    ]
    for runLabel, shortName in taskOneConfigs:
        row = parserByRun[runLabel]
        cells = [
            shortName,
            row["n_predictions"],
            f"{row['unparseable']} ({percent(row['unparseable_share'])})",
            row["unparseable_but_runs"],
            row["unparseable_but_ex_right"],
            percent(row["ex_right_lost_to_parser"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
        if runLabel == lastCoderLabel:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("emParser.tex", lines)


def buildErrorTable():
    errorsByRun = indexBy(readCsv(summaryDir / "errors.csv"), "run")
    coincidenceByRun = indexBy(readCsv(summaryDir / "coincidence.csv"), "run")

    lines = [
        "\\begin{tabular}{@{}l rrr rrr rr@{}}",
        "\\toprule",
        " & & No & & \\multicolumn{3}{c}{Share of wrong answers (\\%)} & Trivial gold, & Pessimistic \\\\",
        "\\cmidrule(lr){5-7}",
        "Configuration & Wrong & prediction & Timeout & Crash & Value only & Structural & EM rejects & EX \\\\",
        "\\midrule",
    ]
    for runLabel, shortName in taskOneConfigs:
        errorRow = errorsByRun[runLabel]
        coincidenceRow = coincidenceByRun[runLabel]
        cells = [
            shortName,
            errorRow["wrong"],
            errorRow["no_sql"],
            errorRow["timeout"],
            percent(errorRow["crash_share_of_wrong"]),
            percent(errorRow["value_only_share_of_wrong"]),
            percent(errorRow["structural_share_of_wrong"]),
            coincidenceRow["right_on_trivial_em_wrong"],
            percent(coincidenceRow["ex_pessimistic"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
        if runLabel == lastCoderLabel:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("errors.tex", lines)


# task 2 tables

featureNames = {
    "logLength": "Log reasoning tokens",
    "reasoningTokens": "Reasoning tokens",
    "branchPer1k": "``Alternatively'' per 1,000 words",
    "sqlDrafts": "SQL drafts (SELECT count)",
    "waitPer1k": "``Wait'' per 1,000 words",
}


def buildDescriptivesTable():
    rows = readCsv(traceDir / "descriptives.csv")
    rowByKey = {}
    for row in rows:
        rowByKey[(row["feature"], row["difficulty"], row["group"])] = row

    levels = difficultyLevels + ["all"]

    lines = [
        "\\begin{tabular}{@{}ll ccccc@{}}",
        "\\toprule",
        "Feature & Group & " + " & ".join(levelHeaders) + " \\\\",
        "\\midrule",
    ]

    # how many traces sit in each cell so the reader can see how thin the easy wrong group is
    countCells = ["$n$", "right / wrong"]
    for level in levels:
        rightCount = rowByKey[("logLength", level, "right")]["n"]
        wrongCount = rowByKey[("logLength", level, "wrong")]["n"]
        countCells.append(f"{rightCount} / {wrongCount}")
    lines.append(" & ".join(countCells) + " \\\\")
    lines.append("\\midrule")

    # raw token counts are shown since they read better and the tests use the log of the same thing
    for featureName in ["reasoningTokens", "branchPer1k", "sqlDrafts", "waitPer1k"]:
        for groupName in ["right", "wrong"]:
            if groupName == "right":
                cells = [featureNames[featureName], groupName]
            else:
                cells = ["", groupName]
            for level in levels:
                row = rowByKey[(featureName, level, groupName)]
                cells.append(f"{plainNumber(row['median'])} [{plainNumber(row['q1'])}, {plainNumber(row['q3'])}]")
            lines.append(" & ".join(cells) + " \\\\")
        if featureName != "waitPer1k":
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("descriptives.tex", lines)


def buildSingleTestsTable():
    testsByFeature = indexBy(readCsv(traceDir / "tests.csv"), "feature")
    aurocByFeature = indexBy(readCsv(traceDir / "auroc.csv"), "feature")

    lines = [
        "\\begin{tabular}{@{}l rr rr rr rrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Mann--Whitney} & \\multicolumn{2}{c}{Kruskal--Wallis} & \\multicolumn{2}{c}{Spearman $\\rho$} & \\multicolumn{5}{c}{AUROC, wrong vs right} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-12}",
        "Feature & $p$ & $r_{rb}$ & $H$ & $p$ & Difficulty & Length & Easy & Med. & Hard & Extra & All \\\\",
        "\\midrule",
    ]
    for featureName in ["logLength", "branchPer1k", "sqlDrafts", "waitPer1k"]:
        testRow = testsByFeature[featureName]
        aurocRow = aurocByFeature[featureName]
        cells = [
            featureNames[featureName],
            scientificP(testRow["mannwhitney_p"]),
            f"{float(testRow['rank_biserial']):.2f}",
            f"{float(testRow['kruskal_h']):.0f}",
            scientificP(testRow["kruskal_p"]),
            f"{float(testRow['spearman_with_difficulty']):.2f}",
            f"{float(testRow['spearman_with_length']):.2f}",
        ]
        for level in difficultyLevels + ["all"]:
            cells.append(f"{float(aurocRow[f'auroc_{level}']):.2f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("singleTests.tex", lines)


def buildRegressionTable():
    rows = readCsv(traceDir / "regression.csv")
    rowByKey = {}
    for row in rows:
        rowByKey[(row["feature"], row["model"])] = row

    def oddsCell(row) -> str:
        return f"{float(row['odds_ratio']):.3f} [{float(row['or_ci_low']):.3f}, {float(row['or_ci_high']):.3f}], $p$ = {scientificP(row['p_clustered'])}"

    lines = [
        "\\begin{tabular}{@{}l ll@{}}",
        "\\toprule",
        "Feature & Difficulty + feature & Difficulty + log length + feature \\\\",
        "\\midrule",
    ]
    for featureName in ["logLength", "branchPer1k", "sqlDrafts", "waitPer1k"]:
        withoutRow = rowByKey[(featureName, "without length")]
        withCell = ""
        if (featureName, "with length") in rowByKey:
            withCell = oddsCell(rowByKey[(featureName, "with length")])
        lines.append(f"{featureNames[featureName]} & {oddsCell(withoutRow)} & {withCell} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("regression.tex", lines)


def replicationName(runName: str) -> str:
    # full-qwen3-4b-think-seed66 becomes 4B seed 66 and the run without a seed suffix is seed 42
    sizePart = runName.replace("full-qwen3-", "").split("-think")[0].upper()
    if "seed" in runName:
        seedPart = runName.split("seed")[-1]
    else:
        seedPart = "42"
    return f"{sizePart}, seed {seedPart}"


def buildReplicationTable():
    rows = readCsv(traceDir / "replication.csv")

    lines = [
        "\\begin{tabular}{@{}l rr rr rrr rrr@{}}",
        "\\toprule",
        " & & & \\multicolumn{2}{c}{Median tokens} & \\multicolumn{3}{c}{Log length} & \\multicolumn{3}{c}{$p$ beyond length} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
        "Run & EX & Unfinished & Right & Wrong & AUROC & OR & $p$ & Altern. & Drafts & Wait \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [
            replicationName(row["run"]),
            percent(row["ex"]),
            row["thinking_not_finished"],
            plainNumber(row["median_tokens_right"]),
            plainNumber(row["median_tokens_wrong"]),
            f"{float(row['auroc_length']):.3f}",
            f"{float(row['length_odds_ratio']):.3f}",
            scientificP(row["length_p"]),
            f"{float(row['branchPer1k_p_beyond_length']):.2f}",
            scientificP(row["sqlDrafts_p_beyond_length"]),
            f"{float(row['waitPer1k_p_beyond_length']):.2f}",
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("replication.tex", lines)


# task 3 tables

def buildSelfConsistencyTable():
    accuracyByRun = indexBy(readCsv(summaryDir / "accuracy.csv"), "run")
    efficiencyByRun = indexBy(readCsv(summaryDir / "efficiency.csv"), "run")
    byDifficulty = indexBy(readCsv(consistencyDir / "by_difficulty.csv"), "difficulty")

    mcnemarRows = readCsv(summaryDir / "mcnemar.csv")
    mcnemarByKey = {}
    for row in mcnemarRows:
        mcnemarByKey[(row["question"], row["metric"])] = row

    solutions = [
        ("Qwen3-4B think", "Single sample (seed 42)", None),
        ("Qwen3-4B think self consistency N=3", "EBSC, $N$ = 3", "self consistency 4b N3 vs single"),
        ("Qwen3-4B think self consistency N=5", "EBSC, $N$ = 5 (primary)", "self consistency 4b N5 vs single"),
    ]

    lines = [
        "\\begin{tabular}{@{}l rrrr rr r c c r@{}}",
        "\\toprule",
        " & \\multicolumn{5}{c}{EX} & & EX gain & Fixed / & McNemar & Tokens per \\\\",
        "\\cmidrule(lr){2-6}",
        "Solution & Easy & Med. & Hard & Extra & All & EM all & [95\\% CI] & broken & $p$ (Holm) & question \\\\",
        "\\midrule",
    ]
    for runLabel, shortName, question in solutions:
        accuracyRow = accuracyByRun[runLabel]
        cells = [shortName]
        for level in difficultyLevels + ["all"]:
            cells.append(percent(accuracyRow[f"ex_{level}"]))
        cells.append(percent(accuracyRow["em_all"]))

        if question is None:
            cells = cells + ["", "", ""]
        else:
            testRow = mcnemarByKey[(question, "ex")]
            cells.append(f"{signedPoints(testRow['difference'])} {interval(testRow['difference_ci_low'], testRow['difference_ci_high'])}")
            cells.append(f"{testRow['only_second_right']} / {testRow['only_first_right']}")
            if testRow["family"] == "rq3 primary":
                cells.append(pValue(testRow["p_value"]))
            else:
                cells.append(f"{pValue(testRow['p_value'])} ({pValue(testRow['p_holm'])})")

        cells.append(plainNumber(round(float(efficiencyByRun[runLabel]["output_tokens_mean"]))))
        lines.append(" & ".join(cells) + " \\\\")

    # the share of questions where at least one of the five samples was right
    upperCells = ["Best of 5 (upper bound)"]
    for level in difficultyLevels + ["all"]:
        upperCells.append(percent(byDifficulty[level]["upper_bound_any_right"]))
    upperCells = upperCells + ["", "", "", "", ""]
    lines.append("\\midrule")
    lines.append(" & ".join(upperCells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("selfConsistency.tex", lines)


def buildExamplesTable():
    # the cases the report walks through picked from the fixed and broken lists in selfConsistency.md
    examples = [
        (6, "Fixed"),
        (137, "Fixed"),
        (111, "Broken"),
    ]
    baselineById = {}
    for row in readJsonl(runsDir / "full-qwen3-4b-think_scored.jsonl"):
        baselineById[row["dev_idx"]] = row
    votedById = {}
    for row in readJsonl(runsDir / "full-qwen3-4b-think-sc5_scored.jsonl"):
        votedById[row["dev_idx"]] = row

    lines = [
        "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{0.2\\linewidth}>{\\raggedright\\arraybackslash}X@{}}",
        "\\toprule",
    ]
    for position, (devIdx, outcome) in enumerate(examples):
        baseline = baselineById[devIdx]
        voted = votedById[devIdx]

        if baseline["ex"] == 1:
            baselineMark = "right"
        else:
            baselineMark = "wrong"
        if voted["ex"] == 1:
            votedMark = "right"
        else:
            votedMark = "wrong"

        lines.append(f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{{outcome}, dev {devIdx} ({baseline['difficulty']}).}} {latexEscape(baseline['question'])}}} \\\\")
        lines.append(f"Gold & \\texttt{{{latexEscape(' '.join(baseline['gold'].split()))}}} \\\\")
        lines.append(f"Seed 42 ({baselineMark}) & \\texttt{{{latexEscape(baseline['pred_sql'])}}} \\\\")
        lines.append(f"Vote, {voted['sc_winner_votes']} of 5 ({votedMark}) & \\texttt{{{latexEscape(voted['pred_sql'])}}} \\\\")
        if position < len(examples) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    writeTable("examples.tex", lines)


def buildScalingTable():
    scalingRows = readCsv(consistencyDir / "scaling.csv")
    agreementRows = readCsv(consistencyDir / "agreement.csv")

    nCells = ["$N$"]
    meanCells = ["Voted EX, mean over seed subsets"]
    rangeCells = ["Range over seed subsets"]
    boundCells = ["Best of $N$ (upper bound)"]
    for row in scalingRows:
        nCells.append(row["n_samples"])
        meanCells.append(percent(row["voted_ex_mean"]))
        if row["voted_ex_min"] == row["voted_ex_max"]:
            rangeCells.append(percent(row["voted_ex_min"]))
        else:
            rangeCells.append(f"{percent(row['voted_ex_min'])} to {percent(row['voted_ex_max'])}")
        boundCells.append(percent(row["upper_bound_mean"]))

    lines = [
        "\\begin{tabular}{@{}l ccccc@{}}",
        "\\toprule",
        " & ".join(nCells) + " \\\\",
        "\\midrule",
        " & ".join(meanCells) + " \\\\",
        " & ".join(rangeCells) + " \\\\",
        " & ".join(boundCells) + " \\\\",
        "\\midrule",
        "\\multicolumn{6}{@{}l}{\\textit{Agreement with $N$ = 5: samples in the winning group}} \\\\",
    ]

    # most agreement first since that is where most questions sit
    agreementRows.reverse()
    agreementHeader = ["Agreeing samples"]
    shareCells = ["Share of questions (\\%)"]
    votedCells = ["Voted EX (\\%)"]
    for row in agreementRows:
        if int(row["agreeing_samples"]) < 1:
            continue
        agreementHeader.append(row["agreeing_samples"])
        shareCells.append(percent(row["share_of_questions"]))
        votedCells.append(percent(row["voted_ex"]))
    lines.append(" & ".join(agreementHeader) + " \\\\")
    lines.append(" & ".join(shareCells) + " \\\\")
    lines.append(" & ".join(votedCells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("scaling.tex", lines)


# appendix tables

def buildComponentsTable():
    componentsByRun = indexBy(readCsv(summaryDir / "components.csv"), "run")
    componentNames = ["select", "select(no AGG)", "where", "where(no OP)", "group(no Having)", "group", "order", "and/or", "IUEN", "keywords"]

    headerCells = ["Configuration"]
    for componentName in componentNames:
        headerCells.append(latexEscape(componentName))

    lines = [
        "\\begin{tabular}{@{}l rrrrrrrrrr@{}}",
        "\\toprule",
        " & ".join(headerCells) + " \\\\",
        "\\midrule",
    ]
    for runLabel, shortName in taskOneConfigs:
        row = componentsByRun[runLabel]
        cells = [shortName]
        for componentName in componentNames:
            cells.append(percent(row[componentName]))
        lines.append(" & ".join(cells) + " \\\\")
        if runLabel == lastCoderLabel:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("components.tex", lines)


def buildSoftwareTable():
    # versions straight from each runs colab log so the appendix shows exactly what ran
    settingsByRun = indexBy(readCsv(summaryDir / "settings.csv"), "run")

    runFiles = sorted(runsDir.glob("full-*_config.json"))

    lines = [
        "\\begin{tabular}{@{}l l l l l r@{}}",
        "\\toprule",
        "Run & Revision & transformers & PyTorch & Python & Wall time (min) \\\\",
        "\\midrule",
    ]
    for configPath in runFiles:
        runName = configPath.name.replace("_config.json", "")
        # voted runs reuse the seed 42 config so they arent separate generations
        if "-sc" in runName:
            continue
        runConfig = json.loads(configPath.read_text())
        session = runConfig["sessions"][-1]
        lines.append(
            f"\\texttt{{{latexEscape(runName)}}} & \\texttt{{{runConfig['revision']}}} & {session['transformers']} & {latexEscape(session['torch'])} & {session['python']} & {float(session['wall_sec']) / 60:.1f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    writeTable("software.tex", lines)


def buildPromptListing():
    # a listing so the template appears exactly as the code sends it with long lines wrapped
    examplePrompt = readJsonl(outDir / "prompts.jsonl")[0]

    lines = [
        "\\begin{lstlisting}",
        "[system]",
        systemPrompt,
        "",
        "[user]",
        userTemplate,
        "\\end{lstlisting}",
    ]
    writeTable("promptTemplate.tex", lines)

    exampleLines = [
        "\\begin{lstlisting}",
        "[user]",
        examplePrompt["messages"][1]["content"],
        "\\end{lstlisting}",
    ]
    writeTable("promptExample.tex", exampleLines)


def main():
    buildModelsTable()
    buildInferenceTable()
    buildAccuracyTable()
    buildMcNemarTable()
    buildEfficiencyTable()
    buildThinkingTable()
    buildEmParserTable()
    buildErrorTable()
    buildDescriptivesTable()
    buildSingleTestsTable()
    buildRegressionTable()
    buildReplicationTable()
    buildSelfConsistencyTable()
    buildExamplesTable()
    buildScalingTable()
    buildComponentsTable()
    buildSoftwareTable()
    buildPromptListing()


if __name__ == "__main__":
    main()
