"""
scoring for spider predictions on the mac
the official evaluation.py only prints totals but task 2 needs right or wrong for every single example
so scoreExample redoes the official per example logic with the official functions
then crossCheckWithOfficial runs the untouched official script on the same predictions and makes sure the totals agree
also collects the papers component matching scores from the official output and saves everything to a summary file
has to run from a terminal or with !python on colab and never inside a notebook cell
the official execution check uses asyncio.run which breaks inside jupyter

how to run
    python src/scoring.py --run outputs/runs/<run name>.jsonl
"""

import argparse
import asyncio
import copy
import json
import sqlite3
import subprocess
import sys
import time
import warnings
from pathlib import Path

import nltk

from spiderUtils import (
    dbDir,
    difficultyLevels,
    evalDir,
    getDbPath,
    invalidSql,
    outDir,
    readJsonl,
    tablesJsonPath,
    writeJsonl,
)


# the official parser splits sql into tokens with nltk which needs this data file
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet = True)

# the eval repo is older than python 3.12 and spams harmless regex warnings
warnings.filterwarnings("ignore", category = SyntaxWarning)

# the eval repo isnt a package so python only finds it if its folder is on the import path
sys.path.insert(0, str(evalDir))

from process_sql import Schema, get_schema, get_sql  # noqa: E402
from evaluation import (  # noqa: E402
    Evaluator,
    build_foreign_key_map_from_json,
    build_valid_col_units,
    rebuild_sql_col,
    rebuild_sql_val,
)
from exec_eval import eval_exec_match, exec_on_db, postprocess, replace_cur_year  # noqa: E402
from parse import get_all_preds_for_execution, remove_distinct  # noqa: E402


# what the official script swaps in when a prediction cant be parsed
# copied exactly so unparseable predictions get treated the same way here
emptyParsedSql = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}

officialEvaluator = Evaluator()

# the gold value check tries every way of filling a predictions value slots with the gold values
# that grows exponentially so a messy prediction with lots of literals can need trillions of tries
# past this many combinations we skip the plugging and fall back to plain ex for that example
maxValueCombinations = 1000

# the official execution has a 60 second timeout that never actually fires
# it wraps a blocking sqlite call in asyncio.wait_for and the blocking call never gives the timer a chance to check
# so a query that never finishes like a join with no join condition hangs everything including the official script
# every prediction gets one run of our own first with a real limit using sqlite's progress handler
# every gold query in spider finishes in well under a second so 30 seconds is generous
predTimeLimitSeconds = 30

# plugging reruns the prediction once per combination so slow queries times many combinations also add up
# skip plugging when that estimate goes past this
pluggingBudgetSeconds = 120

# loaded once and reused since reading every database schema over and over is slow
schemaCache = {}
foreignKeyMaps = None


def loadSchema(dbId: str):
    if dbId not in schemaCache:
        dbPathText = str(getDbPath(dbId))
        schemaCache[dbId] = Schema(get_schema(dbPathText))

    return schemaCache[dbId]


def loadForeignKeyMaps() -> dict:
    global foreignKeyMaps

    if foreignKeyMaps is None:
        foreignKeyMaps = build_foreign_key_map_from_json(str(tablesJsonPath))

    return foreignKeyMaps


def decodeIgnoringBadBytes(rawBytes: bytes) -> str:
    # same as the official connection so broken characters are handled the same way
    return rawBytes.decode(errors = "ignore")


def timePrediction(dbPathText: str, predText: str) -> tuple:
    """
    runs the prediction once with a real time limit and says how it went and how long it took
    status is finished or timed_out or error
    the same cleanup as the official execution so the query that runs here is the one it would run
    """
    query = replace_cur_year(postprocess(predText))

    connection = sqlite3.connect(dbPathText)
    connection.text_factory = decodeIgnoringBadBytes

    deadline = time.monotonic() + predTimeLimitSeconds

    # sqlite calls this every few thousand steps and stops the query as soon as it returns nonzero
    def pastDeadline():
        if time.monotonic() > deadline:
            return 1
        return 0

    connection.set_progress_handler(pastDeadline, 10000)

    startTime = time.monotonic()
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        cursor.fetchall()
        status = "finished"
    except sqlite3.OperationalError as error:
        # interrupted is what sqlite says when the progress handler stops it
        if "interrupted" in str(error):
            status = "timed_out"
        else:
            status = "error"
    except Exception:
        status = "error"
    finally:
        connection.close()

    return status, time.monotonic() - startTime


def countValueCombinations(predText: str, goldSql: str) -> int:
    # same cleanup the official execution check does before it starts plugging so the count matches what it would try
    predClean = remove_distinct(postprocess(predText))
    goldClean = remove_distinct(postprocess(goldSql))

    nCombinations, _ = get_all_preds_for_execution(goldClean, predClean)

    return nCombinations


def getDifficulty(goldSql: str, dbId: str) -> str:
    # the official label so our difficulty buckets match every other paper on spider
    schema = loadSchema(dbId)
    parsedGold = get_sql(schema, goldSql)

    return officialEvaluator.eval_hardness(parsedGold)


def scoreExample(predSql, goldSql: str, dbId: str) -> dict:
    """
    exact set match and execution accuracy for one example
    same default settings as the official script
    values ignored for em and DISTINCT dropped for ex and no gold values plugged in
    predSql of None means nothing was extracted and counts as wrong on everything

    ex_gold_values is execution accuracy the way the spider paper defines it
    the model gets the gold values and only the structure is judged
    our models write their own values so plain ex is the main number and this one is an upper bound
    ex_gold_values_capped is true when plugging was skipped because of too many combinations or too slow a query
    those examples just keep their plain ex result so for them its a lower bound instead
    timed_out is true when the prediction didnt finish within predTimeLimitSeconds
    those count as wrong on ex and skip the official execution entirely since it would hang on them

    two extra checks the official script doesnt do
      em_parse_ok   false when spiders 2018 parser couldnt read the prediction
      exec_error    None when the prediction runs and otherwise the database error
    together they split parser failures into queries that actually run which is probably just newer syntax
    and queries that are really broken like ones using a column that doesnt exist
    the dict keys go into the scored files so they keep their underscores
    """
    schema = loadSchema(dbId)
    parsedGold = get_sql(schema, goldSql)
    difficulty = officialEvaluator.eval_hardness(parsedGold)

    if predSql is None:
        return {
            "difficulty": difficulty,
            "em": 0,
            "ex": 0,
            "em_parse_ok": False,
            "exec_error": "no prediction",
            "ex_gold_values": 0,
            "ex_gold_values_capped": False,
            "timed_out": False,
        }

    # the official script does this swap because old models wrote the word value as a placeholder
    # copying it so our numbers match theirs exactly
    predText = predSql.replace("value", "1")

    try:
        parsedPred = get_sql(schema, predText)
        parseOk = True
    except Exception:
        parsedPred = copy.deepcopy(emptyParsedSql)
        parseOk = False

    # em compares parsed clauses after lining up column references the same way the official script does
    foreignKeyMap = loadForeignKeyMaps()[dbId]

    goldColumnUnits = build_valid_col_units(parsedGold["from"]["table_units"], schema)
    goldForCompare = rebuild_sql_col(goldColumnUnits, rebuild_sql_val(parsedGold), foreignKeyMap)

    predColumnUnits = build_valid_col_units(parsedPred["from"]["table_units"], schema)
    predForCompare = rebuild_sql_col(predColumnUnits, rebuild_sql_val(parsedPred), foreignKeyMap)

    emCorrect = int(bool(officialEvaluator.eval_exact_match(predForCompare, goldForCompare)))


    # one timed run of our own first since the official execution would hang forever on a query that never finishes
    dbPathText = str(getDbPath(dbId))
    runStatus, runSeconds = timePrediction(dbPathText, predText)

    if runStatus == "timed_out":
        return {
            "difficulty": difficulty,
            "em": emCorrect,
            "ex": 0,
            "em_parse_ok": parseOk,
            "exec_error": f"timed out after {predTimeLimitSeconds} seconds",
            "ex_gold_values": 0,
            "ex_gold_values_capped": True,
            "timed_out": True,
        }

    # does the prediction run at all
    # same execution helper the official script uses so errors are judged the same way
    execFlag, execResult = asyncio.run(exec_on_db(dbPathText, postprocess(predText)))

    if execFlag == "result":
        execError = None
    else:
        execError = str(execResult)

    exCorrect = eval_exec_match(
        db = dbPathText,
        p_str = predText,
        g_str = goldSql,
        plug_value = False,
        keep_distinct = False,
        progress_bar_for_each_datapoint = False,
    )

    # the official helper runs the prediction as written first and then every way of filling its value slots with gold values
    # it counts as correct if any of those match so its lenient on purpose
    nCombinations = countValueCombinations(predText, goldSql)
    estimatedPluggingSeconds = nCombinations * runSeconds

    if nCombinations > maxValueCombinations or estimatedPluggingSeconds > pluggingBudgetSeconds:
        exGoldValuesCorrect = exCorrect
        goldValuesCapped = True
    else:
        exGoldValuesCorrect = eval_exec_match(
            db = dbPathText,
            p_str = predText,
            g_str = goldSql,
            plug_value = True,
            keep_distinct = False,
            progress_bar_for_each_datapoint = False,
        )
        goldValuesCapped = False

    return {
        "difficulty": difficulty,
        "em": emCorrect,
        "ex": int(exCorrect),
        "em_parse_ok": parseOk,
        "exec_error": execError,
        "ex_gold_values": int(exGoldValuesCorrect),
        "ex_gold_values_capped": goldValuesCapped,
        "timed_out": False,
    }


# the per example accuracy fields and how they get labelled when printed
accuracyMetrics = ["em", "ex", "ex_gold_values"]
metricLabels = {"em": "EM", "ex": "EX", "ex_gold_values": "EX gold vals"}


def summarizeScores(rows: list) -> dict:
    # accuracy for each difficulty level plus all of them together
    rowsByLevel = {}
    for level in difficultyLevels + ["all"]:
        rowsByLevel[level] = []

    for row in rows:
        rowsByLevel[row["difficulty"]].append(row)
        rowsByLevel["all"].append(row)

    summary = {}

    for level in difficultyLevels + ["all"]:
        levelRows = rowsByLevel[level]
        nRows = len(levelRows)

        # nan instead of crashing when a level is empty like in tiny test runs
        if nRows == 0:
            summary[level] = {"n": 0}
            for metric in accuracyMetrics:
                summary[level][metric] = float("nan")
            continue

        summary[level] = {"n": nRows}
        for metric in accuracyMetrics:
            metricTotal = 0
            for row in levelRows:
                metricTotal = metricTotal + row[metric]
            summary[level][metric] = metricTotal / nRows

    return summary


def printSummary(summary: dict, title: str = ""):
    if title != "":
        print(title)

    columns = difficultyLevels + ["all"]

    headerLine = f"{'':14}"
    countLine = f"{'n':14}"
    for level in columns:
        headerLine = headerLine + f"{level:>10}"
        countLine = countLine + f"{summary[level]['n']:>10}"

    print(headerLine)
    print(countLine)

    for metric in accuracyMetrics:
        metricLine = f"{metricLabels[metric]:14}"
        for level in columns:
            metricLine = metricLine + f"{summary[level][metric]:>10.3f}"
        print(metricLine)


def runOfficialScript(goldFilePath, predFilePath, logFilePath, extraArgs: list) -> str:
    command = [
        sys.executable, "evaluation.py",
        "--gold", str(goldFilePath),
        "--pred", str(predFilePath),
        "--db", str(dbDir),
        "--table", str(tablesJsonPath),
    ]
    command = command + extraArgs

    # run from inside the eval folder since the script imports its neighbour files
    result = subprocess.run(command, cwd = evalDir, capture_output = True, text = True)
    logFilePath.write_text(result.stdout + "\n" + result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"official evaluation failed so check {logFilePath}")

    return result.stdout


def readMetricRow(officialOutput: str, rowStart: str) -> float:
    # the last number on each row of the printed table is the all column
    for line in officialOutput.splitlines():
        if line.startswith(rowStart):
            return float(line.split()[-1])

    raise ValueError(f"no {rowStart} row in the official output")


def readComponentMatching(officialOutput: str) -> dict:
    """
    pulls the papers component matching tables out of the official printout
    the official script calls it partial matching and prints accuracy then recall then f1
    each line is a component name followed by easy medium hard extra all
    some names have spaces like select(no AGG) so the name is everything before the last five numbers
    """
    sectionMarkers = {
        "PARTIAL MATCHING ACCURACY": "accuracy",
        "PARTIAL MATCHING RECALL": "recall",
        "PARTIAL MATCHING F1": "f1",
    }
    columns = difficultyLevels + ["all"]

    componentMatching = {}
    currentSection = None

    for line in officialOutput.splitlines():
        matchedMarker = None
        for marker, sectionName in sectionMarkers.items():
            if marker in line:
                matchedMarker = sectionName
        if matchedMarker is not None:
            currentSection = matchedMarker
            componentMatching[currentSection] = {}
            continue

        if currentSection is None:
            continue

        tokens = line.split()

        # a blank line or a new banner ends the section
        if len(tokens) < len(columns) + 1 or line.startswith("="):
            currentSection = None
            continue

        componentName = " ".join(tokens[:-len(columns)])
        values = tokens[-len(columns):]

        componentMatching[currentSection][componentName] = {}
        for level, value in zip(columns, values):
            componentMatching[currentSection][componentName][level] = float(value)

    return componentMatching


def writeOfficialInputs(rows: list, officialDir: Path, fileTag: str) -> tuple:
    goldFilePath = officialDir / f"{fileTag}_gold.sql"
    predFilePath = officialDir / f"{fileTag}_pred.sql"

    # one query per line in the same order in both files since thats how the script pairs them up
    with open(goldFilePath, "w") as goldFile:
        for row in rows:
            goldFile.write(f"{row['gold']}\t{row['db_id']}\n")

    with open(predFilePath, "w") as predFile:
        for row in rows:
            predLine = row["pred_sql"]
            # empty counts as missing too so the script never sees a blank line and loses its place
            if predLine is None or predLine == "":
                predLine = invalidSql
            predFile.write(f"{predLine}\n")

    return goldFilePath, predFilePath


def crossCheckWithOfficial(rows: list, runTag: str, tolerance: float = 1e-3) -> dict:
    """
    runs the untouched official script on the same predictions and compares totals
    once as normal for em and ex and component matching and once with gold values plugged in
    if either ever disagrees our per example scores cant be trusted so it stops everything
    """
    officialDir = outDir / "official_eval"
    officialDir.mkdir(parents = True, exist_ok = True)

    # the official run would hang on predictions that never finish so those are left out of this check
    finishedRows = []
    for row in rows:
        if not row["timed_out"]:
            finishedRows.append(row)

    nTimedOut = len(rows) - len(finishedRows)

    goldFilePath, predFilePath = writeOfficialInputs(finishedRows, officialDir, runTag)
    standardOutput = runOfficialScript(
        goldFilePath, predFilePath, officialDir / f"{runTag}_official.txt", ["--etype", "all"]
    )

    # the official plug check has no cap so it would hang on the same examples ours skips
    # so this cross check runs on every example except the capped ones and compares like with like
    uncappedRows = []
    for row in rows:
        if not row["ex_gold_values_capped"]:
            uncappedRows.append(row)

    nCapped = len(rows) - len(uncappedRows)

    goldValuesGoldPath, goldValuesPredPath = writeOfficialInputs(uncappedRows, officialDir, f"{runTag}_gold_values")
    goldValuesOutput = runOfficialScript(
        goldValuesGoldPath, goldValuesPredPath, officialDir / f"{runTag}_official_gold_values.txt",
        ["--etype", "exec", "--plug_value"],
    )

    officialScores = {
        "em": readMetricRow(standardOutput, "exact match"),
        "ex": readMetricRow(standardOutput, "execution"),
        "ex_gold_values_uncapped": readMetricRow(goldValuesOutput, "execution"),
    }

    ourFinishedScores = summarizeScores(finishedRows)["all"]
    ourUncappedScores = summarizeScores(uncappedRows)["all"]

    checks = [
        ("em", officialScores["em"], ourFinishedScores["em"]),
        ("ex", officialScores["ex"], ourFinishedScores["ex"]),
        ("ex_gold_values on uncapped examples", officialScores["ex_gold_values_uncapped"], ourUncappedScores["ex_gold_values"]),
    ]

    for metricName, officialValue, ourValue in checks:
        gap = abs(officialValue - ourValue)
        assert gap < tolerance, f"{metricName} disagrees with official {officialValue:.3f} vs ours {ourValue:.3f}"

    officialScores["component_matching"] = readComponentMatching(standardOutput)
    officialScores["n_gold_values_capped"] = nCapped
    officialScores["n_timed_out"] = nTimedOut

    print(
        f"cross check passed with official EM {officialScores['em']:.3f} and EX {officialScores['ex']:.3f} "
        f"on the {len(finishedRows)} examples that finish with {nTimedOut} timed out "
        f"and EX with gold values {officialScores['ex_gold_values_uncapped']:.3f} on the {len(uncappedRows)} uncapped examples "
        f"with {nCapped} capped"
    )

    return officialScores


def printComponentF1(componentMatching: dict):
    # the paper reports f1 per component so thats the one worth eyeballing
    print("component matching f1 over all examples")
    for componentName, scoresByLevel in componentMatching["f1"].items():
        print(f"  {componentName:18}{scoresByLevel['all']:>8.3f}")


def countDiagnostics(rows: list) -> dict:
    # breaks parser failures down so the em vs ex gap can be explained in the report
    validRows = []
    for row in rows:
        if row.get("pred_sql") is not None:
            validRows.append(row)

    execErrorCount = 0
    parseFailRows = []
    for row in validRows:
        if row["exec_error"] is not None:
            execErrorCount = execErrorCount + 1
        if not row["em_parse_ok"]:
            parseFailRows.append(row)

    parseFailButRuns = []
    for row in parseFailRows:
        if row["exec_error"] is None:
            parseFailButRuns.append(row)

    parseFailButRunsCorrect = 0
    for row in parseFailButRuns:
        parseFailButRunsCorrect = parseFailButRunsCorrect + row["ex"]

    cappedCount = 0
    timedOutCount = 0
    for row in rows:
        if row["ex_gold_values_capped"]:
            cappedCount = cappedCount + 1
        if row["timed_out"]:
            timedOutCount = timedOutCount + 1

    return {
        "n_examples": len(rows),
        "gold_values_capped": cappedCount,
        "timed_out": timedOutCount,
        "invalid_outputs": len(rows) - len(validRows),
        "execution_errors": execErrorCount,
        "em_parse_failures": len(parseFailRows),
        "em_parse_failures_that_run": len(parseFailButRuns),
        "em_parse_failures_that_run_and_are_ex_correct": parseFailButRunsCorrect,
        "em_parse_failures_that_crash": len(parseFailRows) - len(parseFailButRuns),
    }


def printDiagnostics(diagnostics: dict):
    print(f"invalid outputs with no sql extracted {diagnostics['invalid_outputs']}")
    print(f"predictions that didnt finish within {predTimeLimitSeconds} seconds and count as wrong {diagnostics['timed_out']}")
    print(f"examples where gold value plugging was skipped for too many combinations or too slow a query {diagnostics['gold_values_capped']}")
    print(f"execution errors among valid outputs {diagnostics['execution_errors']}")
    print(
        f"em parser failures on valid outputs {diagnostics['em_parse_failures']} "
        f"with {diagnostics['em_parse_failures_that_run']} running fine and "
        f"{diagnostics['em_parse_failures_that_run_and_are_ex_correct']} of those ex correct "
        f"and {diagnostics['em_parse_failures_that_crash']} crashing"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required = True, help = "jsonl of predictions from one inference run")
    args = parser.parse_args()

    runPath = Path(args.run)
    rows = readJsonl(runPath)

    # scores get added onto each row so one file holds the output and how it did
    for position, row in enumerate(rows):
        scores = scoreExample(row.get("pred_sql"), row["gold"], row["db_id"])
        row.update(scores)

        # a heartbeat every 100 examples so a slow run never looks frozen
        if (position + 1) % 100 == 0:
            print(f"  scored {position + 1} of {len(rows)}")

    # keeps the original run file untouched next to the scored one
    scoredPath = runPath.with_name(runPath.stem + "_scored.jsonl")
    writeJsonl(rows, scoredPath)

    scoreSummary = summarizeScores(rows)
    diagnostics = countDiagnostics(rows)

    printSummary(scoreSummary, title = f"{args.run}  ({len(rows)} examples)")
    printDiagnostics(diagnostics)

    runTag = scoredPath.stem.replace("_scored", "")
    officialScores = crossCheckWithOfficial(rows, runTag)
    printComponentF1(officialScores["component_matching"])

    # one file per run with every number the report needs so summarizing never has to rescore
    runSummary = {
        "run": runTag,
        "scores_by_difficulty": scoreSummary,
        "component_matching": officialScores["component_matching"],
        "diagnostics": diagnostics,
    }
    summaryPath = runPath.with_name(runPath.stem + "_summary.json")
    summaryPath.write_text(json.dumps(runSummary, indent = 2))

    print(f"per example scores written to {scoredPath} and the summary to {summaryPath}")


if __name__ == "__main__":
    main()