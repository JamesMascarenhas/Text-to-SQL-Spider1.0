"""
scoring for spider predictions on the mac
the official evaluation.py only prints totals but task 2 needs right or wrong for every single example
so scoreExample redoes the official per example logic with the official functions
then crossCheckWithOfficial runs the untouched official script on the same predictions and makes sure the totals agree
has to run from a terminal or with !python on colab and never inside a notebook cell
the official execution check uses asyncio.run which breaks inside jupyter

how to run
    python src/scoring.py --run outputs/runs/<run name>.jsonl
"""

import argparse
import asyncio
import copy
import subprocess
import sys
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
from exec_eval import eval_exec_match, exec_on_db, postprocess  # noqa: E402


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
    predSql of None means nothing was extracted and counts as wrong on both

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

    # does the prediction run at all
    # same execution helper the official script uses so errors are judged the same way
    dbPathText = str(getDbPath(dbId))
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

    # em compares parsed clauses after lining up column references the same way the official script does
    foreignKeyMap = loadForeignKeyMaps()[dbId]

    goldColumnUnits = build_valid_col_units(parsedGold["from"]["table_units"], schema)
    goldForCompare = rebuild_sql_col(goldColumnUnits, rebuild_sql_val(parsedGold), foreignKeyMap)

    predColumnUnits = build_valid_col_units(parsedPred["from"]["table_units"], schema)
    predForCompare = rebuild_sql_col(predColumnUnits, rebuild_sql_val(parsedPred), foreignKeyMap)

    emCorrect = int(bool(officialEvaluator.eval_exact_match(predForCompare, goldForCompare)))

    return {
        "difficulty": difficulty,
        "em": emCorrect,
        "ex": int(exCorrect),
        "em_parse_ok": parseOk,
        "exec_error": execError,
    }


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
            summary[level] = {"n": 0, "em": float("nan"), "ex": float("nan")}
            continue

        emTotal = 0
        exTotal = 0
        for row in levelRows:
            emTotal = emTotal + row["em"]
            exTotal = exTotal + row["ex"]

        summary[level] = {"n": nRows, "em": emTotal / nRows, "ex": exTotal / nRows}

    return summary


def printSummary(summary: dict, title: str = ""):
    if title != "":
        print(title)

    columns = difficultyLevels + ["all"]

    headerLine = f"{'':8}"
    countLine = f"{'n':8}"
    for level in columns:
        headerLine = headerLine + f"{level:>10}"
        countLine = countLine + f"{summary[level]['n']:>10}"

    print(headerLine)
    print(countLine)

    for metric in ["em", "ex"]:
        metricLine = f"{metric.upper():8}"
        for level in columns:
            metricLine = metricLine + f"{summary[level][metric]:>10.3f}"
        print(metricLine)


def crossCheckWithOfficial(rows: list, runTag: str, tolerance: float = 1e-3) -> dict:
    """
    runs the untouched official script on the same predictions and compares totals
    if this ever disagrees our per example scores cant be trusted so it stops everything
    """
    officialDir = outDir / "official_eval"
    officialDir.mkdir(parents = True, exist_ok = True)

    goldFilePath = officialDir / f"{runTag}_gold.sql"
    predFilePath = officialDir / f"{runTag}_pred.sql"
    logFilePath = officialDir / f"{runTag}_official.txt"

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

    command = [
        sys.executable, "evaluation.py",
        "--gold", str(goldFilePath),
        "--pred", str(predFilePath),
        "--db", str(dbDir),
        "--table", str(tablesJsonPath),
        "--etype", "all",
    ]

    # run from inside the eval folder since the script imports its neighbour files
    result = subprocess.run(command, cwd = evalDir, capture_output = True, text = True)
    logFilePath.write_text(result.stdout + "\n" + result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"official evaluation failed so check {logFilePath}")

    # the last number on each row of its printed table is the all column
    officialScores = {}
    for line in result.stdout.splitlines():
        if line.startswith("execution"):
            officialScores["ex"] = float(line.split()[-1])
        elif line.startswith("exact match"):
            officialScores["em"] = float(line.split()[-1])

    ourScores = summarizeScores(rows)["all"]

    for metric in ["em", "ex"]:
        gap = abs(officialScores[metric] - ourScores[metric])
        assert gap < tolerance, (
            f"{metric} disagrees with official {officialScores[metric]:.3f} vs ours {ourScores[metric]:.3f}"
        )

    print(
        f"cross check passed with official EM {officialScores['em']:.3f} and EX {officialScores['ex']:.3f} "
        f"and the log is in {logFilePath}"
    )

    return officialScores


def printDiagnostics(rows: list):
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

    parseFailAndCrashes = len(parseFailRows) - len(parseFailButRuns)

    print(f"invalid outputs with no sql extracted {len(rows) - len(validRows)}")
    print(f"execution errors among valid outputs {execErrorCount}")
    print(
        f"em parser failures on valid outputs {len(parseFailRows)} "
        f"with {len(parseFailButRuns)} running fine and {parseFailButRunsCorrect} of those ex correct "
        f"and {parseFailAndCrashes} crashing"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required = True, help = "jsonl of predictions from one inference run")
    args = parser.parse_args()

    runPath = Path(args.run)
    rows = readJsonl(runPath)

    # scores get added onto each row so one file holds the output and how it did
    for row in rows:
        scores = scoreExample(row.get("pred_sql"), row["gold"], row["db_id"])
        row.update(scores)

    # keeps the original run file untouched next to the scored one
    scoredPath = runPath.with_name(runPath.stem + "_scored.jsonl")
    writeJsonl(rows, scoredPath)

    printSummary(summarizeScores(rows), title = f"{args.run}  ({len(rows)} examples)")
    printDiagnostics(rows)

    runTag = scoredPath.stem.replace("_scored", "")
    crossCheckWithOfficial(rows, runTag)

    print(f"per example scores written to {scoredPath}")


if __name__ == "__main__":
    main()