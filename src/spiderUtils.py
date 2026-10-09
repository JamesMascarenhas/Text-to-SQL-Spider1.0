"""
every other script imports from here so every model sees the exact same inputs
runs on the local comp and on colab and only needs the standard library so colab needs nothing extra for it
"""

import json
import re
import sqlite3
from pathlib import Path


# paths
# everything hangs off the assignment folder so the scripts work no matter where you launch them from
projectRoot = Path(__file__).resolve().parents[1]
dataDir = projectRoot / "data" / "spider_data"
dbDir = dataDir / "database"
tablesJsonPath = dataDir / "tables.json"
evalDir = projectRoot / "eval" / "test-suite-sql-eval"
outDir = projectRoot / "outputs"


# settings shared across scripts
randomSeed = 42
difficultyLevels = ["easy", "medium", "hard", "extra"]

# goes in the prediction file when no sql could be pulled out of a reply
# it fails the parser and the database so it can never score as correct by accident
# SELECT 1 would not be safe since it matches any gold query that returns a single 1 like a count of 1
invalidSql = "SELECT no_prediction FROM no_prediction"


# file helpers

def loadJson(path: Path):
    with open(path) as jsonFile:
        data = json.load(jsonFile)

    return data


def readJsonl(path: Path) -> list:
    rows = []

    with open(path) as jsonlFile:
        for line in jsonlFile:
            # skip blank lines so a trailing newline doesnt turn into a broken row
            if line.strip() == "":
                continue

            rows.append(json.loads(line))

    return rows


def writeJsonl(rows: list, path: Path):
    # make the folder first so a fresh outputs folder doesnt crash the write
    Path(path).parent.mkdir(parents = True, exist_ok = True)

    with open(path, "w") as jsonlFile:
        for row in rows:
            jsonlFile.write(json.dumps(row) + "\n")


def getDbPath(dbId: str) -> Path:
    # spider keeps every database in its own folder named after it
    return dbDir / dbId / f"{dbId}.sqlite"


# schema text

def decodeIgnoringBadBytes(rawBytes: bytes) -> str:
    # a few spider databases have broken characters and without this sqlite crashes on them
    return rawBytes.decode(errors = "ignore")


def buildSchemaText(dbId: str, nRows: int = 0) -> str:
    """
    the schema the model sees written as the CREATE TABLE statements straight from the sqlite file
    those already carry column types and keys so the model can see how tables join
    nRows adds a few example rows per table which helps with value formatting like France vs france
    longer prompts though so whatever gets picked has to be the same for every model
    """
    connection = sqlite3.connect(getDbPath(dbId))
    connection.text_factory = decodeIgnoringBadBytes
    cursor = connection.cursor()

    # sqlite keeps its own bookkeeping tables that start with sqlite_ and the model doesnt need those
    cursor.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
    )
    tables = cursor.fetchall()

    tableBlocks = []

    for tableName, createStatement in tables:
        # some spider files have empty lines inside a statement which just waste prompt space
        createStatement = createStatement.strip()
        createStatement = re.sub(r"\n\s*\n", "\n", createStatement)

        # keep every statement ending the same way
        if not createStatement.endswith(";"):
            createStatement = createStatement + ";"

        tableBlock = createStatement

        if nRows > 0:
            cursor.execute(f'SELECT * FROM "{tableName}" LIMIT {int(nRows)}')

            columnNames = []
            for columnInfo in cursor.description:
                columnNames.append(columnInfo[0])

            exampleRows = cursor.fetchall()

            # tab separated so it reads like a little table inside a sql comment
            rowLines = ["\t".join(columnNames)]
            for exampleRow in exampleRows:
                cellTexts = []
                for cell in exampleRow:
                    cellTexts.append(str(cell))
                rowLines.append("\t".join(cellTexts))

            tableBlock = tableBlock + f"\n/*\n{len(exampleRows)} example rows from {tableName}:\n"
            tableBlock = tableBlock + "\n".join(rowLines)
            tableBlock = tableBlock + "\n*/"

        tableBlocks.append(tableBlock)

    connection.close()

    return "\n\n".join(tableBlocks)


# prompt
# this wording is frozen now since changing it after seeing results would be tuning on the eval set

systemPrompt = (
    "You are an expert in SQLite. Given a database schema and a question, "
    "write one SQLite query that answers the question."
)

userTemplate = (
    "### Database schema\n{schema}\n\n"
    "### Question\n{question}\n\n"
    "Return only the final SQL query inside a ```sql code block."
)


def buildMessages(question: str, schemaText: str) -> list:
    # stays as plain chat messages and each models own chat template turns them into its prompt format on colab
    userText = userTemplate.format(schema = schemaText, question = question)

    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": userText},
    ]

    return messages


# pulling sql out of replies

# a fenced block that may or may not be labelled sql or sqlite
codeBlockPattern = re.compile(r"```(?:sql|sqlite)?[ \t]*\n?(.*?)```", re.S | re.I)

# fallback for replies with no block where we grab from the first SELECT or WITH onward
bareSqlPattern = re.compile(r"\b(SELECT|WITH)\b.*", re.S | re.I)


def collapseToOneLine(sql: str) -> str:
    # the eval script reads one prediction per line so a newline inside a query would shift every prediction after it
    return " ".join(sql.split())


def extractSql(rawText: str, isReasoning: bool = False) -> tuple:
    """
    pulls the final query out of a raw reply and returns it with a status label
    sql comes back as None when theres nothing usable
    statuses
      code_block               found a fenced block which is the format we asked for
      fallback                 no block so took everything from the first SELECT or WITH
      truncated_in_reasoning   reasoning model never finished thinking
      no_sql                   nothing that looks like sql at all
    the status strings go into the run files so they keep their underscores
    """
    if isReasoning:
        # no closing tag means it ran out of tokens mid thought so there is no answer to grab
        # qwen3 writes the opening tag itself but some r1 distill templates put it in the prompt
        # so checking only for the closing tag works for both
        if "</think>" not in rawText:
            return None, "truncated_in_reasoning"

        # only the part after the thinking is the answer
        rawText = rawText.split("</think>")[-1]

    codeBlocks = codeBlockPattern.findall(rawText)

    if len(codeBlocks) > 0:
        # take the last block since models sometimes draft a query and then fix it
        sql = codeBlocks[-1]
        status = "code_block"
    else:
        bareMatch = bareSqlPattern.search(rawText)

        if bareMatch is None:
            return None, "no_sql"

        sql = bareMatch.group(0)
        status = "fallback"

    # keep only the first statement in case the model tacked on a second one
    sql = sql.strip()
    sql = sql.split(";")[0]
    sql = collapseToOneLine(sql)

    # an empty block counts as no answer
    if sql == "":
        return None, "no_sql"

    return sql, status