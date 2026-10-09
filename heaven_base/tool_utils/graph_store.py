"""The graph backend behind KnowledgeGraphBuilder — `GRAPH_BACKEND=neo4j|kuzu`.

THE EMBEDDED ENGINE IS LADYBUG, PINNED `ladybug==0.21.2`. Ladybug is the continuation of Kuzu
(upstream Kuzu was archived at 0.11.3): same Cypher dialect, same Python API surface this file uses
(`Database(path)`, `Connection(db)`, `execute(q, params)`, `has_next`, `get_next`,
`get_column_names`), and the same error texts the schema-on-demand regexes parse. Only the import
changed — `import ladybug`, never a fallback to `kuzu`, because a fallback would run an engine
nobody pinned. The canonical record of the decision, the upgrade rule and the migration procedure
is `application/carton-saas/kuzu-port/LADYBUG.md`.

`GRAPH_BACKEND=kuzu` STAYS THE ONE VALUE, WITH NO `ladybug` ALIAS. The value names the backend —
the embedded, single-writer, Kuzu-dialect graph this file adapts to — not the pip package that
serves it, and every box, `system_config.sh` and test already says `kuzu`. An alias would be a
second spelling of one choice, and `resolve_backend` exists so there is exactly one place and one
word for it. The class names (`KuzuStore`, `KuzuHttpStore`) and the `KUZU_*` env names stay for
the same reason.

THREE LADYBUG LAWS THIS FILE ENFORCES, each pinned by a test in `tests/test_graph_store.py`:
  1. CONNECTION RENEWAL. Ladybug's Python client caches a prepared statement per connection for
     every parameterized query, keyed on (query text, parameter shape), and caches it even when
     preparation FAILED. So after any schema change the connection is replaced — see
     `KuzuStore._renew_connection`.
  2. BOTH KEY SPELLINGS. Ladybug returns a node/rel/path dict's internal keys upper-case
     (`_ID _LABEL _SRC _DST _NODES _RELS`); kuzu returned them lower-case. This file passes those
     dicts through untouched (`_rows` reads column names only), so the law binds their READERS —
     today one, `CartOnUtils._relationship_type_path` — which accept both spellings. A reader
     that names one spelling gets an empty answer from the other engine, never an error.
  3. NO READ-ONLY OPEN. A second process's `read_only` open SUCCEEDS on ladybug while the writer
     holds the file, and serves a stale snapshot. `KUZU_READ_ONLY` therefore refuses; readers ask
     the owner over `KuzuHttpStore`.

WHY THIS FILE EXISTS, AND WHY IT IS HERE RATHER THAN IN carton-mcp. The carton-saas tenant box
runs an embedded graph instead of a JVM sidecar (see application/carton-saas/kuzu-port/). The
plan for that port named `CartOnUtils._execute_neo4j_query` as "the one execution point". It is
not. Measured 2026-08-12 across the tree:

    KnowledgeGraphBuilder.execute_query   139 call sites / 15 files   the WRITE path + most reads
    CartOnUtils._execute_neo4j_query       11 call sites /  2 files   the MCP read facade only

An adapter at the second one covers reads and silently misses every write — every property set,
every CartonObj fence edit, every schema registration, every relationship delete, the whole queue
drain. So the seam is the CLASS, and the class lives here. Every consumer either constructs a
KnowledgeGraphBuilder, takes the `get_shared_graph()` singleton, or is handed one as
`shared_connection`, which is what makes this placement work: route the class's own method at a
backend and all 139 sites keep working untouched.

⛔ THE NEO4J PATH MUST STAY BYTE-IDENTICAL. It is what Isaac's local carton runs on (643,090
nodes). `Neo4jStore.execute` is therefore a verbatim copy of what `execute_query` did before this
file existed — `[dict(record) for record in session.run(...)]` — values included, which means a
`RETURN c` still hands back a live `neo4j.graph.Node` exactly as it always did. Serialization is
NOT done here; it stays in the read facade that always owned it.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The :Wiki table's BASE columns, MEASURED against the full live graph (643,090 nodes,
# 2026-08-12) rather than guessed: the five that are effectively universal plus the seven the
# system manages. They are declared up front because they are on nearly every node and deserve
# real types. The other 1,223 measured keys are NOT enumerated here — 817 of them appear on
# exactly one node, and `set_properties` accepts any non-reserved key by design, so the set is
# unbounded going forward. Those arrive as columns ON DEMAND (see below).
WIKI_FIXED_COLUMNS = (
    ("n", "STRING"),                 # 643090 — every node
    ("linked", "BOOLEAN"),           # 643090 — every node
    ("d", "STRING"),                 # 608477
    ("t", "TIMESTAMP"),              # 592750
    ("c", "STRING"),                 # 486074
    ("last_modified", "STRING"),     # 299101
    ("score", "DOUBLE"),             # 233564
    ("source", "STRING"),            # 210989
    ("timeline_linked", "BOOLEAN"),  # 91550
    ("region", "STRING"),            # 73733
    ("odyssey_linked", "BOOLEAN"),   # 41170
    ("soma_region", "STRING"),       # 14082
)

# ⛔ THERE IS NO OVERFLOW COLUMN, and the reason is a measurement rather than a preference.
#
# The first design put the scratch lane in one JSON column, since the key set is unbounded. That
# works only if every scratch property is read and written through the sanctioned property
# surface. It is not: `sm_gate.py` touches `s.status` — a scratch key — in SIX places of RAW
# CYPHER, including the retrieval gate's own lock (`WHERE s.status = 'locked'`,
# `SET s.status = 'locked', s.equipped_sm_id = $sm_id, s.sm_chain_index = 0`). Against a JSON
# column those would silently match nothing, and the state machine that gates carton retrieval
# would fail in exactly the quiet way this port keeps finding.
#
# So properties are REAL COLUMNS, added ON DEMAND — the same mechanism already proven for
# dynamically-minted relationship tables, extended to properties. Arbitrary Cypher then simply
# works, with no rewriting, no python-side filtering, and no cost asymmetry against neo4j.
#
# Types are declared from a MEASURED map (below) because kuzu needs one at ALTER time; anything
# unmeasured defaults to STRING and a genuine mismatch fails loudly at the SET rather than
# silently storing the wrong thing.
PROPERTY_TYPES = {
    # measured on the live graph — node properties beyond the fixed twelve
    "status": "STRING", "canonical_intent": "STRING", "session": "STRING",
    "convo_start": "BOOLEAN", "sophia_checked": "BOOLEAN", "superseded": "BOOLEAN",
    "is_async": "BOOLEAN", "cb_x": "DOUBLE", "cb_y": "DOUBLE", "cb_encoded": "STRING",
    "tk_order": "INT64", "equipped_sm_id": "STRING", "sm_chain_index": "INT64",
    # measured on the live graph — relationship properties (ts is on 3.5M edges)
    "ts": "TIMESTAMP", "weight": "DOUBLE", "order": "INT64", "reason": "STRING",
    "compression_type": "STRING", "soma_composed": "BOOLEAN", "prop": "STRING",
    "source_type": "STRING", "renamed_from": "STRING", "copied_from": "STRING",
    "tk": "STRING", "warrant": "DOUBLE", "warrant_at": "TIMESTAMP",
    "required_pattern": "STRING",
}
DEFAULT_PROPERTY_TYPE = "STRING"


def _engine():
    """The embedded engine module. Imported lazily so neo4j-only installs need no ladybug."""
    import ladybug

    return ladybug


class GraphStore:
    """A graph backend. One method matters: `execute(query, params) -> list[dict]`.

    THE PROPERTY SURFACE IS THE SECOND THING A BACKEND MUST OWN, and it is not obvious why until
    you look at what the property functions actually do: they put an ARBITRARY, USER-SUPPLIED KEY
    where Cypher expects an IDENTIFIER —

        SET c += $props                      (any keys at all)
        WHERE c.`status` = $w_0              (the key is part of the query text)
        REMOVE c.`tk_lane`

    On neo4j that is fine, because a node takes any property. On a schema-full backend the column
    has to exist first, and the two engines do not even spell the operations the same way — kuzu
    has no `SET c += $map` and no `REMOVE` at all. So the three property operations are asked of
    the store rather than spelled as Cypher by the caller, and each backend answers in its own
    dialect.
    """

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def set_properties(self, concept_name: str, properties: Dict[str, Any]) -> None:
        """SET the given properties on an EXISTING concept. Never creates a node."""
        raise NotImplementedError

    def remove_properties(self, concept_name: str, keys: List[str]) -> None:
        """REMOVE the given property keys from an existing concept."""
        raise NotImplementedError

    def find_by_properties(self, where: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Concepts whose properties match every key/value in `where` (AND).

        Returns each match's `n` plus the value of every key in `where`.
        """
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def driver(self):
        """The raw driver, or None for backends that have none.

        Two places in the tree historically reached past the class for `.driver.session()`
        (carton_utils' read facade and the observation worker's GIINT path resolver). Both now go
        through `execute`, but the attribute is kept so that anything still reaching for it gets a
        clear None rather than an AttributeError halfway through a query.
        """
        return None


class Neo4jStore(GraphStore):
    """The neo4j backend — deliberately unchanged behaviour, down to the returned value shape."""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase  # imported here so kuzu-only installs need no neo4j

        self.uri = uri
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=5.0,
            max_connection_lifetime=30,
        )

    @property
    def driver(self):
        return self._driver

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # VERBATIM the pre-adapter body of KnowledgeGraphBuilder.execute_query. `dict(record)`
        # and NOT a serialization pass: a `RETURN c` must keep handing back a live Node, because
        # at least one caller (add_concept_tool.py:3473) consumes it that way.
        with self._driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    # The three property operations, spelled as the EXACT Cypher carton_utils issued before this
    # surface existed — same clauses, same backtick quoting, same parameterization. Moving them
    # here is a relocation, not a rewrite; the live graph must not be able to tell.
    def set_properties(self, concept_name: str, properties: Dict[str, Any]) -> None:
        self.execute("MATCH (c:Wiki {n: $n}) SET c += $props",
                     {"n": concept_name, "props": properties})

    def remove_properties(self, concept_name: str, keys: List[str]) -> None:
        clauses = ", ".join(f"c.`{k}`" for k in keys)
        self.execute(f"MATCH (c:Wiki {{n: $n}}) REMOVE {clauses}", {"n": concept_name})

    def find_by_properties(self, where: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        where_clauses = " AND ".join(f"c.`{k}` = $w_{i}" for i, k in enumerate(where))
        params: Dict[str, Any] = {f"w_{i}": v for i, (k, v) in enumerate(where.items())}
        params["lim"] = limit
        return_props = ", ".join(f"c.`{k}` AS `{k}`" for k in where)
        return self.execute(
            f"MATCH (c:Wiki) WHERE {where_clauses} RETURN c.n AS n, {return_props} LIMIT $lim",
            params,
        )

    def close(self) -> None:
        self._driver.close()


class KuzuStore(GraphStore):
    """The embedded backend — ladybug, pinned `ladybug==0.21.2` (Kuzu's continuation).

    Three things make carton's shape work on a schema-full engine:

    THE SCHEMA is created on open — the twelve measured columns plus one JSON `props` column for
    the unbounded scratch lane.

    THE REL TABLES ARE CREATED ON DEMAND. carton mints relationship types dynamically
    (`[r:{rel_type.upper()}]`), and kuzu requires each to be declared. Rather than pre-declaring a
    list that would be wrong the moment someone uses a new predicate, a query that fails on an
    unknown rel table gets that table created and is retried exactly once. The dialect probe
    proved runtime `CREATE REL TABLE` works (12/12, and it is the crux of the whole port).

    THE CONNECTION IS RENEWED AFTER EVERY SCHEMA CHANGE. Ladybug's client keeps an implicit
    prepared-statement cache per connection, keyed on (query text, parameter shape), and stores
    the entry even when preparation FAILED. Retrying a parameterized query on the same connection
    after the missing table or column was created replays the cached failure; a query that
    succeeded before an `ALTER TABLE ... ADD` keeps its old plan and silently omits the new column
    from `RETURN c`. A fresh connection has an empty cache, so `execute` opens one after any
    schema fix and after any successful `CREATE|ALTER|DROP [NODE|REL] TABLE`.
    """

    # kuzu names what is missing in its error text; these pull it back out so it can be created.
    _MISSING_TABLE = re.compile(
        r"(?:Table|table)\s+(\w+)\s+does not exist|Binder exception:.*?(\w+) does not exist",
    )
    # "Binder exception: Cannot find property ts for rel." — the property, and the VARIABLE it was
    # addressed through (which is what tells us whether to alter a node table or a rel table).
    _MISSING_PROPERTY = re.compile(
        r"Cannot find property (\w+) for (\w+)\.?", re.IGNORECASE)
    # "Binder exception: Cannot bind Rule as a node pattern label." — a missing NODE table, and
    # it must NEVER be repaired as a relationship table. Kuzu needs a PRIMARY KEY to create a
    # node table and there is nothing in the error to derive one from, so this is not repairable
    # on demand; what matters is that the attempt is refused rather than guessed. A guess here
    # is worse than the error: it creates a REL table under that name, the real node table can
    # then never be created, and the failure moves to a later, stranger place. Measured
    # 2026-08-17 — a code-graph parse asked for `:Rule` and got `REL Rule(FROM Wiki TO Wiki)`.
    _MISSING_NODE_TABLE = re.compile(
        r"Cannot bind (\w+) as a node pattern label", re.IGNORECASE)

    # A statement after which every cached plan on the connection may be wrong.
    _SCHEMA_DDL = re.compile(r"^\s*(CREATE|ALTER|DROP)\s+(NODE\s+|REL\s+)?TABLE\b", re.IGNORECASE)

    def __init__(self, db_path: str):
        # NO read_only PARAMETER. The process that opens the file owns it and writes; any other
        # process reads through KuzuHttpStore. A read_only open from a second process succeeds on
        # ladybug and serves a stale snapshot, so it is not offered at all (see make_store).
        engine = _engine()
        self._engine = engine
        self._db = engine.Database(db_path)
        self._conn = engine.Connection(self._db)
        # REENTRANT deliberately: the read-modify-write property paths hold the lock across
        # several `execute` calls, and `execute` takes it too. A plain Lock deadlocks there.
        self._lock = threading.RLock()
        self._known_rel_tables: set[str] = set()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cols = ", ".join(f"{name} {ktype}" for name, ktype in WIKI_FIXED_COLUMNS)
        self._conn.execute(
            f"CREATE NODE TABLE IF NOT EXISTS Wiki({cols}, PRIMARY KEY (n))")
        self._renew_connection()

    def _renew_connection(self) -> None:
        """Replace the connection, dropping its prepared-statement cache with it.

        Closing the old one is what frees the cached statements (ladybug's `close` destroys each
        cached C++ prepared statement); dropping the reference alone leaks them.
        """
        old, self._conn = self._conn, self._engine.Connection(self._db)
        try:
            old.close()
        except Exception as exc:  # noqa: BLE001 - the new connection is already in place
            logger.debug("kuzu: closing the replaced connection failed: %s", exc)

    def _create_rel_table(self, rel_type: str) -> bool:
        """Declare a relationship table. Returns True if it now exists."""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rel_type):
            return False
        try:
            self._conn.execute(
                f"CREATE REL TABLE IF NOT EXISTS {rel_type}(FROM Wiki TO Wiki)"
            )
            self._known_rel_tables.add(rel_type)
            logger.info("kuzu: declared rel table %s on demand", rel_type)
            return True
        except Exception as exc:
            logger.warning("kuzu: could not declare rel table %s: %s", rel_type, exc)
            return False

    # --- the dialect deltas, measured by running carton's REAL writer strings ---------------- #
    # Probing the actual queries the daemon issues (not synthetic shapes) found that 8 of 9
    # dialect features carton depends on work verbatim — MERGE, ON CREATE SET, UNWIND over a
    # param list of maps, CASE WHEN inside SET, CONTAINS, coalesce, string concatenation, and
    # relationship MERGE. Exactly TWO do not, and both are translated here so the ~62 write-Cypher
    # strings can stay put, which was the point of putting the seam in the class.
    #
    # This is deliberately NOT a Cypher rewriter. It is two token-level substitutions and one
    # statement kind that is skipped — narrow, named, and each pinned by a test. Anything beyond
    # that belongs in the query, not in a regex.

    # neo4j `datetime()` → kuzu `current_timestamp()`; `datetime(x)` → `timestamp(x)`.
    _DATETIME_NOW = re.compile(r"\bdatetime\(\s*\)", re.IGNORECASE)
    _DATETIME_OF = re.compile(r"\bdatetime\(", re.IGNORECASE)
    # neo4j index DDL has no kuzu equivalent AND needs none: the Wiki table declares
    # PRIMARY KEY (n), so the lookup the index exists to serve is already indexed.
    _INDEX_DDL = re.compile(r"^\s*(CREATE|DROP)\s+(INDEX|CONSTRAINT)\b", re.IGNORECASE)
    # neo4j's `toString(` has no kuzu spelling; kuzu's is `to_string(`. A pure identifier rename,
    # so unlike `substring` it needs no argument parsing. THIS IS NOT COSMETIC: the doc-mirror
    # memory-net skill's rehydration queries all project `toString(e.t) AS ts`, and on kuzu the
    # untranslated call is a CATALOG error that fails the whole read.
    # ⚠ THE FUNCTION TRANSLATES BUT THE FORMAT DIFFERS, and callers that PARSE the string must
    # know: neo4j renders a datetime ISO-8601 (`2026-08-12T23:18:00.086000000+00:00`), kuzu
    # renders `2026-08-12 23:18:00.086` — a space instead of the `T`, and no offset. Anything
    # splitting on 'T' or reading an offset sees a different string on each engine.
    _TO_STRING = re.compile(r"\btoString\s*\(", re.IGNORECASE)
    # neo4j's `type(rel)` is spelled `label(rel)` in kuzu, and the untranslated call is a CATALOG
    # error. The blanket rename is safe because in neo4j `type()` accepts ONLY a relationship —
    # there is no other meaning for it to collide with — while kuzu's `label()` accepts both a
    # node and a relationship, so it answers everything the original could have been asked.
    _TYPE_OF = re.compile(r"\btype\s*\(", re.IGNORECASE)

    def _translate(self, query: str) -> Optional[str]:
        """The query as kuzu should see it, or None if it is a no-op on this backend."""
        if self._INDEX_DDL.match(query):
            logger.debug("kuzu: skipping index DDL (the Wiki table's PRIMARY KEY already serves it)")
            return None
        translated = self._DATETIME_NOW.sub("current_timestamp()", query)
        translated = self._DATETIME_OF.sub("timestamp(", translated)
        translated = self._TO_STRING.sub("to_string(", translated)
        translated = self._TYPE_OF.sub("label(", translated)
        return self._translate_substring(translated)

    @classmethod
    def _translate_substring(cls, query: str) -> str:
        """neo4j `substring` is 0-INDEXED; kuzu's is 1-indexed. Shift a literal start by one.

        ⛔ THIS ONE IS SILENT AND IT IS EVERYWHERE. `substring(n.d, 0, 200)` returns the first 200
        characters on neo4j and an EMPTY STRING on kuzu — no error, just nothing. It is the shape
        the doc-mirror memory-net skill instructs agents to use on every preview query, so without
        this every rehydration read would come back blank while looking perfectly healthy.

        Only a LITERAL integer start is translated. A computed start cannot be shifted safely by
        text substitution, so it RAISES with instructions rather than being quietly left to differ
        between engines — measured across carton and the doc-mirror read layer, every real call
        site uses a literal, so this refuses nothing that exists today.
        """
        out, i = [], 0
        while True:
            m = re.compile(r"\bsubstring\s*\(", re.IGNORECASE).search(query, i)
            if not m:
                out.append(query[i:])
                return "".join(out)
            out.append(query[i:m.end()])
            args, depth, start = [], 1, m.end()
            j = m.end()
            while j < len(query) and depth:
                ch = query[j]
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth -= 1
                    if depth == 0:
                        args.append(query[start:j])
                        break
                elif ch == "," and depth == 1:
                    args.append(query[start:j])
                    start = j + 1
                j += 1
            if len(args) >= 2 and args[1].strip().lstrip("+-").isdigit():
                args[1] = f" {int(args[1].strip()) + 1}"
            elif len(args) >= 2:
                raise ValueError(
                    f"substring() with a computed start argument ({args[1].strip()!r}) cannot be "
                    "translated between engines: neo4j indexes from 0 and kuzu from 1. Use a "
                    "literal start, or left(x, n) which means the same thing on both."
                )
            out.append(",".join(args) + ")")
            i = j + 1

    # A query can legitimately need several schema additions (a new rel table AND a property on
    # it). Bounded so a genuinely broken query cannot spin: each attempt must make a DIFFERENT
    # addition or the loop stops and the real error is raised.
    _MAX_SCHEMA_FIXES = 8

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a query, growing the schema on demand for anything carton mints at runtime.

        NO SILENT FALLBACK. Exactly two failures are treated as "the schema has not caught up
        yet" — a relationship table that does not exist, and a property that does not exist —
        because both are things carton creates dynamically and neither indicates a bad query.
        Anything else raises immediately, and so does either of those if the addition fails: an
        empty list would read as "no rows matched", which is a normal answer and therefore the
        worst possible disguise for an error.
        """
        query = self._translate(query)
        if query is None:
            return []
        is_ddl = bool(self._SCHEMA_DDL.match(query))
        with self._lock:
            applied: set = set()
            for _ in range(self._MAX_SCHEMA_FIXES + 1):
                try:
                    rows = self._rows(self._conn.execute(query, params or {}))
                    if is_ddl:
                        self._renew_connection()
                    return rows
                except Exception as exc:
                    fix = self._schema_fix_for(str(exc), query)
                    if fix is None or fix in applied:
                        raise
                    applied.add(fix)
                    # The failed prepare is cached on this connection under this exact
                    # (query, parameter shape); retrying here would replay it.
                    self._renew_connection()
            rows = self._rows(self._conn.execute(query, params or {}))
            if is_ddl:
                self._renew_connection()
            return rows

    def _schema_fix_for(self, message: str, query: str) -> Optional[str]:
        """Apply the one schema addition this error asks for; return its id, or None."""
        node = self._MISSING_NODE_TABLE.search(message)
        if node:
            logger.warning(
                "kuzu: %r is a missing NODE table, which cannot be created on demand (a node "
                "table needs a primary key and the error names none). Declare it first — see "
                "context-alignment's ensure_code_graph_schema for the shape.", node.group(1))
            return None

        table = self._missing_table_name(message)
        if table and table not in self._known_rel_tables and self._create_rel_table(table):
            return f"table:{table}"

        m = self._MISSING_PROPERTY.search(message)
        if m:
            prop, variable = m.group(1), m.group(2)
            owner = self._table_for_variable(query, variable)
            if owner and self._add_property(owner, prop):
                return f"prop:{owner}.{prop}"
        return None

    @staticmethod
    def _table_for_variable(query: str, variable: str) -> Optional[str]:
        """Which table a query variable is bound to, read out of the pattern that binds it.

        A relationship variable appears as `[var:TYPE]` and a node variable as `(var:Label`, so
        both are stated by the query itself. `Wiki` remains the answer for an UNLABELLED node
        variable, which is every one carton writes — it has exactly one node label, which is what
        made a flat `return "Wiki"` correct for as long as carton was the only writer.

        IT IS NOT THE ONLY WRITER ANY MORE. context-alignment stores a code graph — File, Class,
        Method, Function and a dozen more — in the same database, and under the flat answer a
        missing `File.module_name` was repaired by ALTERing `Wiki`: a schema fix applied to the
        wrong table, reported as applied, leaving the real error in place. Reading the label is
        what makes the answer true for any writer rather than for one.
        """
        rel = re.search(rf"\[\s*{re.escape(variable)}\s*:\s*(\w+)", query)
        if rel:
            return rel.group(1)
        node = re.search(rf"\(\s*{re.escape(variable)}\s*:\s*(\w+)", query)
        if node:
            return node.group(1)
        return "Wiki"

    def _add_property(self, table: str, prop: str) -> bool:
        """ALTER a table to carry a property carton has just decided exists."""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            return False
        ktype = PROPERTY_TYPES.get(prop, DEFAULT_PROPERTY_TYPE)
        try:
            self._conn.execute(f"ALTER TABLE {table} ADD IF NOT EXISTS `{prop}` {ktype}")
            logger.info("kuzu: added property %s.%s (%s) on demand", table, prop, ktype)
            return True
        except Exception as exc:
            logger.warning("kuzu: could not add property %s.%s: %s", table, prop, exc)
            return False

    def _missing_table_name(self, message: str) -> Optional[str]:
        m = self._MISSING_TABLE.search(message)
        if not m:
            return None
        return m.group(1) or m.group(2)

    # ----------------------------------------------------------------- #
    # The property surface. Because properties are real columns (added on demand), these are
    # ordinary Cypher — the same questions the neo4j backend answers, in the two spellings kuzu
    # accepts: it has no `SET c += $map` and no `REMOVE`, so a per-key SET and `SET x = NULL` are
    # the equivalents. Nothing here is a workaround for a missing column; that is handled below.
    # ----------------------------------------------------------------- #

    def set_properties(self, concept_name: str, properties: Dict[str, Any]) -> None:
        for key, value in properties.items():
            self.execute(f"MATCH (c:Wiki {{n: $n}}) SET c.`{key}` = $v",
                         {"n": concept_name, "v": value})

    def remove_properties(self, concept_name: str, keys: List[str]) -> None:
        # NULL is what "absent" means on a schema-full engine, and it is already what a read of a
        # never-set property returns — so this is the same observable state neo4j's REMOVE leaves.
        for key in keys:
            self.execute(f"MATCH (c:Wiki {{n: $n}}) SET c.`{key}` = NULL", {"n": concept_name})

    def find_by_properties(self, where: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        clauses = " AND ".join(f"c.`{k}` = $w_{i}" for i, k in enumerate(where))
        params: Dict[str, Any] = {f"w_{i}": v for i, (k, v) in enumerate(where.items())}
        projection = ", ".join(f"c.`{k}` AS `{k}`" for k in where)
        return self.execute(
            f"MATCH (c:Wiki) WHERE {clauses} RETURN c.n AS n, {projection} LIMIT {int(limit)}",
            params,
        )

    @staticmethod
    def _rows(result) -> List[Dict[str, Any]]:
        """QueryResult → the same `list[dict]` shape neo4j's path returns.

        kuzu hands back plain python values (a node arrives as a dict already), so no
        Node/Relationship/Path conversion is needed on this side — which is why the serializer in
        the read facade can stay exactly where it is and simply pass these through.
        """
        rows: List[Dict[str, Any]] = []
        while result.has_next():
            row = result.get_next()
            rows.append({result.get_column_names()[i]: v for i, v in enumerate(row)})
        return rows

    def close(self) -> None:
        self._conn = None
        self._db = None


class KuzuHttpStore(GraphStore):
    """kuzu reached over localhost HTTP, because the file can only be held by ONE process.

    WHY THIS EXISTS. The engine is embedded: the process that opens the database directory holds
    it, a second read-write open is refused by the file lock, and a second read_only open is
    either refused (kuzu 0.11.3) or — on ladybug — allowed and served a stale snapshot. carton is
    not one process — the MCP server, the worker daemon and every agent's stdio subprocess all read the
    graph. Under neo4j they each opened a bolt connection to a shared server; under kuzu exactly
    one process can own the file, so everybody else has to ASK it. That process is the worker,
    which is also the only writer, and this is the client the others use.

    It is the shape carton already uses twice — SOMA on :8091 and the chroma daemon on :8190 —
    and it is deliberately DUMB: urllib only, ZERO kuzu import, no connection state. A process
    holding this store cannot open a kuzu file even by accident, which is the property that makes
    the single-writer rule structural instead of remembered.

    SELECTED BY `KUZU_QUERY_URL`, which takes precedence over `KUZU_DB_PATH` on purpose: if a
    process is told where to ask, it must ASK rather than open, even when it also knows the path.
    """

    CREDENTIAL_ENV = ("CARTON_USER", "CARTON_KEY")

    def __init__(self, url: str, timeout: float = 30.0, user=None, key=None):
        self._url = url.rstrip("/")
        self._timeout = timeout
        # THE CREDENTIALS COME FROM THE ENVIRONMENT, which is where a tenant's MCP
        # settings put them. Nothing is stored, negotiated or minted here: the
        # client is told who it is and hands that to the server on every call.
        self._user = user if user is not None else os.environ.get("CARTON_USER", "")
        self._key = key if key is not None else os.environ.get("CARTON_KEY", "")

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        import json
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        # Sent only when configured, so the in-box localhost wire (which has no
        # credentials and needs none) is byte-identical to before.
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        if self._user:
            headers["X-Carton-User"] = self._user
        req = urllib.request.Request(f"{self._url}{path}", data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # The endpoint answers a failed QUERY with a 400 carrying the engine's own message.
            # Surfacing that verbatim keeps a binder/catalog error readable as itself instead of
            # arriving as a bare HTTP status with the cause thrown away.
            detail = exc.read().decode("utf-8", "replace")[:600]
            raise RuntimeError(f"kuzu query endpoint {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"kuzu query endpoint unreachable at {self._url} ({exc.reason}). Under the "
                "single-writer design the worker owns the database file and serves this endpoint; "
                "if it is not running, nothing can read the graph."
            ) from exc
        if not body.get("ok"):
            raise RuntimeError(f"kuzu query endpoint error: {body.get('error')}")
        return body.get("result")

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._post("/query", {"query": query, "params": params or {}}) or []

    def set_properties(self, concept_name: str, properties: Dict[str, Any]) -> None:
        self._post("/set_properties", {"concept_name": concept_name, "properties": properties})

    def remove_properties(self, concept_name: str, keys: List[str]) -> None:
        self._post("/remove_properties", {"concept_name": concept_name, "keys": list(keys)})

    def find_by_properties(self, where: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        return self._post("/find_by_properties", {"where": where, "limit": limit}) or []

    def close(self) -> None:
        """Nothing to close — the point of this store is that it holds no database handle."""

    @property
    def driver(self):
        """None, and that is the invariant: a client process has no driver and no file handle.

        `KnowledgeGraphBuilder` mirrors this onto its own `.driver`, so any code that still
        reaches past the class for a raw session fails LOUDLY here rather than quietly opening a
        second handle onto a file another process owns.
        """
        return None


def resolve_backend() -> str:
    """Which backend this process talks to. Default neo4j — an unset env changes nothing.

    ⛔ THERE IS NO OVERRIDE, AND ADDING ONE BACK IS THE BUG (Isaac, 2026-08-18: *"THERE SHOULD BE
    ABSOLUTELY NOTHING anywhere that is available to swap the graph because it should be ONE env
    var that is set by system_config.sh which exports everything into the env of the container"*).

    This function briefly took an `override` so context-alignment could answer differently than
    carton inside one process. That solved a real incident — the nightly CA refresh rebuilding the
    code graph into the personal record — by creating a SECOND place the graph could be chosen,
    and then a third was needed (per-MCP env), and a fourth (box internals). One graph means one
    place to name it: `GRAPH_BACKEND` in `/home/GOD/system_config.sh`, which every process sources
    and every carton call therefore inherits. Two consumers wanting two stores is a PROVISIONING
    problem — seed the store — never a routing problem to solve with another knob.
    """
    return (os.environ.get("GRAPH_BACKEND") or "neo4j").strip().lower()


def make_store(uri: str, user: str, password: str) -> GraphStore:
    """Build the backend named by GRAPH_BACKEND. Unknown value = a loud error, never a default.

    Takes no backend/url arguments by design — see `resolve_backend`. The environment is the only
    input, so no caller can route itself somewhere else.
    """
    backend = resolve_backend()
    if backend == "neo4j":
        return Neo4jStore(uri, user, password)
    if backend == "kuzu":
        # ASK BEFORE OPEN. A process told where the endpoint is must never open the file, even
        # if it also knows the path — the owner holds the directory, so a second opener does not
        # get a live view, it gets a lock failure or a stale snapshot (see
        # Kuzu_Handle_Visibility_Boundary).
        query_url = os.environ.get("KUZU_QUERY_URL")
        if query_url:
            # THE TIMEOUT IS CONFIGURABLE BECAUSE NOT EVERY CALLER IS INTERACTIVE. 30s is right
            # for a query someone is waiting on; a batch writer (a repository parse issues
            # thousands of statements against a single-writer store) can meet a longer stall
            # while the store flushes, and a client-side timeout there aborts a run that was
            # succeeding. Measured 2026-08-17: a code-graph parse wrote 461 nodes and then hit
            # the 30s ceiling on one statement that completes in 0.1s on its own.
            return KuzuHttpStore(query_url, timeout=_env_float("KUZU_QUERY_TIMEOUT_S", 30.0))
        # ⛔ A READ-ONLY OPEN IS REFUSED, NOT IGNORED. On ladybug a second process's read_only
        # open succeeds while the worker holds the file and answers from a snapshot that never
        # sees the worker's later writes — every read looks healthy and is stale. Ignoring the
        # variable instead would open the file read-write, which the lock refuses. Either way
        # the setting cannot do what it says, so it stops the process and names the way that
        # works: the owner's endpoint.
        if _env_true("KUZU_READ_ONLY"):
            raise ValueError(
                "KUZU_READ_ONLY is not supported: a second process's read_only open serves a "
                "stale snapshot on ladybug. Set KUZU_QUERY_URL to the worker's query endpoint "
                "instead (the worker owns the database file; every other process reads through "
                "KuzuHttpStore)."
            )
        db_path = os.environ.get("KUZU_DB_PATH")
        if not db_path:
            raise ValueError(
                "GRAPH_BACKEND=kuzu requires KUZU_DB_PATH (the embedded database's directory). "
                "Under the single-writer design the worker process owns that file exclusively — "
                "every other process reads through KuzuHttpStore (KUZU_QUERY_URL)."
            )
        return KuzuStore(db_path)
    raise ValueError(
        f"unknown GRAPH_BACKEND {backend!r}; expected 'neo4j' or 'kuzu' "
        "('kuzu' is the embedded engine, served by the ladybug package)"
    )


def _env_float(name: str, default: float) -> float:
    """A numeric env override that REFUSES junk rather than silently falling back to the default.

    A timeout that quietly becomes 30 because someone typed `KUZU_QUERY_TIMEOUT_S=6OO` is the
    kind of setting that looks applied and is not.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number of seconds") from None


def _env_true(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes")


