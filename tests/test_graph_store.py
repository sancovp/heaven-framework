#!/usr/bin/env python3
"""The gate for the graph backend seam (`GRAPH_BACKEND=neo4j|kuzu`).

Run as a SCRIPT (the repo's own convention for these suites):
    python3 base/heaven-framework/tests/test_graph_store.py

WHAT THIS SUITE IS ACTUALLY PROTECTING. Two things, and the second is the one that would be
expensive to get wrong:

1. THE NEO4J PATH IS UNCHANGED. It is the live path of a 643,090-node production graph, so
   `Neo4jStore.execute` must return exactly what `execute_query` returned before the seam existed
   — `[dict(record)]` with the VALUES untouched, meaning a `RETURN c` still hands back a live Node
   object rather than a serialized dict.

2. REMOVING THE `.driver.session()` BYPASS IN THE READ FACADE IS SAFE. That rewrite rests on one
   claim: `_serialize_record` treats a plain dict exactly as it treats a neo4j Record. If that is
   false, every MCP read silently changes shape. It is asserted here rather than assumed.

The embedded cases need the engine the seam imports — `ladybug==0.21.2`. They SKIP loudly if it
is absent rather than passing vacuously (a real pytest skip under pytest, a SKIP line as a script)
— a skipped test that reads as green is the failure mode this whole port has been correcting. The
same suite runs against the frozen kuzu 0.11.3 reference through the shim in
`application/carton-saas/kuzu-port/ladybug-probes/reference_shim/`; `test_ENGINE_UNDER_TEST`
prints which engine a run proved.
"""
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heaven_base.tool_utils.graph_store import (  # noqa: E402
    KuzuStore,
    Neo4jStore,
    PROPERTY_TYPES,
    WIKI_FIXED_COLUMNS,
    make_store,
    resolve_backend,
)

PASSED, FAILED, SKIPPED = [], [], []


def check(name, fn):
    try:
        result = fn()
        if result == "skip":
            SKIPPED.append(name)
            print(f"  SKIP  {name}")
        else:
            PASSED.append(name)
            print(f"  ok    {name}")
    except Exception as exc:
        FAILED.append((name, exc))
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def has_kuzu():
    """Whether the engine the seam imports (`ladybug`) is installed."""
    try:
        from heaven_base.tool_utils.graph_store import _engine
        _engine()
        return True
    except ImportError:
        return False


def _skip(reason="the embedded engine (ladybug) is not installed"):
    """A REAL skip under pytest (it shows as `s`, never as a pass); the "skip" marker as a script."""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(reason)
    return "skip"


def test_ENGINE_UNDER_TEST():
    """Names the engine this run proved, in the output, so a green run says what it was green ON."""
    if not has_kuzu():
        return _skip()
    from heaven_base.tool_utils.graph_store import _engine
    eng = _engine()
    print(f"ENGINE {eng.__name__} {eng.__version__}")


# --------------------------------------------------------------------------- #
# 1. The backend selector — an unset env must change nothing, a wrong one must be loud
# --------------------------------------------------------------------------- #

def test_default_backend_is_neo4j_so_an_unset_env_changes_nothing():
    old = os.environ.pop("GRAPH_BACKEND", None)
    try:
        assert resolve_backend() == "neo4j", resolve_backend()
    finally:
        if old is not None:
            os.environ["GRAPH_BACKEND"] = old


def test_an_unknown_backend_RAISES_rather_than_defaulting():
    """A typo must not silently land on neo4j — that is how a tenant box would quietly run on the
    wrong substrate while every log line looked correct."""
    os.environ["GRAPH_BACKEND"] = "kuzoo"
    try:
        make_store("bolt://x", "u", "p")
        raise AssertionError("expected a ValueError for an unknown backend")
    except ValueError as exc:
        assert "kuzoo" in str(exc), exc
    finally:
        os.environ.pop("GRAPH_BACKEND", None)


def test_kuzu_without_a_db_path_RAISES_and_says_why():
    os.environ["GRAPH_BACKEND"] = "kuzu"
    old = os.environ.pop("KUZU_DB_PATH", None)
    try:
        make_store("bolt://x", "u", "p")
        raise AssertionError("expected a ValueError when KUZU_DB_PATH is unset")
    except ValueError as exc:
        assert "KUZU_DB_PATH" in str(exc), exc
    finally:
        os.environ.pop("GRAPH_BACKEND", None)
        if old is not None:
            os.environ["KUZU_DB_PATH"] = old


# --------------------------------------------------------------------------- #
# 2. ⭐ THE LOAD-BEARING ONE — the read facade's serializer treats a dict like a Record
# --------------------------------------------------------------------------- #

def test_the_serializer_treats_a_DICT_exactly_like_a_RECORD():
    """This is what makes removing `graph.driver.session()` from `_execute_neo4j_query` safe.

    `_serialize_record` only ever calls `record.keys()` and `record[key]`. A neo4j Record and a
    plain dict answer both identically, so feeding it the dicts that `execute_query` returns must
    produce byte-identical output to feeding it Records. Asserted with a stand-in Record that
    supports exactly the Record API surface the serializer uses — if someone later makes the
    serializer reach for something Record-only (`.values()`, `.data()`, index access), this fails
    and names the reason.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge", "carton-mcp"))
    from carton_utils import CartOnUtils

    class FakeRecord:
        """Only what a Record gives that a dict also gives — deliberately no more."""
        def __init__(self, data):
            self._data = data

        def keys(self):
            return self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    payload = {"name": "A_Concept", "count": 3, "tags": ["x", "y"], "nested": {"k": "v"}}
    utils = CartOnUtils()
    from_record = utils._serialize_record(FakeRecord(payload))
    from_dict = utils._serialize_record(dict(payload))
    assert from_record == from_dict, (from_record, from_dict)
    assert from_dict["tags"] == ["x", "y"], from_dict
    assert from_dict["nested"] == {"k": "v"}, from_dict


def _driver_session_calls(path):
    """Every `<anything>.driver.session(...)` CALL in a file, found by parsing rather than by text.

    A text search cannot tell code from prose, and both of these files now carry docstrings that
    NAME the bypass in order to explain why it was removed — so grepping for the string reports a
    violation that is really an explanation. Parsing asks the question that was actually meant.
    """
    import ast
    tree = ast.parse(open(path).read())
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "session"
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "driver"):
            hits.append(node.lineno)
    return hits


def test_the_read_facade_no_longer_reaches_for_driver_session():
    """The bypass is gone from the CODE, not merely unused."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "knowledge", "carton-mcp", "carton_utils.py")
    hits = _driver_session_calls(path)
    assert hits == [], f"the read facade still bypasses the class at line(s) {hits}"
    assert "graph.execute_query(cypher_query" in open(path).read(), \
        "the read facade should ask the connection"


def test_the_worker_no_longer_reaches_for_driver_session():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "knowledge", "carton-mcp", "observation_worker_daemon.py")
    hits = _driver_session_calls(path)
    assert hits == [], f"the worker still bypasses the class at line(s) {hits}"


# --------------------------------------------------------------------------- #
# 3. The neo4j path's shape (no server needed — the contract is the return shape)
# --------------------------------------------------------------------------- #

def test_neo4j_execute_returns_dict_of_record_with_VALUES_UNTOUCHED():
    """`RETURN c` must keep handing back the live Node. add_concept_tool.py:3473 consumes one that
    way, so a serialization pass here would change behaviour on the production path."""
    class Sentinel:
        """Stands in for a neo4j Node — the point is that it survives unconverted."""

    node = Sentinel()

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, query, params):
            return [{"c": node, "n": "A_Concept"}]

    class FakeDriver:
        def session(self):
            return FakeSession()

        def close(self):
            pass

    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = FakeDriver()
    rows = store.execute("MATCH (c:Wiki) RETURN c, c.n AS n", {})
    assert rows == [{"c": node, "n": "A_Concept"}], rows
    assert rows[0]["c"] is node, "the Node was converted — that is a behaviour change"


# --------------------------------------------------------------------------- #
# 4. The overflow column — the scratch lane's home
# --------------------------------------------------------------------------- #


def test_the_fixed_columns_are_the_twelve_that_were_measured():
    names = [c[0] for c in WIKI_FIXED_COLUMNS]
    assert len(names) == 12, names
    # The five universal ones, measured on the full graph.
    for essential in ("n", "linked", "d", "t", "c"):
        assert essential in names, essential
    # And the managed lane.
    for managed in ("last_modified", "score", "source", "timeline_linked",
                    "region", "odyssey_linked", "soma_region"):
        assert managed in names, managed


# --------------------------------------------------------------------------- #
# 5. kuzu, against a real embedded database
# --------------------------------------------------------------------------- #

def _with_kuzu(fn):
    if not has_kuzu():
        return _skip()
    tmp = tempfile.mkdtemp(prefix="kuzu_gate_")
    try:
        store = KuzuStore(os.path.join(tmp, "db"))
        return fn(store)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kuzu_creates_the_wiki_table_and_round_trips_a_node():
    def body(store):
        store.execute(
            "CREATE (c:Wiki {n: $n, d: $d, linked: false})",
            {"n": "A_Concept", "d": "a description"},
        )
        rows = store.execute("MATCH (c:Wiki) RETURN c.n AS n, c.d AS d", {})
        assert rows == [{"n": "A_Concept", "d": "a description"}], rows
        return None
    return _with_kuzu(body)


def test_kuzu_declares_an_UNKNOWN_rel_table_on_demand_and_retries():
    """The crux of the whole port: carton mints relationship types dynamically, and kuzu requires
    each to be declared. A query naming a rel table nobody created must succeed."""
    def body(store):
        store.execute("CREATE (c:Wiki {n: 'Child', linked: false})", {})
        store.execute("CREATE (c:Wiki {n: 'Parent', linked: false})", {})
        # PART_OF has never been declared in this database.
        store.execute(
            "MATCH (a:Wiki {n: 'Child'}), (b:Wiki {n: 'Parent'}) CREATE (a)-[:PART_OF]->(b)", {}
        )
        rows = store.execute(
            "MATCH (a:Wiki)-[:PART_OF]->(b:Wiki) RETURN a.n AS child, b.n AS parent", {}
        )
        assert rows == [{"child": "Child", "parent": "Parent"}], rows
        assert "PART_OF" in store._known_rel_tables, store._known_rel_tables
        return None
    return _with_kuzu(body)


def test_kuzu_raises_on_a_genuinely_bad_query_rather_than_returning_no_rows():
    """NO SILENT FALLBACKS. An empty list that means 'the query was broken' is indistinguishable
    from one that means 'nothing matched', and the second is a normal answer."""
    def body(store):
        try:
            store.execute("THIS IS NOT CYPHER", {})
        except Exception as exc:
            # Assert on the MESSAGE, not merely that something was raised: the error has to reach
            # the caller intact, because "the DDL-on-demand retry swallowed the real reason" is
            # the specific way this could go wrong.
            assert str(exc).strip(), "the backend raised an error with no message"
            assert "does not exist" not in str(exc), (
                "a syntax error was misread as a missing rel table: " + str(exc)
            )
            return None
        raise AssertionError("a broken query returned instead of raising")
    return _with_kuzu(body)



def test_the_builder_on_kuzu_has_no_driver_but_still_answers():
    """A driverless backend must be a first-class citizen of the class, not a special case."""
    if not has_kuzu():
        return _skip()
    from heaven_base.tool_utils.neo4j_utils import KnowledgeGraphBuilder

    tmp = tempfile.mkdtemp(prefix="kuzu_builder_")
    old_backend = os.environ.get("GRAPH_BACKEND")
    old_path = os.environ.get("KUZU_DB_PATH")
    os.environ["GRAPH_BACKEND"] = "kuzu"
    os.environ["KUZU_DB_PATH"] = os.path.join(tmp, "db")
    try:
        graph = KnowledgeGraphBuilder()
        graph._ensure_connection()          # the RETURN 1 test must pass on this backend
        assert graph.driver is None, "kuzu has no driver and must say so plainly"
        graph.execute_query("CREATE (c:Wiki {n: $n, linked: false})", {"n": "Via_Builder"})
        rows = graph.execute_query("MATCH (c:Wiki) RETURN c.n AS n", {})
        assert rows == [{"n": "Via_Builder"}], rows
        graph.close()
    finally:
        for key, val in (("GRAPH_BACKEND", old_backend), ("KUZU_DB_PATH", old_path)):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 5b. ⛔ THE LADYBUG LAWS — connection renewal, and no read-only open
#
# Ladybug's Python client caches a prepared statement per connection for every PARAMETERIZED
# query, keyed on (query text, parameter shape), and caches it even when preparation FAILED. The
# schema-on-demand retry in `KuzuStore.execute` re-runs the SAME query with the SAME params after
# creating the missing table or column — so on a connection that is not renewed, the retry
# replays the cached failure. Every case below passes on kuzu 0.11.3 with or without renewal
# (kuzu had no implicit cache) and FAILS on ladybug 0.21.2 without it. Each uses PARAMETERS on
# purpose: a literal-only query is not cached, and would pass without the fix.
# --------------------------------------------------------------------------- #

def test_a_MISSING_NODE_LABEL_never_becomes_a_REL_TABLE():
    """`MATCH (r:Rule ...)` on an undeclared node label fails as "Table Rule does not exist" — the
    same text a missing rel type gives. The fix must NOT answer it by creating `REL TABLE Rule`:
    the error is re-raised, and `show_tables()` lists no table named Rule afterwards."""
    def body(store):
        import pytest as _pytest
        with _pytest.raises(Exception) as excinfo:
            store.execute("MATCH (r:Rule {n: $n}) RETURN r.n AS n", {"n": "x"})
        assert "Rule" in str(excinfo.value), str(excinfo.value)
        names = [row[0] if isinstance(row, (list, tuple)) else row
                 for row in store._conn.execute("CALL show_tables() RETURN *").get_all()]
        assert not any(str(n) == "Rule" for n in names), names
        assert "Rule" not in store._known_rel_tables
        return None
    return _with_kuzu(body)


def test_ladybug_DYNAMIC_REL_TYPE_MERGE_with_params_lands_after_its_table_is_created():
    """carton mints rel types at runtime; the first MERGE of a new type must land, params and all."""
    def body(store):
        store.execute("MERGE (c:Wiki {n: $n}) ON CREATE SET c.linked = false", {"n": "Dyn_A"})
        store.execute("MERGE (c:Wiki {n: $n}) ON CREATE SET c.linked = false", {"n": "Dyn_B"})
        store.execute(
            "MATCH (a:Wiki {n: $a}), (b:Wiki {n: $b}) MERGE (a)-[r:HAS_THING]->(b)",
            {"a": "Dyn_A", "b": "Dyn_B"})
        rows = store.execute("MATCH (a:Wiki)-[:HAS_THING]->(b:Wiki) RETURN a.n AS a, b.n AS b", {})
        assert rows == [{"a": "Dyn_A", "b": "Dyn_B"}], rows
        return None
    return _with_kuzu(body)


def test_ladybug_SET_a_NEW_REL_PROPERTY_with_params_lands_after_its_column_is_added():
    """The daemon's `SET r.ts = ...` on an edge type that has never carried `ts`."""
    def body(store):
        store.execute("CREATE (a:Wiki {n: 'Rp_A', linked: false})", {})
        store.execute("CREATE (b:Wiki {n: 'Rp_B', linked: false})", {})
        store.execute("MATCH (a:Wiki {n:'Rp_A'}), (b:Wiki {n:'Rp_B'}) MERGE (a)-[:HAS_THING]->(b)", {})
        store.execute(
            "MATCH (a:Wiki {n: $a})-[r:HAS_THING]->(b:Wiki {n: $b}) SET r.reason = $why",
            {"a": "Rp_A", "b": "Rp_B", "why": "because"})
        rows = store.execute("MATCH (:Wiki)-[r:HAS_THING]->(:Wiki) RETURN r.reason AS why", {})
        assert rows == [{"why": "because"}], rows
        return None
    return _with_kuzu(body)


def test_ladybug_SET_PROPERTIES_with_a_NEW_node_property_lands():
    """`set_properties` is parameterized by construction, so every new scratch key hits the cache."""
    def body(store):
        store.execute("CREATE (c:Wiki {n: 'Np_A', linked: false})", {})
        store.set_properties("Np_A", {"status": "locked"})
        rows = store.execute("MATCH (c:Wiki {n: 'Np_A'}) RETURN c.status AS st", {})
        assert rows == [{"st": "locked"}], rows
        return None
    return _with_kuzu(body)


def test_ladybug_SET_PROPERTIES_with_a_SECOND_new_node_property_lands_too():
    """A second, different new key after the first succeeded — each new column is its own miss."""
    def body(store):
        store.execute("CREATE (c:Wiki {n: 'Np_B', linked: false})", {})
        store.set_properties("Np_B", {"status": "locked"})
        store.set_properties("Np_B", {"equipped_sm_id": "sm_1", "sm_chain_index": 0})
        rows = store.execute(
            "MATCH (c:Wiki {n: 'Np_B'}) RETURN c.status AS st, c.equipped_sm_id AS sm, "
            "c.sm_chain_index AS ix", {})
        assert rows == [{"st": "locked", "sm": "sm_1", "ix": 0}], rows
        return None
    return _with_kuzu(body)


def test_ladybug_a_plan_cached_BEFORE_an_ALTER_does_not_hide_the_new_column():
    """The silent one: a query that SUCCEEDED before `ALTER TABLE ... ADD` keeps its old plan on a
    connection that is not renewed, and `RETURN c` comes back without the new column — no error."""
    def body(store):
        store.execute("CREATE (c:Wiki {n: 'Stale_A', linked: false})", {})
        before = store.execute("MATCH (c:Wiki {n: $n}) RETURN c", {"n": "Stale_A"})[0]["c"]
        assert "tk_lane" not in before, before
        store.execute("ALTER TABLE Wiki ADD IF NOT EXISTS tk_lane STRING", {})
        after = store.execute("MATCH (c:Wiki {n: $n}) RETURN c", {"n": "Stale_A"})[0]["c"]
        assert "tk_lane" in after, f"the plan cached before the ALTER is still in use: {sorted(after)}"
        return None
    return _with_kuzu(body)


def test_KUZU_READ_ONLY_REFUSES_and_names_the_endpoint():
    """A second process's read_only open succeeds on ladybug and serves a stale snapshot, so the
    setting refuses rather than open anything — and says what to do instead."""
    prev = {k: os.environ.get(k) for k in ("GRAPH_BACKEND", "KUZU_READ_ONLY", "KUZU_DB_PATH", "KUZU_QUERY_URL")}
    os.environ.update(GRAPH_BACKEND="kuzu", KUZU_READ_ONLY="1", KUZU_DB_PATH="/nonexistent/never-opened")
    os.environ.pop("KUZU_QUERY_URL", None)
    try:
        make_store("", "", "")
        raise AssertionError("KUZU_READ_ONLY opened a store instead of refusing")
    except ValueError as exc:
        assert "KUZU_QUERY_URL" in str(exc) and "stale" in str(exc), exc
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_KuzuStore_offers_NO_read_only_open():
    import inspect
    assert "read_only" not in inspect.signature(KuzuStore.__init__).parameters, \
        "KuzuStore takes read_only again — a second-process read_only open serves a stale snapshot"


# --------------------------------------------------------------------------- #
# 6. The property surface — an arbitrary key where Cypher wants an identifier
# --------------------------------------------------------------------------- #

def _capturing_neo4j_store():
    """A Neo4jStore that records the Cypher it would run instead of running it."""
    store = Neo4jStore.__new__(Neo4jStore)
    store.issued = []

    def fake_execute(query, params=None):
        store.issued.append((" ".join(query.split()), params or {}))
        return []

    store.execute = fake_execute
    return store


def test_neo4j_property_writes_issue_THE_SAME_CYPHER_AS_BEFORE():
    """Relocating these out of carton_utils must be a move, not a rewrite — the live graph must
    not be able to tell. Each assertion is the literal clause the old code built."""
    store = _capturing_neo4j_store()

    store.set_properties("A_Concept", {"status": "open", "order": 3})
    query, params = store.issued[-1]
    assert query == "MATCH (c:Wiki {n: $n}) SET c += $props", query
    assert params == {"n": "A_Concept", "props": {"status": "open", "order": 3}}, params

    store.remove_properties("A_Concept", ["status", "tk_lane"])
    query, params = store.issued[-1]
    assert query == "MATCH (c:Wiki {n: $n}) REMOVE c.`status`, c.`tk_lane`", query
    assert params == {"n": "A_Concept"}, params

    store.find_by_properties({"status": "open", "tk_lane": "doing"}, 25)
    query, params = store.issued[-1]
    assert query == (
        "MATCH (c:Wiki) WHERE c.`status` = $w_0 AND c.`tk_lane` = $w_1 "
        "RETURN c.n AS n, c.`status` AS `status`, c.`tk_lane` AS `tk_lane` LIMIT $lim"
    ), query
    assert params == {"w_0": "open", "w_1": "doing", "lim": 25}, params



def test_a_scratch_write_does_not_DEADLOCK_on_its_own_lock():
    """Pinned because it nearly shipped: the read-modify-write path holds the store's lock across
    several `execute` calls, and `execute` takes that lock too. With a plain Lock that is a
    deadlock — the process hangs with no error at all, which is the worst possible failure shape.
    Run in a thread with a join timeout so a regression FAILS instead of hanging the suite.
    """
    if not has_kuzu():
        return _skip()
    import threading

    done, error = [], []

    def run():
        tmp = tempfile.mkdtemp(prefix="kuzu_lock_")
        try:
            store = KuzuStore(os.path.join(tmp, "db"))
            store.execute("CREATE (c:Wiki {n: 'Locky', linked: false})", {})
            store.set_properties("Locky", {"status": "open"})
            store.remove_properties("Locky", ["status"])
            done.append(True)
        except Exception as exc:                      # noqa: BLE001 - reported below
            error.append(exc)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=30)
    assert not error, f"the scratch path raised: {error[0]}"
    assert done, "the scratch property path did not finish in 30s — it deadlocked on its own lock"



def test_kuzu_finds_by_a_SCRATCH_key_which_has_no_column_to_filter_on():
    """The case that cannot be a WHERE clause on a schema-full engine, and the reason
    find_by_properties is the backend's job rather than a shared Cypher string."""
    def body(store):
        for name, lane in (("Card_A", "doing"), ("Card_B", "done"), ("Card_C", "doing")):
            store.execute("CREATE (c:Wiki {n: $n, linked: false})", {"n": name})
            store.set_properties(name, {"tk_lane": lane})
        found = store.find_by_properties({"tk_lane": "doing"}, 25)
        assert sorted(r["n"] for r in found) == ["Card_A", "Card_C"], found
        # The requested key comes back alongside the name, as the neo4j path returns it.
        assert all(r["tk_lane"] == "doing" for r in found), found
        return None
    return _with_kuzu(body)


def test_kuzu_finds_on_a_MIX_of_fixed_and_scratch_and_honours_the_limit():
    def body(store):
        for name in ("Mix_A", "Mix_B", "Mix_C"):
            store.execute("CREATE (c:Wiki {n: $n, linked: false, region: 'soup'})", {"n": name})
            store.set_properties(name, {"status": "open"})
        store.execute("CREATE (c:Wiki {n: 'Mix_D', linked: false, region: 'code'})", {})
        store.set_properties("Mix_D", {"status": "open"})

        both = store.find_by_properties({"region": "soup", "status": "open"}, 25)
        assert sorted(r["n"] for r in both) == ["Mix_A", "Mix_B", "Mix_C"], both
        assert both[0]["region"] == "soup" and both[0]["status"] == "open", both
        assert len(store.find_by_properties({"region": "soup", "status": "open"}, 2)) == 2
        return None
    return _with_kuzu(body)


# --------------------------------------------------------------------------- #
# 7. The dialect deltas — measured by running carton's REAL writer strings
# --------------------------------------------------------------------------- #

# Verbatim from observation_worker_daemon.batch_create_concepts_neo4j. If the daemon's query
# changes, this copy goes stale — which is the point: it is what makes "the strings stay put"
# an assertion instead of a hope.
DAEMON_CREATE_QUERY = """
UNWIND $concepts AS c
MERGE (n:Wiki {n: c.name})
ON CREATE SET n.c = c.canonical, n.linked = false
SET n.d = CASE
    WHEN n.d IS NULL OR n.d = '' THEN c.description
    WHEN c.update_mode = 'replace' THEN c.description
    WHEN c.description CONTAINS n.d THEN c.description
    WHEN c.update_mode = 'append' THEN n.d + $sep + c.description
    ELSE n.d
END
SET n.t = CASE WHEN n.t IS NULL THEN (CASE WHEN c.timestamp IS NOT NULL THEN datetime(c.timestamp) ELSE datetime() END) ELSE n.t END
SET n.last_modified = datetime()
SET n.source = CASE WHEN n.source IS NULL THEN c.source ELSE n.source END
SET n.region = coalesce(c.region, n.region, 'soup')
"""


def test_kuzu_runs_THE_DAEMONS_ACTUAL_NODE_WRITE_verbatim():
    """The load-bearing assumption of the whole port — carton's write-Cypher strings stay put and
    execution routes through the seam (kuzu-port/LADYBUG.md) — asserted against the real query
    rather than a synthetic MERGE. It exercises UNWIND over a param list of maps, MERGE, ON CREATE
    SET, a multi-branch CASE inside SET, CONTAINS, string concatenation, coalesce, and both forms
    of datetime() in one statement."""
    def body(store):
        rows = [{
            "name": "Daemon_Written", "canonical": "daemon_written",
            "description": "first", "timestamp": None, "update_mode": "append",
            "source": "agent", "region": "soup",
        }]
        sep = "\n\n---\n\n"                       # the daemon's real separator
        store.execute(DAEMON_CREATE_QUERY, {"concepts": rows, "sep": sep})
        got = store.execute(
            "MATCH (c:Wiki {n:'Daemon_Written'}) RETURN c.d AS d, c.region AS region, c.c AS canon", {})
        assert got[0]["d"] == "first", got
        assert got[0]["region"] == "soup", got
        assert got[0]["canon"] == "daemon_written", got

        # And the append branch of the CASE, on a second pass over the same node.
        rows[0]["description"] = "second"
        store.execute(DAEMON_CREATE_QUERY, {"concepts": rows, "sep": sep})
        got = store.execute("MATCH (c:Wiki {n:'Daemon_Written'}) RETURN c.d AS d", {})
        assert got[0]["d"] == "first" + sep + "second", got
        return None
    return _with_kuzu(body)


def test_an_escaped_newline_in_a_CYPHER_LITERAL_is_ENGINE_DEPENDENT():
    """⛔ THE SILENT ONE. This is why the daemon's separator is a parameter, and stays one.

    neo4j processes backslash escapes inside a string literal, so `'\\n\\n---\\n\\n'` in the query
    text is two newlines, a rule, and two newlines. kuzu 0.11.3 does NOT — it drops the
    backslashes and keeps the letters, so the separator would have become `nn---nn` and EVERY
    appended concept description would have been quietly corrupted, with no error anywhere.
    ladybug 0.21.2 processes them, as neo4j does.

    Pinned PER ENGINE as a characterisation test: if either engine's behaviour moves, this fails
    and says so. The parameter stays regardless — one literal means three different strings across
    the three engines carton has run on, and the frozen reference still eats the escapes.
    """
    from heaven_base.tool_utils.graph_store import _engine
    expected = {"kuzu": "annb", "ladybug": "a\n\nb"}

    def body(store):
        name = _engine().__name__
        got = store.execute(r"RETURN 'a\n\nb' AS v", {})[0]["v"]
        assert got == expected[name], f"{name}'s escape behaviour changed: {got!r}"
        # The two shapes that DO survive: a parameter (what the daemon now uses)...
        assert store.execute("RETURN $s AS v", {"s": "a\n\nb"})[0]["v"] == "a\n\nb"
        # ...and a real newline inside the query text.
        assert store.execute("RETURN 'a\n\nb' AS v", {})[0]["v"] == "a\n\nb"
        return None
    return _with_kuzu(body)


def test_no_cypher_literal_in_the_writers_relies_on_escape_processing():
    """The general form of the finding, guarded at the source: a Cypher string literal carrying a
    backslash escape is silently backend-dependent. There were exactly two (the append and prepend
    separators, same query); both are parameters now."""
    import re as _re
    carton = os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge", "carton-mcp")
    offenders = []
    for fname in os.listdir(carton):
        if not fname.endswith(".py"):
            continue
        src = open(os.path.join(carton, fname)).read()
        for m in _re.finditer(r'("""|\'\'\')(.*?)\1', src, _re.S):
            body_text = m.group(2)
            if "\\n" in body_text and _re.search(r"\b(MERGE|UNWIND|SET|MATCH)\b", body_text):
                for line in body_text.splitlines():
                    if "\\n" in line and "'" in line:
                        offenders.append(f"{fname}: {line.strip()[:70]}")
    assert not offenders, "Cypher literals relying on escape processing:\n  " + "\n  ".join(offenders)


def test_kuzu_translates_BOTH_forms_of_datetime():
    """kuzu has no `datetime` function at all. Translated rather than left to fail, so the daemon's
    strings need no per-backend variants: `datetime()` is now, `datetime(x)` parses x."""
    def body(store):
        store.execute("CREATE (c:Wiki {n: 'Stamped', linked: false})", {})
        store.execute("MATCH (c:Wiki {n:'Stamped'}) SET c.last_modified = datetime()", {})
        store.execute("MATCH (c:Wiki {n:'Stamped'}) SET c.t = datetime('2026-08-12T12:00:00')", {})
        rows = store.execute("MATCH (c:Wiki {n:'Stamped'}) RETURN c.t AS t, c.last_modified AS lm", {})
        assert rows[0]["t"].year == 2026 and rows[0]["t"].hour == 12, rows
        assert rows[0]["lm"], rows
        return None
    return _with_kuzu(body)


def test_kuzu_SKIPS_index_ddl_because_the_primary_key_already_serves_it():
    """Not a translation — a no-op. The Wiki table declares PRIMARY KEY (n), so the lookup the
    neo4j index exists to serve is already indexed. Returning [] beats raising on a statement the
    daemon issues on every batch and already wraps in a try/except."""
    def body(store):
        assert store.execute("CREATE INDEX wiki_name IF NOT EXISTS FOR (w:Wiki) ON (w.n)", {}) == []
        assert store.execute("DROP INDEX wiki_name IF EXISTS", {}) == []
        # A query that merely CONTAINS the word index is NOT a no-op.
        store.execute("CREATE (c:Wiki {n: 'Indexish', linked: false})", {})
        assert store.execute("MATCH (c:Wiki {n:'Indexish'}) RETURN c.n AS n", {}) == [{"n": "Indexish"}]
        return None
    return _with_kuzu(body)


def test_the_reserved_word_alias_is_backticked_at_the_daemon():
    """`desc` is a RESERVED WORD on the embedded backend — it parses as the DESC sort keyword and
    the whole query fails. Fixed in the daemon's query rather than by a rewriter, because
    backticks are valid identifier quoting in BOTH dialects and the returned key is unchanged."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "knowledge", "carton-mcp", "observation_worker_daemon.py")
    body = open(path).read()
    assert "n.d AS `desc`" in body, "the dedup query's reserved-word alias is not backticked"
    assert "n.d AS desc\"" not in body, "an unbackticked `desc` alias is still present"

    def kuzu_body(store):
        store.execute("CREATE (c:Wiki {n: 'Aliased', d: 'text', linked: false})", {})
        rows = store.execute(
            "UNWIND $names AS name MATCH (n:Wiki {n: name}) WHERE n.d IS NOT NULL "
            "RETURN n.n AS name, n.d AS `desc`", {"names": ["Aliased"]})
        assert rows == [{"name": "Aliased", "desc": "text"}], rows
        return None
    return _with_kuzu(kuzu_body)


def test_kuzu_adds_an_UNDECLARED_property_as_a_real_column_on_demand():
    """The mechanism that replaced the JSON overflow column. It has to be a real column because
    `sm_gate.py` reads and writes `s.status` in RAW CYPHER — against a blob those queries would
    silently match nothing and the retrieval gate would fail quietly."""
    def body(store):
        store.execute("CREATE (c:Wiki {n: 'Fresh', linked: false})", {})
        # `status` is not in the twelve base columns; the SET must still land.
        store.execute("MATCH (c:Wiki {n:'Fresh'}) SET c.status = 'locked'", {})
        assert store.execute("MATCH (s:Wiki) WHERE s.status = 'locked' RETURN s.n AS n", {}) == [{"n": "Fresh"}]
        # and the sm_gate shapes that motivated it
        assert store.execute("MATCH (s:Wiki {n:'Fresh'}) RETURN coalesce(s.status,'unlocked') AS st",
                             {})[0]["st"] == "locked"
        return None
    return _with_kuzu(body)


def test_kuzu_adds_an_UNDECLARED_RELATIONSHIP_property_on_demand():
    """The daemon writes `SET rel.ts = datetime()` on every edge it creates. Rel tables are
    schema-full too, so without this every relationship write fails — which is exactly what the
    first end-to-end run of the real daemon did."""
    def body(store):
        store.execute("CREATE (a:Wiki {n:'A', linked:false})", {})
        store.execute("CREATE (b:Wiki {n:'B', linked:false})", {})
        store.execute("MATCH (a:Wiki{n:'A'}),(b:Wiki{n:'B'}) MERGE (a)-[r:IS_A]->(b) SET r.ts = datetime()", {})
        got = store.execute("MATCH (:Wiki)-[r:IS_A]->(:Wiki) RETURN r.ts IS NOT NULL AS has_ts", {})
        assert got == [{"has_ts": True}], got
        # a SECOND undeclared property on the SAME rel table, added later
        store.execute("MATCH (:Wiki)-[r:IS_A]->(:Wiki) SET r.reason = 'because'", {})
        assert store.execute("MATCH (:Wiki)-[r:IS_A]->(:Wiki) RETURN r.reason AS why", {}) == [{"why": "because"}]
        return None
    return _with_kuzu(body)


def test_property_types_are_declared_from_measurement_not_guessed():
    """kuzu needs a type at ALTER time. `ts` carries 3.5M edges on the live graph and must be a
    TIMESTAMP, not a string, or `coalesce(r.weight,1.0) + $delta`-style arithmetic breaks."""
    assert PROPERTY_TYPES["ts"] == "TIMESTAMP", PROPERTY_TYPES["ts"]
    assert PROPERTY_TYPES["weight"] == "DOUBLE", PROPERTY_TYPES["weight"]
    assert PROPERTY_TYPES["status"] == "STRING", PROPERTY_TYPES["status"]


def test_kuzu_translates_neo4js_ZERO_INDEXED_substring():
    """⛔ ANOTHER SILENT ONE. neo4j's substring is 0-indexed and kuzu's is 1-indexed, so
    `substring(n.d,0,200)` — the exact shape the doc-mirror memory-net skill tells every agent to
    use for previews — returns an EMPTY STRING on kuzu instead of the first 200 characters."""
    def body(store):
        store.execute("CREATE (c:Wiki {n:'Sliced', d:'abcdefghij', linked:false})", {})
        assert store.execute("MATCH (c:Wiki {n:'Sliced'}) RETURN substring(c.d,0,4) AS p",
                             {})[0]["p"] == "abcd"
        # a nested first argument (commas inside parens) must not confuse the translation
        assert store.execute("MATCH (c:Wiki {n:'Sliced'}) RETURN substring(coalesce(c.d,''),0,3) AS p",
                             {})[0]["p"] == "abc"
        return None
    return _with_kuzu(body)


def test_collect_over_an_UNMATCHED_optional_match_is_NULL_not_an_empty_list():
    """⛔ THE 14TH DELTA, AND SILENT. neo4j's `collect()` over an OPTIONAL MATCH that matched
    nothing yields `[]`, so `collect(a) + collect(b)` is the matched half. On kuzu it yields
    NULL, `list + NULL` is NULL, and `UNWIND` over NULL emits NO ROWS — so the WHOLE answer
    disappears, including the half that DID match, with no error anywhere.

    FOUND by the gnosys e2e grader's collection walk returning zero members for a collection
    that demonstrably had one: check 4 would have called a correct run an empty collection.

    BISECTED, so the record says which construct is at fault and not merely 'the query broke':
    plain MATCH, the multi-rel filter `[:A|B]`, OPTIONAL MATCH alone, collect+UNWIND, and even
    list concat with two NON-empty sides ALL work. Only the empty-side concat fails.

    This is a CHARACTERISATION test: it asserts what kuzu DOES, so if a later version starts
    returning `[]` we are TOLD the constraint moved instead of silently depending on it. The
    callers' fix is not to work around it in Cypher but to ask each direction separately and
    merge in python — valid on both engines and needing no dialect knowledge.
    """
    def body(store):
        store.execute("CREATE (c:Wiki {n:'Coll', linked:false})", {})
        store.execute("CREATE (m:Wiki {n:'Mem', linked:false})", {})
        store.execute("MATCH (c:Wiki {n:'Coll'}), (m:Wiki {n:'Mem'}) MERGE (c)-[:HAS_PART]->(m)", {})
        # the matched half ALONE is fine
        assert [r["n"] for r in store.execute(
            "MATCH (c:Wiki {n:'Coll'})-[:HAS_PART]->(d:Wiki) RETURN DISTINCT d.n AS n", {})] == ["Mem"]
        # concatenated with a collect over a direction that matched NOTHING, it vanishes
        vanished = store.execute(
            "MATCH (c:Wiki {n:'Coll'}) "
            "OPTIONAL MATCH (c)-[:HAS_PART]->(d:Wiki) "
            "OPTIONAL MATCH (u:Wiki)-[:PART_OF]->(c) "
            "WITH collect(DISTINCT d.n) + collect(DISTINCT u.n) AS ns "
            "UNWIND ns AS n RETURN DISTINCT n", {})
        assert vanished == [], (
            "kuzu started returning the matched half — the constraint MOVED and the callers "
            f"that split this into two queries can be simplified again: {vanished}")
        return None
    return _with_kuzu(body)


def test_a_COMPUTED_substring_start_is_REFUSED_rather_than_quietly_differing():
    """It cannot be shifted safely by text substitution, and a wrong slice is invisible. Every
    real call site in carton and the doc-mirror read layer uses a literal, so this refuses
    nothing that exists — it only stops a new one being written blind."""
    try:
        KuzuStore._translate_substring("RETURN substring(n.d, $start, 10) AS d")
    except ValueError as exc:
        assert "left(" in str(exc), "the refusal should name the portable alternative"
        return None
    raise AssertionError("a computed substring start was translated instead of refused")


def test_toString_is_translated_because_the_REHYDRATION_QUERIES_ALL_USE_IT():
    """`toString(` has no kuzu spelling and its absence is a CATALOG error, not a wrong answer.

    This is not a corner case: the doc-mirror memory-net skill's rehydration queries every one
    project `toString(e.t) AS ts`, so untranslated, the entire rehydration read fails on kuzu.
    """
    out = KuzuStore._translate(KuzuStore.__new__(KuzuStore), "RETURN toString(e.t) AS ts")
    assert "to_string(" in out, out
    assert "toString(" not in out, out


def test_toString_TRANSLATES_but_the_RENDERED_FORMAT_DIFFERS_and_that_is_recorded():
    """A characterisation test: the call works on both engines, the STRING does not match.

    neo4j renders a datetime ISO-8601 with a 'T' and an offset; kuzu renders 'YYYY-MM-DD hh:mm:ss'
    with a space and no offset. Anything that SPLITS on 'T' or reads an offset gets a different
    answer per engine, so this pins the fact rather than leaving a caller to discover it.
    """
    def body(store):
        store.execute("MERGE (a:Wiki {n:'T1'}) ON CREATE SET a.t = datetime()", {})
        rendered = store.execute("MATCH (c:Wiki {n:'T1'}) RETURN toString(c.t) AS ts", {})[0]["ts"]
        assert rendered, "toString produced nothing"
        assert "T" not in rendered, f"kuzu is expected to render with a SPACE, got {rendered!r}"
        assert rendered[4] == "-" and rendered[7] == "-", rendered
        return None

    return _with_kuzu(body)


def test_the_table_a_property_is_added_to_is_READ_FROM_THE_PATTERN_not_assumed():
    """A schema fix must land on the table the query names, and carton is no longer alone here.

    `_table_for_variable` answered `Wiki` for every node variable, which was correct while carton
    — one node label — was the only writer. context-alignment stores a code graph (File, Class,
    Method, ...) in the same database, and under the flat answer a missing `File.module_name`
    was repaired by ALTERing `Wiki`: the fix reported as applied, the real error still there.

    The unlabelled case still answers Wiki, which is every node variable carton writes, so this
    is a generalization rather than a change of behaviour for the existing writer.
    """
    from heaven_base.tool_utils.graph_store import KuzuStore

    for query, var, want, why in [
        ("MATCH (f:File {path:$p}) SET f.module_name = $m", "f", "File", "a labelled node"),
        ("MERGE (c:Class {full_name:$n}) ON CREATE SET c.name = $x", "c", "Class", "a MERGE label"),
        ("MATCH (a:Wiki{n:'A'}),(b:Wiki{n:'B'}) MERGE (a)-[r:IS_A]->(b) SET r.ts = 1", "r", "IS_A",
         "a relationship still reads its type"),
        ("MATCH (c:Wiki {n:$n}) SET c.status = 'x'", "c", "Wiki", "carton's own labelled node"),
        ("MATCH (x) WHERE x.name = $n SET x.deep_analyzed = true", "x", "Wiki",
         "an UNLABELLED node keeps the old answer"),
    ]:
        got = KuzuStore._table_for_variable(query, var)
        assert got == want, f"{why}: expected {want}, got {got} for {query!r}"


def test_a_missing_NODE_table_is_REFUSED_not_created_as_a_relationship():
    """A guess here is worse than the error it replaces.

    Kuzu says `Cannot bind Rule as a node pattern label` for a missing NODE table. The repair
    path read every missing table as a RELATIONSHIP table, so it created `REL Rule(FROM Wiki TO
    Wiki)` — after which the real node table can NEVER be created (the name is taken by the
    wrong kind) and the failure moves to a later, stranger place. Measured 2026-08-17 on a real
    box during a code-graph parse.

    A node table needs a primary key and the error names none, so this is not repairable on
    demand; the whole fix is that it is refused rather than guessed.
    """
    from heaven_base.tool_utils.graph_store import KuzuStore

    class _Probe(KuzuStore):
        def __init__(self):  # no database — only the decision is under test
            self.created = []
            self._known_rel_tables = set()

        def _create_rel_table(self, rel_type):
            self.created.append(rel_type)
            return True

    probe = _Probe()
    fix = probe._schema_fix_for(
        "Binder exception: Cannot bind Rule as a node pattern label.",
        "MATCH (r:Repository {name: $n})-[:CONTAINS]->(x:Rule) DETACH DELETE x")
    assert fix is None, f"a missing node table must not be repaired, got {fix!r}"
    assert probe.created == [], f"it created a rel table named {probe.created}"

    # and the relationship case still repairs, or this guard would have broken the mechanism
    fix = probe._schema_fix_for("Binder exception: Table FOLLOWS_PATTERN does not exist",
                                "MATCH (a:Wiki),(b:Wiki) MERGE (a)-[:FOLLOWS_PATTERN]->(b)")
    assert fix == "table:FOLLOWS_PATTERN", fix
    assert probe.created == ["FOLLOWS_PATTERN"], probe.created


if __name__ == "__main__":
    print("graph store gate\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    if SKIPPED:
        print("SKIPPED (these are NOT passes):")
        for s in SKIPPED:
            print(f"  - {s}")
    sys.exit(1 if FAILED else 0)
