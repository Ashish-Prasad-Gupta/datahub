import time
from typing import Dict, Optional

from datahub.ingestion.source.state.checkpoint import Checkpoint
from datahub.ingestion.source.state.resource_fingerprint import (
    ResourceChangeDetectionConfig,
    ResourceFingerprintCheckpointState,
    ResourceFingerprintEntry,
    ResourceFingerprintHandler,
    compute_fingerprint,
)
from datahub.ingestion.source.state.stateful_ingestion_base import (
    StatefulIngestionConfig,
    StatefulIngestionConfigBase,
)

TEST_RESOURCE_URN = "urn:li:container:abc123"
TEST_ENTITY_URNS = [
    "urn:li:dataset:(urn:li:dataPlatform:postgres,db.public.t1,PROD)",
    "urn:li:dataset:(urn:li:dataPlatform:postgres,db.public.t2,PROD)",
]


def test_compute_fingerprint_is_order_independent() -> None:
    rows_a = [("t1", "c1", 1, "varchar"), ("t1", "c2", 2, "int")]
    rows_b = [("t1", "c2", 2, "int"), ("t1", "c1", 1, "varchar")]
    assert compute_fingerprint(rows_a) == compute_fingerprint(rows_b)


def test_compute_fingerprint_changes_with_content() -> None:
    rows_a = [("t1", "c1", 1, "varchar")]
    rows_b = [("t1", "c1", 1, "varchar"), ("t1", "c2", 2, "int")]
    assert compute_fingerprint(rows_a) != compute_fingerprint(rows_b)


def test_checkpoint_state_roundtrip() -> None:
    state = ResourceFingerprintCheckpointState(
        resources={
            TEST_RESOURCE_URN: ResourceFingerprintEntry(
                fingerprint="deadbeef",
                entity_urns=TEST_ENTITY_URNS,
                last_full_refresh_at=1700000000.0,
            )
        }
    )
    checkpoint = Checkpoint(
        job_name="postgres_schema_fingerprint",
        pipeline_name="test_pipeline",
        run_id="run1",
        state=state,
    )

    aspect = checkpoint.to_checkpoint_aspect(max_allowed_state_size=2**24)
    assert aspect is not None

    restored = Checkpoint.create_from_checkpoint_aspect(
        job_name="postgres_schema_fingerprint",
        checkpoint_aspect=aspect,
        state_class=ResourceFingerprintCheckpointState,
    )
    assert restored is not None
    entry = restored.state.resources[TEST_RESOURCE_URN]
    assert entry.fingerprint == "deadbeef"
    assert entry.entity_urns == TEST_ENTITY_URNS
    assert entry.last_full_refresh_at == 1700000000.0


class _FakeStateProvider:
    """A minimal stand-in for StateProviderWrapper, sufficient to exercise
    ResourceFingerprintHandler without a real checkpoint-backed pipeline."""

    def __init__(self, last_checkpoint: Optional[Checkpoint] = None):
        self._last_checkpoint = last_checkpoint
        self._cur_checkpoint: Optional[Checkpoint] = None
        self._usecase_handlers: Dict[str, ResourceFingerprintHandler] = {}

    def register_stateful_ingestion_usecase_handler(self, handler) -> None:
        self._usecase_handlers[handler.job_id] = handler

    def is_stateful_ingestion_configured(self) -> bool:
        return True

    def get_last_checkpoint(self, job_id, checkpoint_state_class):
        return self._last_checkpoint

    def get_current_checkpoint(self, job_id):
        if self._cur_checkpoint is None:
            self._cur_checkpoint = self._usecase_handlers[job_id].create_checkpoint()
        return self._cur_checkpoint


class _FakeSource:
    def __init__(self, state_provider: _FakeStateProvider, platform: str = "postgres"):
        self.state_provider = state_provider
        self.platform = platform


def _make_handler(
    last_checkpoint: Optional[Checkpoint] = None,
    enabled: bool = True,
    full_refresh_interval_hours: int = 24,
    job_id_suffix: str = "schema_fingerprint",
):
    state_provider = _FakeStateProvider(last_checkpoint=last_checkpoint)
    source = _FakeSource(state_provider)
    config = StatefulIngestionConfigBase(
        stateful_ingestion=StatefulIngestionConfig(enabled=True)
    )
    change_detection_config = ResourceChangeDetectionConfig(
        enabled=enabled, full_refresh_interval_hours=full_refresh_interval_hours
    )
    handler = ResourceFingerprintHandler(
        source=source,  # type: ignore[arg-type]
        config=config,
        change_detection_config=change_detection_config,
        pipeline_name="test_pipeline",
        run_id="run1",
        job_id_suffix=job_id_suffix,
    )
    return handler, state_provider


def _checkpoint_with_entry(entry: ResourceFingerprintEntry) -> Checkpoint:
    return Checkpoint(
        job_name="postgres_schema_fingerprint",
        pipeline_name="test_pipeline",
        run_id="prior_run",
        state=ResourceFingerprintCheckpointState(resources={TEST_RESOURCE_URN: entry}),
    )


def test_job_id_uses_platform_and_suffix() -> None:
    handler, _ = _make_handler(job_id_suffix="table_fingerprint")
    assert handler.job_id == "postgres_table_fingerprint"


def test_should_skip_resource_no_prior_entry() -> None:
    handler, _ = _make_handler(last_checkpoint=None)
    assert handler.should_skip_resource(TEST_RESOURCE_URN, "fp1") is False


def test_should_skip_resource_matching_fingerprint_within_window() -> None:
    entry = ResourceFingerprintEntry(
        fingerprint="fp1",
        entity_urns=TEST_ENTITY_URNS,
        last_full_refresh_at=time.time(),
    )
    handler, _ = _make_handler(last_checkpoint=_checkpoint_with_entry(entry))
    assert handler.should_skip_resource(TEST_RESOURCE_URN, "fp1") is True


def test_should_skip_resource_mismatched_fingerprint() -> None:
    entry = ResourceFingerprintEntry(
        fingerprint="fp1",
        entity_urns=TEST_ENTITY_URNS,
        last_full_refresh_at=time.time(),
    )
    handler, _ = _make_handler(last_checkpoint=_checkpoint_with_entry(entry))
    assert handler.should_skip_resource(TEST_RESOURCE_URN, "fp2") is False


def test_should_skip_resource_unknown_fingerprint_fails_open() -> None:
    entry = ResourceFingerprintEntry(
        fingerprint="fp1",
        entity_urns=TEST_ENTITY_URNS,
        last_full_refresh_at=time.time(),
    )
    handler, _ = _make_handler(last_checkpoint=_checkpoint_with_entry(entry))
    assert handler.should_skip_resource(TEST_RESOURCE_URN, None) is False


def test_should_skip_resource_stale_beyond_refresh_interval() -> None:
    entry = ResourceFingerprintEntry(
        fingerprint="fp1",
        entity_urns=TEST_ENTITY_URNS,
        last_full_refresh_at=time.time() - 100 * 3600,
    )
    handler, _ = _make_handler(
        last_checkpoint=_checkpoint_with_entry(entry), full_refresh_interval_hours=24
    )
    assert handler.should_skip_resource(TEST_RESOURCE_URN, "fp1") is False


def test_should_skip_resource_disabled_config() -> None:
    entry = ResourceFingerprintEntry(
        fingerprint="fp1",
        entity_urns=TEST_ENTITY_URNS,
        last_full_refresh_at=time.time(),
    )
    handler, _ = _make_handler(
        last_checkpoint=_checkpoint_with_entry(entry), enabled=False
    )
    assert handler.should_skip_resource(TEST_RESOURCE_URN, "fp1") is False


def test_carry_forward_resource_returns_and_persists_prior_urns() -> None:
    entry = ResourceFingerprintEntry(
        fingerprint="fp1",
        entity_urns=TEST_ENTITY_URNS,
        last_full_refresh_at=time.time(),
    )
    handler, state_provider = _make_handler(
        last_checkpoint=_checkpoint_with_entry(entry)
    )

    carried = handler.carry_forward_resource(TEST_RESOURCE_URN)
    assert carried == TEST_ENTITY_URNS

    cur_checkpoint = state_provider.get_current_checkpoint(handler.job_id)
    assert cur_checkpoint is not None
    assert (
        cur_checkpoint.state.resources[TEST_RESOURCE_URN].entity_urns
        == TEST_ENTITY_URNS
    )


def test_carry_forward_resource_with_no_prior_entry_returns_empty() -> None:
    handler, _ = _make_handler(last_checkpoint=None)
    assert handler.carry_forward_resource(TEST_RESOURCE_URN) == []


def test_record_full_refresh_persists_deduped_sorted_urns() -> None:
    handler, state_provider = _make_handler(last_checkpoint=None)

    duplicated_urns = [TEST_ENTITY_URNS[1], TEST_ENTITY_URNS[0], TEST_ENTITY_URNS[0]]
    handler.record_full_refresh(TEST_RESOURCE_URN, "fp_new", duplicated_urns)

    cur_checkpoint = state_provider.get_current_checkpoint(handler.job_id)
    assert cur_checkpoint is not None
    entry = cur_checkpoint.state.resources[TEST_RESOURCE_URN]
    assert entry.fingerprint == "fp_new"
    assert entry.entity_urns == sorted(TEST_ENTITY_URNS)


def test_record_full_refresh_noop_when_disabled() -> None:
    handler, state_provider = _make_handler(last_checkpoint=None, enabled=False)
    handler.record_full_refresh(TEST_RESOURCE_URN, "fp_new", TEST_ENTITY_URNS)
    # is_checkpointing_enabled() is False, so create_checkpoint() must never be
    # invoked and no current checkpoint should materialize.
    assert state_provider._cur_checkpoint is None


def test_single_entity_urn_reuse_for_table_granularity() -> None:
    """Databricks reuses this handler at table granularity, where entity_urns
    is just [resource_urn] itself rather than a list of child entities."""
    handler, state_provider = _make_handler(
        last_checkpoint=None, job_id_suffix="table_fingerprint"
    )
    table_urn = "urn:li:dataset:(urn:li:dataPlatform:databricks,cat.schema.t1,PROD)"
    handler.record_full_refresh(table_urn, "fp1", [table_urn])

    cur_checkpoint = state_provider.get_current_checkpoint(handler.job_id)
    assert cur_checkpoint is not None
    assert cur_checkpoint.state.resources[table_urn].entity_urns == [table_urn]
