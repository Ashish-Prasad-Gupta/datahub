import time
from datetime import datetime
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

from databricks.sdk.service.catalog import TableType

from datahub.ingestion.api.common import PipelineContext
from datahub.ingestion.source.state.checkpoint import Checkpoint
from datahub.ingestion.source.state.resource_fingerprint import (
    ResourceFingerprintCheckpointState,
    ResourceFingerprintEntry,
)
from datahub.ingestion.source.unity.config import UnityCatalogSourceConfig
from datahub.ingestion.source.unity.proxy_types import (
    Catalog,
    HiveTableType,
    Metastore,
    Schema,
    Table,
)
from datahub.ingestion.source.unity.source import UnityCatalogSource


class _FakeStateProvider:
    """A minimal in-memory stand-in for StateProviderWrapper: it lets the
    real ResourceFingerprintHandler/StaleEntityRemovalHandler wiring inside
    UnityCatalogSource run end-to-end without a live GMS-backed checkpoint
    provider or network access."""

    def __init__(self) -> None:
        self.last_checkpoints: Dict[str, Optional[Checkpoint]] = {}
        self.cur_checkpoints: Dict[str, Optional[Checkpoint]] = {}
        self._usecase_handlers: Dict[str, object] = {}

    def register_stateful_ingestion_usecase_handler(self, handler) -> None:
        self._usecase_handlers[handler.job_id] = handler

    def is_stateful_ingestion_configured(self) -> bool:
        return True

    def get_last_checkpoint(self, job_id, checkpoint_state_class):
        return self.last_checkpoints.get(job_id)

    def get_current_checkpoint(self, job_id):
        if job_id not in self.cur_checkpoints:
            self.cur_checkpoints[job_id] = self._usecase_handlers[
                job_id
            ].create_checkpoint()
        return self.cur_checkpoints[job_id]

    def prepare_for_commit(self) -> None:
        pass


def _make_schema() -> Schema:
    metastore = Metastore(
        id="metastore",
        name="metastore",
        comment=None,
        global_metastore_id=None,
        metastore_id=None,
        owner=None,
        region=None,
        cloud=None,
    )
    catalog = Catalog(
        id="test_catalog",
        name="test_catalog",
        metastore=metastore,
        comment=None,
        owner=None,
        type=None,
    )
    return Schema(
        id="test_catalog.test_schema",
        name="test_schema",
        catalog=catalog,
        comment=None,
        owner=None,
    )


def _make_table(
    schema: Schema,
    name: str = "t1",
    updated_at=datetime(2024, 1, 1),
    table_type=TableType.MANAGED,
) -> Table:
    return Table(
        id=f"test_catalog.test_schema.{name}",
        name=name,
        comment=None,
        schema=schema,
        columns=[],
        storage_location=None,
        data_source_format=None,
        table_type=table_type,
        owner=None,
        generation=None,
        created_at=None,
        created_by=None,
        updated_at=updated_at,
        updated_by=None,
        table_id=None,
        view_definition=None,
        properties={},
    )


def _make_source(
    change_detection_enabled: bool = True,
    stateful_ingestion_enabled: bool = True,
    full_refresh_interval_hours: int = 24,
) -> UnityCatalogSource:
    config = UnityCatalogSourceConfig.model_validate(
        {
            "token": "test_token",
            "workspace_url": "https://test.databricks.com",
            "warehouse_id": "test_warehouse",
            "include_hive_metastore": False,
            "change_detection": {
                "enabled": change_detection_enabled,
                "full_refresh_interval_hours": full_refresh_interval_hours,
            },
            "stateful_ingestion": {"enabled": stateful_ingestion_enabled},
        }
    )
    # A mock graph satisfies StateProviderWrapper's constructor-time check for
    # a checkpoint provider; we discard the real provider it builds right
    # after by swapping in the in-memory fake below.
    ctx = PipelineContext(
        run_id="test_run", pipeline_name="test_pipeline", graph=MagicMock()
    )
    with (
        patch("datahub.ingestion.source.unity.source.UnityCatalogApiProxy"),
        patch("datahub.ingestion.source.unity.source.HiveMetastoreProxy"),
    ):
        source = UnityCatalogSource.create(config, ctx)
    # Swap in an in-memory fake before constructing the handlers, so
    # get_workunit_processors() below binds them to it instead of trying to
    # talk to a real (nonexistent) GMS-backed checkpoint provider.
    source.state_provider = _FakeStateProvider()
    list(source.get_workunit_processors())
    return source


def _seed_prior_fingerprint(
    source: UnityCatalogSource, table: Table, fingerprint: str
) -> None:
    """Simulates a *previous* run having already recorded this fingerprint,
    by populating the fake state provider's "last checkpoint" directly
    (distinct from the "current" checkpoint this run writes to)."""
    dataset_urn = source.gen_dataset_urn(table.ref)
    handler = source.table_fingerprint_handler
    checkpoint = Checkpoint(
        job_name=handler.job_id,
        pipeline_name="test_pipeline",
        run_id="prior_run",
        state=ResourceFingerprintCheckpointState(
            resources={
                dataset_urn: ResourceFingerprintEntry(
                    fingerprint=fingerprint,
                    entity_urns=[dataset_urn],
                    last_full_refresh_at=time.time(),
                )
            }
        ),
    )
    source.state_provider.last_checkpoints[handler.job_id] = checkpoint


class TestUnityCatalogChangeDetection:
    def test_skips_process_table_when_fingerprint_unchanged(self):
        schema = _make_schema()
        table = _make_table(schema, updated_at=datetime(2024, 1, 1))
        source = _make_source()

        # Seed the checkpoint as if a prior run already saw this exact updated_at.
        fingerprint = str(table.updated_at.timestamp())
        _seed_prior_fingerprint(source, table, fingerprint)

        with patch.object(source, "process_table") as mock_process_table:
            source.unity_catalog_api_proxy.tables.return_value = [table]
            workunits = list(source.process_tables(schema))

        mock_process_table.assert_not_called()
        assert workunits == []
        assert source.report.num_tables_change_detection_skipped == 1
        assert table.ref not in source.table_refs
        assert table.ref.qualified_table_name not in source.tables

    def test_processes_table_when_fingerprint_changed(self):
        schema = _make_schema()
        table = _make_table(schema, updated_at=datetime(2024, 1, 2))
        source = _make_source()

        # Seed with a different (older) fingerprint than the table's current updated_at.
        _seed_prior_fingerprint(source, table, str(datetime(2024, 1, 1).timestamp()))

        with patch.object(source, "process_table", return_value=iter([])):
            source.unity_catalog_api_proxy.tables.return_value = [table]
            list(source.process_tables(schema))

        assert source.report.num_tables_change_detection_skipped == 0
        assert table.ref in source.table_refs

        dataset_urn = source.gen_dataset_urn(table.ref)
        entry = source.table_fingerprint_handler._get_last_entry(dataset_urn)
        # _get_last_entry reads the *last* (prior-run) checkpoint, so it still
        # reflects the seeded value; confirm the *current* checkpoint was updated.
        cur_checkpoint = source.state_provider.get_current_checkpoint(
            source.table_fingerprint_handler.job_id
        )
        assert cur_checkpoint is not None
        assert cur_checkpoint.state.resources[dataset_urn].fingerprint == str(
            table.updated_at.timestamp()
        )
        assert entry is not None  # sanity: seeded entry exists in the "last" checkpoint

    def test_processes_table_with_no_prior_checkpoint(self):
        schema = _make_schema()
        table = _make_table(schema)
        source = _make_source()

        with patch.object(source, "process_table", return_value=iter([])):
            source.unity_catalog_api_proxy.tables.return_value = [table]
            list(source.process_tables(schema))

        assert source.report.num_tables_change_detection_skipped == 0
        assert table.ref in source.table_refs

    def test_null_updated_at_always_processed_fail_open(self):
        schema = _make_schema()
        table = _make_table(schema, updated_at=None)
        source = _make_source()

        with patch.object(source, "process_table", return_value=iter([])):
            source.unity_catalog_api_proxy.tables.return_value = [table]
            list(source.process_tables(schema))

        assert source.report.num_tables_change_detection_skipped == 0
        assert table.ref in source.table_refs

    def test_hive_metastore_tables_are_never_skipped(self):
        schema = _make_schema()
        table = _make_table(
            schema,
            updated_at=datetime(2024, 1, 1),
            table_type=HiveTableType.HIVE_MANAGED_TABLE,
        )
        source = _make_source()

        fingerprint = str(table.updated_at.timestamp())
        _seed_prior_fingerprint(source, table, fingerprint)

        with patch.object(source, "process_table", return_value=iter([])):
            source.unity_catalog_api_proxy.tables.return_value = [table]
            list(source.process_tables(schema))

        # Even with a matching fingerprint, hive_metastore tables always process.
        assert source.report.num_tables_change_detection_skipped == 0
        assert table.ref in source.table_refs

    def test_disabled_config_always_processes(self):
        schema = _make_schema()
        table = _make_table(schema, updated_at=datetime(2024, 1, 1))
        source = _make_source(change_detection_enabled=False)

        fingerprint = str(table.updated_at.timestamp())
        _seed_prior_fingerprint(source, table, fingerprint)

        with patch.object(source, "process_table", return_value=iter([])):
            source.unity_catalog_api_proxy.tables.return_value = [table]
            list(source.process_tables(schema))

        assert source.report.num_tables_change_detection_skipped == 0
        assert table.ref in source.table_refs

    def test_skipped_table_carried_forward_to_stale_removal_state(self):
        schema = _make_schema()
        table = _make_table(schema, updated_at=datetime(2024, 1, 1))
        source = _make_source()

        fingerprint = str(table.updated_at.timestamp())
        _seed_prior_fingerprint(source, table, fingerprint)
        dataset_urn = source.gen_dataset_urn(table.ref)

        with patch.object(source, "process_table") as mock_process_table:
            source.unity_catalog_api_proxy.tables.return_value = [table]
            list(source.process_tables(schema))

        mock_process_table.assert_not_called()
        stale_removal_job_id = source.stale_entity_removal_handler.job_id
        cur_checkpoint = source.state_provider.get_current_checkpoint(
            stale_removal_job_id
        )
        assert cur_checkpoint is not None
        assert dataset_urn in cur_checkpoint.state.urns
