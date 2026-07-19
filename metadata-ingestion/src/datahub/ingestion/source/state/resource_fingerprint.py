import hashlib
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import pydantic
from pydantic import BaseModel, Field

from datahub.configuration.common import ConfigModel
from datahub.ingestion.api.common import PipelineContext
from datahub.ingestion.api.ingestion_job_checkpointing_provider_base import JobId
from datahub.ingestion.source.state.checkpoint import Checkpoint, CheckpointStateBase
from datahub.ingestion.source.state.stateful_ingestion_base import (
    StatefulIngestionConfig,
    StatefulIngestionConfigBase,
    StatefulIngestionSourceBase,
)
from datahub.ingestion.source.state.use_case_handler import (
    StatefulIngestionUsecaseHandlerBase,
)

logger: logging.Logger = logging.getLogger(__name__)


class ResourceChangeDetectionConfig(ConfigModel):
    """Base shape for a source's opt-in change-detection config. Sources
    should subclass or otherwise provide a config with these two fields so
    it can be passed to `ResourceFingerprintHandler`."""

    enabled: bool = Field(
        default=False,
        description="Enable a lightweight fingerprint check that skips expensive "
        "re-processing of resources that have not changed since the last run. "
        "Requires `stateful_ingestion.enabled` to be set to true.",
    )
    full_refresh_interval_hours: int = Field(
        default=24,
        gt=0,
        description="Force a full re-processing of a resource at least this often, "
        "even if its fingerprint is unchanged. Bounds the staleness of freshness "
        "signals (e.g. lastObserved) on carried-forward entities and guards against "
        "fingerprint false negatives.",
    )


def compute_fingerprint(rows: Iterable[Tuple[Any, ...]]) -> str:
    """
    Computes a deterministic fingerprint over a set of catalog rows (e.g. one
    per table/column), independent of the order the rows were returned in.
    """
    normalized_rows = sorted(str(row) for row in rows)
    hasher = hashlib.sha256()
    for row in normalized_rows:
        hasher.update(row.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


class ResourceFingerprintEntry(BaseModel):
    fingerprint: str
    entity_urns: List[str] = Field(default_factory=list)
    last_full_refresh_at: float


class ResourceFingerprintCheckpointState(CheckpointStateBase):
    resources: Dict[str, ResourceFingerprintEntry] = pydantic.Field(
        default_factory=dict
    )


class ResourceFingerprintHandler(
    StatefulIngestionUsecaseHandlerBase["ResourceFingerprintCheckpointState"]
):
    """
    Stateful-ingestion helper that lets sources skip expensive re-processing of
    resources (e.g. a SQL schema, a Unity Catalog table) whose fingerprint has
    not changed since the last run, carrying forward their previously known
    entity urns so that stale-entity removal doesn't treat them as deleted.
    """

    def __init__(
        self,
        source: StatefulIngestionSourceBase,
        config: StatefulIngestionConfigBase[StatefulIngestionConfig],
        change_detection_config: ResourceChangeDetectionConfig,
        pipeline_name: Optional[str],
        run_id: str,
        job_id_suffix: str = "resource_fingerprint",
    ):
        self.source = source
        self.state_provider = source.state_provider
        self.stateful_ingestion_config: Optional[StatefulIngestionConfig] = (
            config.stateful_ingestion
        )
        self.change_detection_config = change_detection_config
        self.pipeline_name = pipeline_name
        self.run_id = run_id
        self.job_id_suffix = job_id_suffix
        self._job_id = self._init_job_id()
        self.state_provider.register_stateful_ingestion_usecase_handler(self)

    @classmethod
    def create(
        cls,
        source: StatefulIngestionSourceBase,
        config: StatefulIngestionConfigBase,
        change_detection_config: ResourceChangeDetectionConfig,
        ctx: PipelineContext,
        job_id_suffix: str = "resource_fingerprint",
    ) -> "ResourceFingerprintHandler":
        return cls(
            source,
            config,
            change_detection_config,
            ctx.pipeline_name,
            ctx.run_id,
            job_id_suffix,
        )

    def _init_job_id(self) -> JobId:
        platform: Optional[str] = getattr(self.source, "platform", "default")
        return JobId(
            f"{platform}_{self.job_id_suffix}" if platform else self.job_id_suffix
        )

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def is_checkpointing_enabled(self) -> bool:
        return bool(
            self.change_detection_config.enabled
            and self.state_provider.is_stateful_ingestion_configured()
            and self.stateful_ingestion_config
            and self.stateful_ingestion_config.enabled
        )

    def create_checkpoint(
        self,
    ) -> Optional[Checkpoint[ResourceFingerprintCheckpointState]]:
        if not self.is_checkpointing_enabled():
            return None
        assert self.pipeline_name is not None
        return Checkpoint(
            job_name=self.job_id,
            pipeline_name=self.pipeline_name,
            run_id=self.run_id,
            state=ResourceFingerprintCheckpointState(),
        )

    def _get_last_entry(self, resource_urn: str) -> Optional[ResourceFingerprintEntry]:
        if not self.is_checkpointing_enabled():
            return None
        last_checkpoint = self.state_provider.get_last_checkpoint(
            self.job_id, ResourceFingerprintCheckpointState
        )
        if not last_checkpoint:
            return None
        return last_checkpoint.state.resources.get(resource_urn)

    def should_skip_resource(
        self, resource_urn: str, current_fingerprint: Optional[str]
    ) -> bool:
        if not self.is_checkpointing_enabled() or current_fingerprint is None:
            return False
        entry = self._get_last_entry(resource_urn)
        if entry is None or entry.fingerprint != current_fingerprint:
            return False
        max_age_seconds = (
            self.change_detection_config.full_refresh_interval_hours * 3600
        )
        return (time.time() - entry.last_full_refresh_at) <= max_age_seconds

    def carry_forward_resource(self, resource_urn: str) -> List[str]:
        """
        Preserves the previously known entity urns for a resource whose full
        processing is being skipped this run, so it survives into the new
        checkpoint. Returns those urns so the caller can register them with
        the stale-entity-removal handler.
        """
        entry = self._get_last_entry(resource_urn)
        if entry is None:
            return []
        self._set_current_entry(resource_urn, entry)
        return entry.entity_urns

    def record_full_refresh(
        self, resource_urn: str, fingerprint: str, entity_urns: List[str]
    ) -> None:
        if not self.is_checkpointing_enabled():
            return
        self._set_current_entry(
            resource_urn,
            ResourceFingerprintEntry(
                fingerprint=fingerprint,
                entity_urns=sorted(set(entity_urns)),
                last_full_refresh_at=time.time(),
            ),
        )

    def _set_current_entry(
        self, resource_urn: str, entry: ResourceFingerprintEntry
    ) -> None:
        if not self.is_checkpointing_enabled():
            return
        cur_checkpoint = self.state_provider.get_current_checkpoint(self.job_id)
        assert cur_checkpoint is not None
        cur_state = cast(ResourceFingerprintCheckpointState, cur_checkpoint.state)
        cur_state.resources[resource_urn] = entry
