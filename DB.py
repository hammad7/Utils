from typing import Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from initiative_recommendations.database.PGAdapter import pg_adapter

from initiative_recommendations.config import CLIENT_TO_PG_URI
from initiative_recommendations.ef_permutation.utils.custom_logger import logger

# VERSION_MIGRATION_DOC_ID = {
#     "client_db_str": client_id,
#     "db": CLIENT_TO_PG_URI.get(client_id)["db"],
#     "schema": CLIENT_TO_PG_URI.get(client_id)["schema"],
#     "table": "VersionMigrationDoc",
# }


class VersionMigrationDB(BaseModel):
    """
    Backward compatible Object used to store the version of the migration.
    """

    mongo_id: str = Field(default_factory=str)
    description: str
    hash_id: str
    date_file_created: datetime
    nb_inserts: int = 0
    time_insertion_seconds: float = 0.0

    def __getitem__(self, item):  ## else beanie doc not accible using ['']
        return getattr(self, item)


class DB:
    """Domain/service-specific queries related to Initative Recomentdation."""

    def __init__(self):
        pass

    async def get_efdbs(self, TABLE_ID: Dict[str, str]) -> Optional[Dict]:
        """
        Retrieve the version of the migration from the Postgres.

        Parameters
        ----------
        VERSION_MIGRATION_DOC_ID: PG Table Tdentier
        id: str

        Returns
        -------
        Optional[VersionMigrationDB]
        """
        query = """SELECT
            source->>'name' AS source,
            "isCustom",
            array_agg(
                DISTINCT concat_ws(
                    '.',
                    (source->'version'->>'major')::text,
                    (source->'version'->>'minor')::text,
                    (source->'version'->>'patch')::text
                )
            ) AS versions
        FROM
            "carbon_vault"."efdbs"
        WHERE
            "isDisabled" IS NOT TRUE
        GROUP BY
            source->>'name', "isCustom"
        ORDER BY
            source->>'name';"""
        efdbs = await pg_adapter.fetch(TABLE_ID, query)
        if not efdbs:
            efdbs = []
        return {"data": {"efdbSourceVersions": efdbs}}

    async def retrieve_version_migration(
        self, VERSION_MIGRATION_DOC_ID: Dict[str, str], id: str
    ) -> Optional[VersionMigrationDB]:
        """
        Retrieve the version of the migration from the Postgres.

        Parameters
        ----------
        VERSION_MIGRATION_DOC_ID: PG Table Tdentier
        id: str

        Returns
        -------
        Optional[VersionMigrationDB]
        """
        version_migration = await pg_adapter.find_by(
            VERSION_MIGRATION_DOC_ID, {"mongo_id": id}, None, VersionMigrationDB
        )
        if version_migration:
            return version_migration[0]
        return None

    async def upsert_version_migration(
        self, VERSION_MIGRATION_DOC_ID: Dict[str, str], updated_version_migration: VersionMigrationDB
    ) -> VersionMigrationDB:
        """
        By default, we do an upsert of the version of the migration to make sure we are always using the latest one.
        """
        existing_version_migration = await pg_adapter.find_by(
            VERSION_MIGRATION_DOC_ID, {"mongo_id": updated_version_migration.mongo_id}, None, VersionMigrationDB
        )

        if existing_version_migration and existing_version_migration[0].hash_id != updated_version_migration.hash_id:
            # Update existing document if hash_id is different
            return await pg_adapter.update(
                VERSION_MIGRATION_DOC_ID, {"mongo_id": updated_version_migration.mongo_id}, updated_version_migration
            )
        elif not existing_version_migration:
            # Insert new document if not exists
            new_version_migration = await pg_adapter.insert(VERSION_MIGRATION_DOC_ID, [updated_version_migration])
            return new_version_migration
        else:
            # Document with same hash_id exists, no update needed
            return existing_version_migration[0]


db = DB()
