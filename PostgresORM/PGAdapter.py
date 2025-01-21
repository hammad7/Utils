from typing import Dict, Any, Union, Type, List, Tuple, Optional
from pydantic import BaseModel
from collections import defaultdict
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed
import asyncpg
from asyncpg import ConnectionFailureError, InterfaceError, ConnectionDoesNotExistError

from initiative_recommendations.config import CLIENT_TO_PG_URI
from initiative_recommendations.ef_permutation.utils.custom_logger import logger

read_retries = 2
write_retries = 1
wait = 1


def log_retry_attempt(retry_state):
    logger.info(f"Retry attempt number: {retry_state.attempt_number}")


class PGAdapter:
    """
    Provides an interface for  asynchronously interacting and maintaining multiple Postgres databases connection pools.
    No domain/service specific logic in this file. This file is called in DB.py having custom domain specific logic

    Usage:
    ```python
    from initiative_recommendations.database.PGAdapter import pg_adapter

    # Define the table identifier
    table_id = {
        "client_db_str": <org_id>,
        "db": CLIENT_TO_PG_URI.get(<org_id>)["db"],
        "schema": "carbonvault",
        "table": "VersionMigrationDoc",
    }

    # Query the table (No need to worry about initialization, error handling, connection pool, etc)
    documents = await pg_adapter.find_by(
        table_id,
        {"hash_id": "6ab4e69bf569020eb79ef13493d13bfe"}, # where clause
        None, # select clause
        Document ## Document is extended from pydantic.BaseModel
    )

    ```
    """

    _pools = defaultdict(lambda: None)
    _instance = None

    def __new__(cls):
        # Always create a single instance of PGAdapter
        if cls._instance is None:
            cls._instance = super(PGAdapter, cls).__new__(cls)
        return cls._instance

    async def get_connection_pool(
        self, table_id: Dict[str, str], force_new_pool: bool = False, **pool_kwargs: Any
    ) -> Optional[asyncpg.pool.Pool]:
        """
        Retrieves or creates a connection pool for the specified client database.

        Args:
            table_id (Dict[str, str]): A dictionary containing the database configuration,
                                    expected to have a key 'client_db_str' that identifies
                                    the client database.
            force_new_pool (bool): A flag indicating whether to force the creation of a new
                                connection pool even if one already exists. Defaults to False.
            **pool_kwargs (Any): Additional keyword arguments to be passed to the asyncpg.create_pool
                                function.

        Returns:
            Optional[asyncpg.pool.Pool]: The connection pool for the specified client database (org),
                                        or None if the client database identifier is invalid.
        """

        client_db_str = table_id.get("client_db_str", None)

        if client_db_str is None:
            raise ValueError("Missing 'client_db_str' key in the database configuration.")

        # Check if the provided client database identifier is valid
        if client_db_str not in CLIENT_TO_PG_URI:
            logger.error(f"Invalid client: {client_db_str}")
            return None

        ## For Debugging only
        ## Not required on prod as this will double the query load
        # # If a connection already exists and we're not forcing a new one, check before reusing it
        # if client_db_str in self._pools and not force_new_pool:
        #     try:
        #         logger.info(f"Retrieving existing connection: {client_db_str}")
        #         async with self._pools[client_db_str].acquire() as conn:
        #             await conn.execute('SELECT 1')
        #         # return self._pools[client_db_str]
        #     except Exception as e:
        #         logger.info(f"Connection is stale: {client_db_str}, Error: {e}")
        #         del self._pools[client_db_str]
        #         pass  # Intentionally ignore the exception to proceed with creating a new connection below

        # If a pool already exists and force_new_pool is True, close the pool
        if force_new_pool and self._pools[client_db_str] is not None:
            await self.close_pool(client_db_str)

        # If there's no existing pool, create one
        if self._pools[client_db_str] is None:
            URI = CLIENT_TO_PG_URI.get(client_db_str)["uri"]
            self._pools[client_db_str] = await asyncpg.create_pool(URI, timeout=10, **pool_kwargs)

        return self._pools[client_db_str]

    async def close_pool(self, client_db_str: str) -> None:
        """
        Closes the connection pool associated with the given client and removes it from the internal dictionary.

        Args:
            client_db_str: The identifier of the client whose pool should be closed.
        """
        if client_db_str in self._pools:
            try:
                await self._pools[client_db_str].close()
            except Exception as e:
                logger.info(f"Error closing connection-- {e}")
            finally:
                del self._pools[client_db_str]  # Clear reference to avoid memory leaks

    @retry(
        stop=stop_after_attempt(read_retries),
        wait=wait_fixed(wait),
        before_sleep=log_retry_attempt,
    )  # Try n times with m-second delay
    async def _execute_read_with_retry(self, db_config: Dict[str, str], operation, *args, **kwargs):
        return await self._execute_with_retry(db_config, operation, read_retries, *args, **kwargs)

    @retry(
        stop=stop_after_attempt(write_retries),
        wait=wait_fixed(wait),
        before_sleep=log_retry_attempt,
    )
    async def _execute_write_with_retry(self, db_config: Dict[str, str], operation, *args, **kwargs):
        return await self._execute_with_retry(db_config, operation, write_retries, *args, **kwargs)

    async def _execute_with_retry(self, db_config: Dict[str, str], operation, retries, *args, **kwargs) -> Any:
        """
        Internal method to execute an operation with retry logic, handling connection errors and forcing new connections if necessary.

        Args:
            db_config: Database configuration dictionary.
            operation: The asynchronous operation to execute.
            *args: Positional arguments to pass to the operation.
            **kwargs: Keyword arguments to pass to the operation.

        Returns:
            The result of the operation.

        """
        try:
            pool = await self.get_connection_pool(db_config)
            return await operation(pool, *args, **kwargs)
        except (
            ConnectionFailureError,
            InterfaceError,
            ConnectionDoesNotExistError,
        ) as e:
            logger.error(
                f"Query error for {db_config}: {e}, Proceeding to re-create pool with conf: retry: {retries}, wait: {wait}"
            )
            # Force a new connection pool on retry, this  can impact running queries
            await self.get_connection_pool(db_config.get("client_db_str"), force_new_pool=True)
            raise e  # Let tenacity retry the operation
        except Exception as e:
            logger.error(
                f"Query error for {db_config}: {e}, Proceeding to retry with conf: retry: {retries}, wait: {wait}"
            )
            await self.get_connection_pool(db_config.get("client_db_str"))
            raise e  # Let tenacity retry the operation

    async def find_by(
        self,
        db_config: Dict[str, str],
        where_clause: Dict[str, Union[Type[List], Type[Dict], Type[Any]]],
        projection: List[str] = None,
        document_model: Union[Type[BaseModel], Type[str], Type[dict]] = dict,
    ) -> Union[List[Dict[str, Any]], List[BaseModel]]:
        """
        Retrieves documents from the database based on a where clause.

        Args:
            db_config: Database configuration dictionary.
            where_clause: A dictionary representing the WHERE conditions for the query.
            projection: A list of fields to retrieve (optional, defaults to all fields).
            document_model: The pydantic model to use for representing the retrieved documents
                (optional, defaults to a dictionary).

        Returns:
            A list of documents matching the where clause, either as dictionaries or instances
            of the specified pydantic class.
        """
        try:
            table, schema = db_config["table"], db_config["schema"]
            where_clause_str, values = PGAdapter._construct_where(where_clause)
            query = f"""SELECT {', '.join(projection) if projection else '*'} FROM {schema}."{table}" WHERE {where_clause_str}"""

            logger.info(f"QUERY- {query} VALUES-{values}")
            documents = await self._execute_read_with_retry(db_config, PGAdapter._create_fetch_lambda(query), *values)

            if document_model is dict:
                return [dict(record) for record in documents]
            else:
                return [document_model(**doc) for doc in documents]
        except RetryError as e:
            pass

    @classmethod
    def _construct_where(
        cls, where_clause: Dict[str, Union[Type[List], Type[Dict], Type[Any]]], table: str
    ) -> Tuple[str, List[Any]]:
        """
        Constructs a WHERE clause string and a list of values for parameterized queries.

        Args:
            where_clause: A dictionary representing the WHERE conditions.

        Returns:
            A tuple containing the WHERE clause string and the list of values.
        """
        conditions = []
        values = []

        for key, value in where_clause.items():
            if isinstance(value, list):
                # If the value is a list, use the IN clause
                placeholders = ", ".join(f"${i + 1 + len(values)}" for i in range(len(value)))
                field = '"' + table + '"."' + key + '"'
                conditions.append(f"{field} IN ({placeholders})")
                values.extend(value)
            elif isinstance(value, dict):
                # If the value is a dict i.e jsonb, use the >># clause
                jsonb_clauses, jsonb_vals, _ = cls._jsonb_where_clause({key: value}, None, len(values) + 1)
                jsonb_where_str = " and ".join(jsonb_clauses)
                conditions.append(jsonb_where_str)
                values.extend(jsonb_vals)
            else:
                # If the value is a single item, use the equality clause
                field = '"' + table + '"."' + key + '"'
                conditions.append(f"{field} = ${len(values) + 1}")
                values.append(value)

        where_statement = " AND ".join(conditions)
        return where_statement, values

    @classmethod
    def _jsonb_where_clause(cls, where_dict, parent_keys=None, param_index=1):
        """
        Note that the final values are always text list.
        Eg {"a":{"b":"c":["34.56"]}} ->  a.b.c = 34.56

        Args:
            where_dict: A dictionary representing the WHERE conditions.
        Returns:
            clauses, parameters, param_index
        """
        if parent_keys is None:
            parent_keys = []

        clauses = []
        parameters = []

        for key, value in where_dict.items():
            current_keys = parent_keys + [key]
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                sub_clauses, sub_params, param_index = cls._jsonb_where_clause(value, current_keys, param_index)
                clauses.extend(sub_clauses)
                parameters.extend(sub_params)
            elif isinstance(value, list):
                # Generate clause with placeholders
                path = "#>>ARRAY[" + ",".join([f"'{k}'" for k in current_keys[1:]]) + "]"
                placeholders = ", ".join([f"${i}" for i in range(param_index, param_index + len(value))])
                clause = f'"{current_keys[0]}"{path} in ({placeholders})'
                clauses.append(clause)
                parameters.extend(value)
                param_index += len(value)
            else:
                raise ValueError(f"Unsupported value type: {type(value)} for key {key}")

        return clauses, parameters, param_index

    @staticmethod
    def _create_fetch_lambda(query: str):
        """
        Creates an asynchronous function to execute a fetch operation using a connection from the pool.
        """

        async def fetch_operation(pool: asyncpg.Pool, *args: Any, **kwargs: Any) -> List[asyncpg.Record]:
            async with pool.acquire() as connection:
                # await connection.set_type_codec("json", encoder=json.dumps, decoder=json.loads)
                return await connection.fetch(query, *args, **kwargs)

        return fetch_operation

    @staticmethod
    def _create_execute_lambda(query: str):
        """
        Creates an asynchronous function to execute a executemany operation using a connection from the pool.
        """

        async def execute_operation(pool: asyncpg.Pool, *args: Any, **kwargs: Any):
            async with pool.acquire() as connection:
                #         async with connection.transaction():
                await connection.executemany(query, *args, **kwargs)

        return execute_operation

    async def insert(
        self,
        db_config: Dict[str, str],
        docs: Union[List[Type[BaseModel]], List[Dict]],
    ):
        """
        Inserts data into the specified PostgreSQL table.

        Args:
            db_config: A dictionary containing the database configuration.
                - "table": The name of the table to insert into.
                - "schema": The schema the table belongs to.
            docs: A list of either Pydantic BaseModels or dictionaries representing the data to be inserted.

        """
        try:
            table, schema = db_config["table"], db_config["schema"]
            if docs:
                columns = list(docs[0].__fields__.keys()) if isinstance(docs[0], BaseModel) else list(docs[0].keys())

                values = [tuple(getattr(doc, col) for col in columns) for doc in docs]
                values_placeholder = ", ".join(["$" + str(i + 1) for i in range(len(columns))])
                query = f'INSERT INTO {schema}."{table}" ({", ".join(columns)}) VALUES ({values_placeholder})'

                # logger.info(f"QUERY- {query} VALUES-{values} ")
                return await self._execute_write_with_retry(db_config, PGAdapter._create_execute_lambda(query), values)

        except RetryError as e:
            pass

    async def update(
        self,
        db_config: Dict[str, str],
        where_clause: Dict[str, Union[Type[List], Type[Any]]],
        document: Union[Type[BaseModel], Type[dict]],
    ):
        """
        Updates data in a specified PostgreSQL table based on a where clause.

        Args:
            db_config: A dictionary containing the database configuration.
                - "table": The name of the table to update.
                - "schema": The schema the table belongs to.
            where_clause: A dictionary representing the where clause conditions for the update.
                Keys are column names, values can be:
                    - A list of values for an "IN" clause.
                    - Any other value for a direct equality comparison.
            document: Either a Pydantic BaseModel or a dictionary representing the data to be updated.

        """
        try:
            table, schema = db_config["table"], db_config["schema"]
            doc = document
            if isinstance(document, BaseModel):
                doc = document.dict(exclude_unset=True)  # Only include fields with values

            ## unset the doc keys/fields common where clause, as they need not be updated
            for field in list(doc.keys()):
                if field in where_clause:
                    del doc[field]

            columns = list(doc.keys())

            set_clause = ", ".join([f"{col} = ${i+1}" for i, col in enumerate(columns)])
            where_conditions = " AND ".join(
                [
                    (
                        f"{col} IN ({', '.join([f'${i+len(doc)+j+1}' for j in range(len(where_clause[col]))])})"
                        if isinstance(where_clause[col], list)
                        else f"{col} = ${i+len(doc)+1}"
                    )
                    for i, col in enumerate(where_clause)
                ]
            )
            query = f'UPDATE {schema}."{table}" SET {set_clause} WHERE {where_conditions}'
            params = (
                list(doc.values())
                + [val for col_values in where_clause.values() if isinstance(col_values, list) for val in col_values]
                + [
                    val
                    for col_values in where_clause.values()
                    if not isinstance(col_values, list)
                    for val in [col_values]
                ]
            )
            # logger.info(f"QUERY- {query} PARAMS-{params} ")
            return await self._execute_write_with_retry(
                db_config, PGAdapter._create_execute_lambda(query), [tuple(params)]
            )

        except RetryError as e:
            pass

    async def fetch(
        self,
        db_config: Dict[str, str],
        raw_query: str,
        document_model: Union[Type[BaseModel], Type[str], Type[dict]] = dict,
    ) -> Union[List[Dict[str, Any]], List[BaseModel]]:
        """
        Use this for complex queries/pipelines. Prefer find_by whereever possible.

        Args:
            db_config: Database configuration dictionary.
            raw_query: A string of postgres SQL.
            document_model: The pydantic model to use for representing the retrieved documents
                (optional, defaults to a dictionary).

        Returns:
            A list of documents matching the where clause, either as dictionaries or instances
            of the specified pydantic class.
        """
        try:
            documents = await self._execute_read_with_retry(db_config, PGAdapter._create_fetch_lambda(raw_query))

            if document_model is dict:
                return [dict(record) for record in documents]
            else:
                return [document_model(**doc) for doc in documents]
        except RetryError as e:
            pass


async def init_json_decoder(conn):
    """Register a custom JSON decoder for `jsonb`."""

    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",  # Schema for built-in types
        format="text",
    )


pg_adapter = PGAdapter()
