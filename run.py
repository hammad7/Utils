import numpy as np
import pandas as pd
import pickle, asyncpg, json, asyncio, itertools, time, functools, logging

seed = 71

## Test file loading
emb_data = np.load("/Users/hammad/repo/pgvector/target_sentences_encoding.npy")
# print(emb_data[0:2])
# print(len(emb_data))  ## 72354
# print(np.unique(emb_data, axis=0).shape[0])  ## 72170
# print(np.sum(emb_data[0:100] ** 2, axis=1))


with open("/Users/hammad/repo/pgvector/source_unique_sentences.pkl", "rb") as file:
    act_data = pickle.load(file)
act_emb = np.array(list(act_data.values()))


# print(act_emb[0:5 ])
# print(list(act_data.keys())[0:5])
# print(len(act_data))  # 20283
# print(np.unique(act_emb, axis=0).shape[0])  ## 17366

## time logging:

# Configure logging (adjust as needed)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def timeit(func):
    """Decorator to measure and log execution time of a function."""

    @functools.wraps(func)  # Preserves function metadata
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function {func.__name__} took {execution_time:.4f} seconds to execute.")
        return result

    return wrapper


@timeit
async def search_similar_embeddings_batch(pool, query_embeddings, top_k=5, sampleSize=10000):
    """
    Perform a batch search for similar embeddings using cosine similarity.

    Args:
    - connection (asyncpg.connection): The database connection.
    - query_embeddings (list of np.array): A list of query embeddings.
    - top_k (int): The number of top results to return for each query embedding.

    Returns:
    - List of lists: A list of results for each query, containing (id, metadata, distance).
    """
    if sampleSize > len(query_embeddings):
        np.random.seed(seed)
        random_vectors = np.random.uniform(low=-1.0, high=1.0, size=(sampleSize - len(query_embeddings), 384)).astype(
            np.float32
        )
        # Normalize each new vector to have a sum of squares equal to 1
        for i in range(len(random_vectors)):
            norm = np.sqrt(np.sum(random_vectors[i] ** 2))
            random_vectors[i] /= norm
        query_embeddings = np.concatenate((query_embeddings, random_vectors), axis=0).reshape((sampleSize, 384))
    else:
        np.random.seed(seed)
        idc = np.random.choice(query_embeddings.shape[0], size=sampleSize, replace=False)
        query_embeddings = query_embeddings[idc]

    print("Preparing act size: ", len(query_embeddings))

    # Convert the query embeddings to the required format (list of floats)
    query_embeddings = [f"[{', '.join(map(str, embedding.tolist()))}]" for embedding in query_embeddings]

    # Prepare the SQL query for batch processing
    # We will create a union of queries for each query embedding to find the top_k closest embeddings.
    sql = """
    WITH search_results AS (
        SELECT 
            id, embedding, 
            embedding <=> $1 AS distance  
        FROM embeddings
        ORDER BY distance
        LIMIT $2
    )
    SELECT * FROM search_results;
    """
    start_time = time.time()

    # Run the search for all query embeddings concurrently
    async def run_query(query_embedding):
        async with pool.acquire() as connection:
            # print(top_k)
            return await connection.fetch(sql, query_embedding, top_k)

    # Run the queries concurrently using asyncio.gather
    results = await asyncio.gather(*[run_query(query_embedding) for query_embedding in query_embeddings])
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Process the results and return them
    return results, elapsed_time


## pg utils
@timeit
async def batch_insert_embeddings(pool, embeddings, sampleSize=20000):
    """Inserts multiple embeddings and their metadata in a batch."""

    # print("Truncating table.. ")
    async with pool.acquire() as connection:
        await connection.execute(f"TRUNCATE TABLE public.embeddings RESTART IDENTITY")

    if sampleSize > len(embeddings):
        np.random.seed(seed)
        random_vectors = np.random.uniform(low=-1.0, high=1.0, size=(sampleSize - len(embeddings), 384)).astype(
            np.float32
        )
        # Normalize each new vector to have a sum of squares equal to 1
        for i in range(len(random_vectors)):
            norm = np.sqrt(np.sum(random_vectors[i] ** 2))
            random_vectors[i] /= norm
        embeddings = np.concatenate((embeddings, random_vectors), axis=0).reshape((sampleSize, 384))
    else:
        np.random.seed(seed)
        idc = np.random.choice(embeddings.shape[0], size=sampleSize, replace=False)
        embeddings = embeddings[idc]

    print("Preparing embeddings size: ", len(embeddings))

    # await connection.copy_records_to_table(
    #     "embeddings", records=values, columns=["embedding"]  # Specify the columns for the bulk insert
    # )
    ## This format converstion takes most time
    values = [(f"[{', '.join(map(str, embedding.tolist()))}]",) for embedding in embeddings]

    print("Inserting  ... ")
    # Perform a batch insert
    async with pool.acquire() as connection:
        await connection.executemany(
            """
            INSERT INTO public.embeddings (embedding)
            VALUES($1);
            """,
            values,
        )


async def connect_db():
    """Establish an asynchronous connection to the PostgreSQL database."""
    # return await asyncpg.connect(
    #     "postgresql://postgres:postgres@192.168.1.45/postgres",
    #     timeout=15,
    #     # user="postgres", password="postgres", database="postgres", host="192.168.1.45", port=5432
    # )
    # Create a connection pool
    return await asyncpg.create_pool(
        user="postgres",
        password="postgres",
        database="postgres",
        host="192.168.1.45",
        port=5432,
        min_size=10,  # Minimum number of connections in the pool
        max_size=10,  # Maximum number of connections in the pool
    )


async def run_experiment(emb_data_size, act_emb_batch_size, top_k=5):
    ## one time db init
    pool = await connect_db()
    await batch_insert_embeddings(pool, emb_data, emb_data_size)

    # Perform batch search
    results, e_time = await search_similar_embeddings_batch(pool, act_emb, top_k, act_emb_batch_size)
    print(len(results))

    ## Print the results for each query
    # for i, query_result in enumerate(results):
    #     print(f"Results for query {i + 1}:")
    #     for result in query_result:
    #         print(f"  ID: {result['id']}, Cosine Distance: {result['distance']}")

    await pool.close()
    return {
        "emb_data_size": emb_data_size,
        "act_emb_batch_size": act_emb_batch_size,
        # "index_type": index_type,
        "topk": top_k,
        # "elapsed_time": elapsed_time
        "search_time": e_time,
    }


# Connect to the database and perform the batch insert
async def main():
    # Define the parameter values
    param_values = {
        "efdb": [20000, 30000, 40000, 50000, 100000, 150000, 200000],
        "act": [10000, 20000, 30000, 40000, 50000],
        # "efdb": [40000],
        # "act": [30000, 40000, 50000],
        # "index_type": ["ivfflat", "HNSW"], ## not wokring
        "topk": [15, 50, 70, 100],
    }

    # Generate all combinations of parameters using itertools.product
    all_combinations = list(itertools.product(*param_values.values()))

    # Store results in a list
    results = []

    # Run experiments for each combination
    for combination in all_combinations:
        result = await run_experiment(*combination)
        results.append(result)

    # Create a Pandas DataFrame from the results
    df = pd.DataFrame(results)

    # Print the DataFrame or save it to a CSV file
    print(df)


asyncio.run(main())

#
#
#
#
#
## REQUIREMENTS

## postgres docker image:

# apt-get -y update
# apt-get -y install git
# apt-get install make
# apt-get install -y gcc make postgresql-server-dev-17 build-essential

# cd /tmp
# git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
# cd pgvector
# make
# make install

## get docker host i.e 192.168.1.45 from ifconfig en0
## psql -h 192.168.1.45 -U postgres -d postgres -W
# ## type password
# CREATE EXTENSION IF NOT EXISTS vector;

# CREATE TABLE public.embeddings (
#     id SERIAL PRIMARY KEY,
#     embedding vector(384),
#     desciption VARCHAR
# );

# CREATE INDEX ON public.embeddings USING ivfflat (embedding vector_cosine_ops);
# CREATE INDEX ON public.embeddings USING hnsw (embedding vector_cosine_ops);
## ivfflat is the indexing method for efficient vector search.
## vector_cosine_ops is the operator class for cosine similarity.
