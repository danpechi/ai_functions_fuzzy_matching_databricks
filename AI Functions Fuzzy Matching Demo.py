# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Fuzzy Matching Documents
# MAGIC
# MAGIC This demo walks through identifying documents that have some matches in an existing data store
# MAGIC
# MAGIC We do this through a pipeline involving Mosaic AI Query and Vector Search. 
# MAGIC
# MAGIC - With AI Query, we run an LLM with batch inference to quickly generate summaries.
# MAGIC
# MAGIC - With Mosaic Vector Search, we can retrieve similar summaries
# MAGIC
# MAGIC - Finally, we can verify the potential fuzzy matching documents with AI query

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Data Setup

# COMMAND ----------

# dbutils.widgets.text("endpoint_name", "databricks-meta-llama-3-1-8b-instruct", "Endpoint Name")
dbutils.widgets.dropdown("lm_endpoint_name", "databricks-meta-llama-3-1-8b-instruct", ["databricks-meta-llama-3-1-8b-instruct","databricks-meta-llama-3-3-70b-instruct", "databricks-claude-3-7-sonnet"], "LM Endpoint Name")
dbutils.widgets.text("embedding_endpoint_name", "databricks-gte-large-en", "Embedding Endpoint Name")
dbutils.widgets.text("catalog_name", "batch", "Data UC Catalog")
dbutils.widgets.text("schema_name", "dpechi", "Data UC Schema")
dbutils.widgets.text("table_name", "star_trek_episode_text", "Data UC Table Base")

lm_endpoint_name = dbutils.widgets.get("lm_endpoint_name")
embedding_endpoint_name = dbutils.widgets.get("embedding_endpoint_name")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
table_name = dbutils.widgets.get("table_name")

# COMMAND ----------

import pandas as pd
df = pd.read_csv('TNG.csv', encoding='latin1')

# COMMAND ----------

df.head()

# COMMAND ----------

df_agg = df.groupby(['episode', 'act', 'who']).agg({'text': ' '.join}).reset_index()
df_agg

# COMMAND ----------

df_agg2 = df.groupby(['episode', 'act']).agg({'text': ' '.join}).reset_index()
df_agg2

# COMMAND ----------

spark_df = spark.createDataFrame(df_agg)
spark_df.write.mode('overwrite').saveAsTable(".".join([catalog_name, schema_name, table_name]))

spark_df = spark.createDataFrame(df_agg2)
spark_df.write.mode('overwrite').saveAsTable(".".join([catalog_name, schema_name, f"{table_name}_act"]))

# COMMAND ----------

spark.sql(f"""
  select 
  (select count(*) from {".".join([catalog_name, schema_name, table_name])}) as count_text2,
  (select count(*) from {".".join([catalog_name, schema_name, table_name+"_act"])}) as count_text_act
  """).display()

# COMMAND ----------

spark.sql(f"""
          select * from {".".join([catalog_name, schema_name, table_name])}
          """).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Summarizing Documents with DBSQL + AI Query

# COMMAND ----------

import time

#9500 tok/sec endpoint - can be increased with higher throughput endpoint

start_time = time.time()

command = f"""
    SELECT text,  
    ai_query(
        '{lm_endpoint_name}', -- endpoint name
        CONCAT('Provide a 2-3 sentence synopsis of the characters lines from the episode:', text)
    ) AS act_synopsis
    FROM {".".join([catalog_name, schema_name, table_name])}
"""

result = spark.sql(command)

display(result)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Building our Vector Store with Embedded Document Summaries

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

delta_table_name = f"{catalog_name}.{schema_name}.{table_name}_delta"
vector_search_endpoint_name = "synopsis_endpoint"
vs_index_fullname = f"{catalog_name}.{schema_name}.{table_name}_index"

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

result_with_id = result.withColumn("id", monotonically_increasing_id())
result_with_id.write.mode('overwrite').format("delta").option("delta.enableChangeDataFeed", "true").saveAsTable(delta_table_name)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

#building vector store
vsc = VectorSearchClient()

vsc.create_endpoint(
    name=vector_search_endpoint_name,
    endpoint_type="STANDARD"
)

endpoint = vsc.get_endpoint(
  name=vector_search_endpoint_name)

index = vsc.create_delta_sync_index(
  endpoint_name=vector_search_endpoint_name,
  source_table_name=delta_table_name,
  index_name=vs_index_fullname,
  pipeline_type='TRIGGERED',
  primary_key="id",
  embedding_source_column="act_synopsis",
  embedding_model_endpoint_name=embedding_endpoint_name
)
index.describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Embedding our new documents, and retrieving similar ones from the vector store

# COMMAND ----------

spark.sql(
  f"""
  create table if not exists {delta_table_name+"_emb"} as
select ai_query(embedding_endpoint_name, act_synopsis) as search_input, act_synopsis from {delta_table_name}
""")

# COMMAND ----------

spark.sql(
  f"""
  select * from {delta_table_name+"_emb"} limit 5
  """)

# COMMAND ----------

spark.sql(
  f"""
SELECT {delta_table_name + "_emb"}.search_input AS search_input, 
       {delta_table_name + "_emb"}.act_synopsis AS search_input_text, 
       search.act_synopsis AS search_result 
FROM {delta_table_name + "_emb"},
LATERAL (
  SELECT * FROM vector_search(
    index => '{vs_index_fullname}', 
    query_vector => {delta_table_name + "_emb"}.search_input, 
    num_results => 1
  )
) AS search
"""
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Validating that these are fuzzy matches with AI Query

# COMMAND ----------

matching_prompt = "Validate that these 2 summaries are matches. Provide only a yes or no."

final_matches = _sqldf.selectExpr("search_input_text", "search_result", f"ai_query('{lm_endpoint_name}', CONCAT('{matching_prompt}', 'Summary 1:', search_input_text, 'Summary 2:', search_result)) as output")


final_matches.display()
