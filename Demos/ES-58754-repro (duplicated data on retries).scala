// Databricks notebook source
// MAGIC %md ## Reproduction for [ES-58754](https://databricks.atlassian.net/browse/ES-58754)
// MAGIC 
// MAGIC This notebook shows how we can lose/duplicate data if you repartition (shuffle) data using an order dependent expression.
// MAGIC 
// MAGIC ### RCA
// MAGIC An order dependent expression is typically a leaf expression that produces data for a record based on the records' position in the partition instead of using the records' data. Examples of order dependent expressions are: `rand(..)`, `randn(..)`, `uuid()`, `shuffle(..)` & `monotonically_increasing_id()`.
// MAGIC 
// MAGIC We read shuffle blocks in a random order. This is partially by design (requests are send out in random order to avoid pathological access patterns), and it is ultimately determined by the order in which the shuffle blocks arrive. This changes the order in which we process records changes every time we (re)compute a partition. This causes order dependent functions to return different results for the same record every time we recompute a partition.
// MAGIC 
// MAGIC If we use the result of an order dependent expression for shuffling, the record can be shuffled into different reduce partition every time the partition is computed. In itself this would not be a problem if the downstream reduce stage would only read data written by a single attempt of the map partition. Unfortunately this is not the case, the following scenario can unfold:
// MAGIC 1. We compute shuffle map stage 1 (attempt 0). This has 4 partitions. Map stage 1 is reading from a shuffle itself and shuffles data using an order dependent expression.
// MAGIC 2. We start to compute reduce stage 2 (attempt 0). This also has 4 partitions. Reduce stage 2 successfully computes two partitions 0 & 1.
// MAGIC 3. We lose one of the nodes, and as a result we also lose partition 2 & 3 for map stage 1.
// MAGIC 4. We recompute partitions 2 & 3 for map stage 1 (attempt 1). This now shuffles records into different reduce partitions. This means previously computed reduce stage partitions 0 & 1 do contain data that was written for them by the second attempt of map stage 1, and we lose data here because we do not recompute partitions 0 & 1.
// MAGIC 5. We compute the remaining partitions (2 & 3) for reduce stage 2. This now reads records that were previously assigned to partitions 0 & 1 (data duplication).
// MAGIC 
// MAGIC In theory this can also happen for scans, there is nothing that mandates that scans need to produce data in the same order on a retry. File based datasources in general do not have this problem. More exotic datasource, e.g. JDBC, probably have.
// MAGIC 
// MAGIC This problem is a generalization of the problem described in [SPARK-23207](https://issues.apache.org/jira/browse/SPARK-23207).
// MAGIC  
// MAGIC Nielsen hit this particular problem because of the following reasons:
// MAGIC - The job they were running was extreme resource heavy and it was not optimally tuned (too few partitions, resulting in 20GB per partition). The led to a couple of executors failing under memory pressure. These failures triggered retries of various shuffle stages.
// MAGIC - The job was repartitioning its data just before writing using an order dependent expression: `val repartitionedDf = final_data_with_lesser_cols.select(final_data_with_lesser_cols_lower:_*).repartition($"promotypeid",$"tableid", $"period_id", round(rand() * 10))`
// MAGIC 
// MAGIC ### Short term mitigation
// MAGIC Fixes the user can apply to mitigate the issue:
// MAGIC - Stop using order dependent expressions (or fields that were computed these expressions) in shuffle expressions (i.e. no `rand()` and friends). This might be hard to eliminate completely as it commonly used as a mechanism to spread out records accross partitions. For delta we can use optimized writes to solve this, for other data sources we can use the hash of the complete record (provided you don't have too many duplicates).
// MAGIC - Add a sort on all fields before you compute any order dependent function. This is expensive to do, and if you have map fields this might not work.
// MAGIC 
// MAGIC ### Long term fixes
// MAGIC All the fixes described in this section require significant engineering effort (2 weeks or more), and have not been completely flushed out:
// MAGIC - Add checks to the analyzer that fail when a user tries to shuffle using an order based expression. This might not be completely uncontroversial, so this warrants some discussion.
// MAGIC - Mark stages that have this problem as indeterminate. The scheduler will then retry all affected stages completely. This is a bit tricky with result stages that write data, however you should be fine if you use the default commit protocols provided by Databricks. This would extend the fixes made for [SPARK-23207](https://issues.apache.org/jira/browse/SPARK-23207) & [SPARK-23243](https://issues.apache.org/jira/browse/SPARK-23243).
// MAGIC - Make shuffle deterministic. This would also force us to make scans deterministic. Both are non-trivial to do. For the shuffle we need to buffer data when it arrives out of order, this increases the memory used by the shuffle, and make reading shuffles slower because we need to wait for earlier blocks to arrive. For the scan we need to make it declare whether it produces records in a deterministic order, and if it does not we would need to insert a sort to make it deterministic, which is very expensive.
// MAGIC 
// MAGIC ### Prerequisites for running this notebook
// MAGIC - There need to be more than one node in the cluster.
// MAGIC - The shuffle service needs to be disabled (set `spark.shuffle.service.enabled` to `false` when starting the cluster).

// COMMAND ----------

// MAGIC %md ## DO NOT USE THIS NOTEBOOK ON A SHARED CLUSTER

// COMMAND ----------

import org.apache.spark.TaskContext
import org.apache.spark.sql.functions._
import org.apache.spark.util.Utils

// Nuke the executor that touches a partition in a stage for the first time. The loss of the executor
// forces us to recompute all the shuffles that were stored on that node. 
def nukeExecutorForPartition(partitionId: Int) = udf { (id: Long) =>
  val ctx = TaskContext.get()
  // Kill the executor process on the first attempt
  if (ctx.partitionId == partitionId && ctx.attemptNumber == 0 && ctx.stageAttemptNumber == 0) {
    System.exit(0)
  }
  id
}

// COMMAND ----------

// Order dependent (non-deterministic) partitioning expression. This illustrates the buggy behavior.
val (cnt, key_cntd, value_cntd) = spark
  .range(1, 1000000, 1, 100)
  .repartition(100, $"id" % 10)
  .withColumn("value", concat(lit("id_"), $"id"))
  .repartition(100, $"id", round(rand(31) * 10)) // <-- Repartition by a order dependent expression
  .select(nukeExecutorForPartition(99)($"id").as("key"), $"value")
  .groupBy()
  .agg(
    count(lit(1)).as("cnt"),
    countDistinct($"key").as("key_cntd"),
    countDistinct($"value").as("value_cntd"))
  .as[(Long, Long, Long)]
  .head
if (cnt != 999999 && key_cntd != 999999 && value_cntd != 999999) {
  println("Incorrect Result")
} else {
  println("Correct Result")
}

// COMMAND ----------

// Deterministic partitioning expression. This illustrates the correct behavior.
val (cnt, key_cntd, value_cntd) = spark
  .range(1, 1000000, 1, 100)
  .repartition(100, $"id" % 10)
  .withColumn("value", concat(lit("id_"), $"id"))
  .repartition(100, $"value") // <-- Repartition by a normal (data dependent) expression
  .select(nukeExecutorForPartition(99)($"id").as("key"), $"value")
  .groupBy()
  .agg(
    count(lit(1)).as("cnt"),
    countDistinct($"key").as("key_cntd"),
    countDistinct($"value").as("value_cntd"))
  .as[(Long, Long, Long)]
  .head
if (cnt != 999999 && key_cntd != 999999 && value_cntd != 999999) {
  println("Incorrect Result")
} else {
  println("Correct Result")
}

// COMMAND ----------

// Order dependent (non-deterministic) partitioning expression + total ordering. This illustrates a potential fix
val (cnt, key_cntd, value_cntd) = spark
  .range(1, 1000000, 1, 100)
  .repartition(100, $"id" % 10)
  .withColumn("value", concat(lit("id_"), $"id"))
  .sortWithinPartitions($"id", $"value") // <-- Add total ordering
  .repartition(100, round(rand(31) * 100))
  .select(nukeExecutorForPartition(99)($"id").as("key"), $"value")
  .groupBy()
  .agg(
    count(lit(1)).as("cnt"),
    countDistinct($"key").as("key_cntd"),
    countDistinct($"value").as("value_cntd"))
  .as[(Long, Long, Long)]
  .head
if (cnt != 999999 && key_cntd != 999999 && value_cntd != 999999) {
  println("Incorrect Result")
} else {
  println("Correct Result")
}

// COMMAND ----------

// Full hash partitioning. This illustrates a potential fix
val (cnt, key_cntd, value_cntd) = spark
  .range(1, 1000000, 1, 100)
  .repartition(100, $"id" % 10)
  .withColumn("value", concat(lit("id_"), $"id"))
  .repartition(100, $"id", hash($"id", $"value") % 20) // <-- use `hash($"id", $"value") % 20` instead of `round(rand(31) * 10)`
  .select(nukeExecutorForPartition(99)($"id").as("key"), $"value")
  .groupBy()
  .agg(
    count(lit(1)).as("cnt"),
    countDistinct($"key").as("key_cntd"),
    countDistinct($"value").as("value_cntd"))
  .as[(Long, Long, Long)]
  .head
if (cnt != 999999 && key_cntd != 999999 && value_cntd != 999999) {
  println("Incorrect Result")
} else {
  println("Correct Result")
}

// COMMAND ----------

