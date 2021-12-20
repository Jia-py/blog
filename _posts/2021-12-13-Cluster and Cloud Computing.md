---
layout:     post
title:      "notebook - Cluster and Cloud Computing"
date:       2021-12-13 12:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - HKU CS
---

**by prof. Cho-Li Wang**

# Introduction

* two hands-on experience
* check the CPU model `cat/proc/cpuinfo`
* students need to repeat the spark and hadoop installation for 3 times.
* meeting record - screen shot of every meeting and write down the meeting result.
* **Trouble Shooting Report** : report technical problems and write down how you solve it
* report submitted 7 days before the deadline will be given 5% bonus

Why Cloud?

Cloud: Dynamic resource scaling, instead of provisioning for peak; without cloud, company should balance the capacity with demand carefully.

Virtualization techiniques: VMware Xen KVM

虚拟化技术帮助可以将一台服务器的硬件资源进行切分，但一般每切分出一个独立的单元都需要装一个系统，而系统是很臃肿的（What Hypervisor does, such as Xen, KVM, and VMWare）。所以，docker产生了，可以不用再切分出系统，而是在一个系统上可以运行不同的相互独立的环境 (under one Container Engine (Docker) )。

It's OK to deploy containers on virtual machines.

Cloud Deployment Models

* Public Cloud: Google, Amazon sell it, a rental basis
* Private Cloud: for exclusive use by a single organization of subscribers. more safe.
* Community Cloud: e.g., cryptocurrency mining
* Hybrid Cloud: e.g., private + public

# Cloud Service Models

* infrastructure as a service **Iaas**
  * network virtualization
  * 一个独立的主机，有独立的系统，a virtual machine
* container as a service **Caas**
  * 只是一个container，装一些软件，运行脚本
* platform as a service **Paas**
  * **deploy an entire application**
  * 多是基于云的平台，用户只需要直接写代码就可以或写文字就可以。如印象笔记、spotify。
* function as a service **Faas**
  * **just deploy a single function**： 比如用户模糊手机中的图片，或旋转图片，都是一个function。
  * event-driven: 比如事件为获取倒今天股市走势很糟糕，则运行代码发送伤心的表情。
  * AWS Lambda: write your functions in python..., uploaded as a zip file
* software as a service **SaaS**
  * 给用户使用的软件，比如各种apps，云盘软件，云游戏

# Hadoop

File access model: read/ append only; most reads are sequential

HDFS has two kinds of nodes. Name node, store metadata like names, block locations. Data nodes, store data.

HDFS中的数据只能写入一次，不能修改。因为一旦修改，其他节点存储的该数据的复制项则失去了一致性。

Heartbeats: default every 3s, data node sends heartbeats to name node to report its status.

Block Reports: default every 6 hours, block report provide the name node with data location status.

User can control which keys go to which reducer by implementing a custom Partitioner.

# Problem Shooting

* Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
  * maybe low version of apt-get or unstable internet make this mistake, run `sudo apt-get clean` and run `sudo apt-get update` and redownload it again with like `sudo apt-get -y install sysbench`
* How to run the command in each terminal simultaneously?
  * just use the function of xshell, use the tool-发送键输入到所有会话

# 期末复习

# SaaS-PaaS-IaaS

Infrastructure as a Service (IaaS)

Container as a Service (CaaS) `(newer)`

Platform as a Service (PaaS)

Function as a service (FaaS) `(newer)`

Software as a Service (SaaS)

## IaaS

AWS EC2

the lowest-level control of resources in the cloud

virtual machine, 一台虚拟机，拥有独立的操作系统

<img src="https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211213145651692.png" alt="image-20211213145651692" style="zoom:67%;" />

## CaaS

多个container建立在同一个操作系统上

<img src="https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211213145637224.png" alt="image-20211213145637224" style="zoom: 67%;" />

## PaaS

for developers who want to create services but don't want to build their own cloud infrastructure

customer only need to take care of data and application

## FaaS

AWS Lambda, Google Cloud Functions...

Applications get split up into different functionalities, which are triggered by `events`.

pay for the resources the `functions use`.

## FaaS vs PaaS

PaaS: deploy an entire application. typically running on at least one server

FaaS: deploy some single functions. designed to be a serverless architecture

## SaaS

Google Drive, Youtube...

## 2020 Final Exam Question

![image-20211213152507848](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211213152507848.png)

![image-20211213152517640](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211213152517640.png)

# Hadoop

![image-20211213115229066](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211213115229066.png)

## Overview of Hadoop

Hadoop 2.x = HDFS （1 Namenode + N DataNodes） + YARN （1 ResourceManager + N NodeManagers）

![image-20211212212859227](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211212212859227.png)

一般Namenode与ResourceManager需要部署在不同的机器上，但本节课为了简单，部署在一台机器上。

Yarn：1. ResourceManager 2. NodeManager 3. ApplicationMaster

RM只有一个，分配资源给每个应用，如MapReduce，Spark。

NM是worker，每个node一个，Launch containers。

AM是每个应用一个，负责从RM拿资源，并与NM一起工作运行应用。

**Hadoop framework：**

1. resource manager启动一个container来开启app master
2. app master向resource manager请求资源
3. resource manager通过node manager分配资源

## HDFS

数据被切为64/128MB的块，存储在不同的DataNode上。每个数据块至少要被复制3份以上，即拥有至少3个datanode存储的文件内容是一样的，提高容错。一个datanode上可以存储多个block。

唯一的NameNode不存储数据，只存储metadata（file names, block locations, etc）。**如果NameNode Down了，HDFS就down了。**因此，hadoop3支持了多namenode，将额外的namenode与首个namenode相连，每隔一段时间更新第二个namenode。

NameNode需要确保每个block都有足够的重复量，也需要保证不是所有的replicas都在一个rack（机架，一个rack可以有多个datanode）上。

![image-20211118213848910](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211118213848910.png)

如果replicas多了，即over replicated，需要移除多余的replicas。有两条准则，尽量不减少机架数，即尽量删去在一个机架上存在多个replicas中的一份；二是倾向于删去可用磁盘容量最少的datanode上的replicas。

**Heartbeats & Block Reports**

Heartbeats：默认，每3s DataNode发送一次“我还存活”的消息给NameNode

如果有missing heartbeats了，则通过NameNode可以查到哪些block消失了，则在其他DataNode上重新复制这些block。

Block Reports：默认每6小时，Datanode发送一次“我有block A与C”的消息给NameNode

## MapReduce

large input -- simple operations

**mapreduce design principle: divide and conquer**

map and reduce all run parallel.

Map phase: Master assigns each map task to a worker, produce R local files containing intermediate k/v pairs.

Reduce phase: Master assigns each reduce task to a worker. sorts intermediate k/v pairs from map workers and produce the output.

### Partitioner

Partitioner: determines which intermediate k/v pairs generated by map tasks are to be processed by which reduce tasks.

**Reduce input Constraint**: all intermediate pairs produced by all map tasks `with the same key` should be processed by the `same Reduce` operation in the `same Reduce task`.

Mapreduce Framework: Map -- Shuffle -- Reduce

### Inputsplits

Inputsplits: a logical division of data, represents the data to be processed by an individual map task.

inputsplits数量可以比block数量少

### Locality

1. data on the same node (data-local task)2. different nodes on the same rack (rack-local task)3. different racks in the same center (off-rack task)4. nodes in different data centers(use http)

if data isn't local. Namenode suggests another node without data but in the same rack to run the Map.

### Reduce

shuffle/copy: moving map outputs to the reducers. Each reducer gets the `relevant partitions` of the output from all the mappers. `may causes network traffic`

local sort(on reducer side): merge sort by keys before presented to the reducer

reduce: perform the `actual` reduce function, generate output file stored in HDFS.

![image-20211212215030385](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211212215030385.png)

Each Reducer periodically queries the Application Master for Mapper hosts until it has received a final list of nodes hosting its partitions.

mapreduce.job.reduce.slowstart.completed.maps = 0.05 default, reducer tasks start when 5% of map tasks are complete.

"Coupled relationship": actual reduce function can not be executed before all data arrive

### Number of Mappers and Reducers

\# of mappers: can only be adjusted the blocksize to get the desired # of map tasks.

\# of reducers: job.setNumReduceTasks(11)

### Fault Tolerance

if a task crashes: retry on another node. ok for a map for no dependencies. ok for reduce for map's output are saved on disk.

if a node crashes: re-launch lost tasks on other nodes. need re-run maps for their output files were lost along with the crashed node.

### Speculative Execution

If a task is going slowly (“straggler”), launch a second copy of the task on another node. Take the output of whichever copy finishes first, and kill the other.

mapreduce.map.speculative: recommendation False

### Summary

![image-20211212221514175](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211212221514175.png)

Name Node never becomes a bottleneck for any data IO in the cluster

## Yarn Scheduler

yarn supports non-MapReduce workloads: multiple ways to interact with the data in HDFS, mapreduce, spark, HBase

Yarn Container is a JVM process, `not` K8S Container.

yarn container是按照可用内存再去新建新的container，因此可能会启动container失败。

![image-20211213124607158](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211213124607158.png)

Yarn supports memory-based resource allocation only. most mapreduce app are memory hungry.

![image-20211213132051238](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211213132051238.png)

在一个yarn container中的内存分配：1. heap memory, 用来跑map, reduce 2. Non-Heap Memory, used by JVM to store loaded java classes code and meta-data 3. other, used by JVM itself.

### OOM

OOM: MapReduce jobs tend run into Out Of Memory java errors if a YARN container grows beyond its heap size setting the task will fail.

solution: increase map/reduce memory and java heap size

### GC (Garbage Collection) time

GC is triggered when your heap size is almost full (70%-80%), it will block any other work running in the machine.

Reduce is GC-heavy

# Spark

Weakness of MapReduce: 1. support for Batch Processing only. 2. many problems are not suitable to the two-step process of map and reduce.

Goal of Spark: To design a programming model that supports a much wider class of applications than MapReduce, while retaining the properties of MapReduce(fault tolerance, data locality, scalability).

RDD: in-memory cache - up to 100x faster than MapReduce

One Master node + multiple Worker nodes

## Key elements of a Spark cluster

Spark Driver

Cluster Manager: Yarn

Workers (K8S的一个container)

Executors: are worker nodes' JVM processes (yarn container, 根据cpu几核可以同时处理几个partition)

---

Spark driver - generate SparkContext (stores configuration paramteters) - connect to the cluster through Yarn's resource manager

start spark: 1. spark-submit 2. spark shell

## Deploy Modes

Cluster Mode: the Driver runs inside an Application Master(use one Yarn container)

the best practice is to run the application in cluster mode, spark driver and spark executor are under the supervision of Yarn.

Client mode: Spark driver runs on the host where the job is submitted, can run at any other worker node in the cluster where customer type the command. still need 1 Yarn container, but driver code is not running here.

client mode方便debugging，suffer higher latency if client is not in the cluster.

## RDD

RDD is a `distributed` data structure. is logically partitioned across many nodes.

two types of operations: Transformations (map, union) and Actions (return a value to the driver program, e.g., saveAsTextFile, collect, count)

All transformations in Spark are lazy, RDD is only computed when an Action requires to be returned.

RDDs are immutable: can't be modified once created

An RDD contains list of partitions

RDD keeps a pointer to it's parent RDD(s)

**Two Types of Transformations Narrow & Wide**

---

![image-20211214125421924](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211214125421924.png)

narrow no shuffling, wide shuffle needed -- expensive

narrow是多对一，wide是一对多

join()可以是narrow也可以是wide，narrow没有shuffle，wide有shuffle

![image-20211214130141632](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211214130141632.png)

**RDD Partitions**

---

generally, a partition is created for every HDFS partition of size 64MB/128MB; or the number of files you're reading

types of partitioning in spark: Hash Partitioner (default), Range Partitioner. Custom Partitioner

**spark are lazy**

---

cache(): Spark won’t keep the results in memory unless you tell it explicitly !!

**DAG -- job (an action, a job) -- stages (stages are sequences of RDDs, don't have a shuffle in between) -- tasks (one task for each partition)**

spark are lazy advantages:

1. Saves Computation and increases speed
2. Opportunities of optimization. Spark can make optimization decisions after it had a chance to look at the DAG in entirety.

**In-memory Computing: RDD Cache**

---

use .cache()

executor memory is limited, spark会自动驱逐RDD partitions

cached RDDs can be reused by other jobs of the same application

we store RDDs in the JVM heap space

Three situations you should cache your RDDs:
– Use an RDD many times
– Performing multiple actions on the same RDD
– For long chains of (or very expensive) transformations

## Spark Executor

Hadoop: By default, Hadoop launches a new JVM for each map or reduce task High JVM startup overhead.
Spark: Each executor can run multiple tasks in parallel if enough memory & spark.executor.cores> 1.

When all jobs have finished and the application exits, the executors will be removed.

Spark Executor Memory 即 JVM heap memory

`spark.memory.fraction`: user memory, for storing the user defined data structures

`spark.memory.storageFraction`: used for storing RDD cache

![image-20211214141935419](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211214141935419.png)

![image-20211214142032486](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211214142032486.png)

Execution memory may evict Storage memory, but only up to a threshold set in spark.storage.memoryFraction

## RDD Persistence

rdd.persist()

save the intermediate result,so that we can reuse it if required.

advantages of RDD's Immutable: 1. No consistence issue 2. lost RDD can be recreated deterministically at any point of time

create and cache many RDD instances: increasing memory consumption and putting pressure on the GC.

which storage level to choose?

memory_only > memory_only_ser > spill to disk

## DAG Scheduler

Mission: Partition and creation of stages + stage submission

The Task Scheduler is NOT aware of dependencies between the stages. Only DAGScheduler knows it.

Stages (i.e. their tasks) can run in parallel if there is no dependency between them and there is enough resources in a cluster to run the tasks.

### Locality levels in spark

`Process_local`: data and processing are localized on the same JVM (same executor), can reuse/share cached RDD partitions.

`Node_local`: data and processing are in the same node but on a different executor

`Rack_local`: data is located in another node on the same rack

Scheduling preference:process -> node -> rack -> any

spark.locality.wait(default 3s): Spark will wait to launch a task on an executor local to the data using this value. After this period if the data-local node is still unavailable, Spark will give up and launch the task on anotherless-local node.

### summary

![image-20211214173412567](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211214173412567.png)

## Fault Tolerance in Spark

### RDD Lineage

different to hadoop, nor replicating data. logs lineage(pointer to parent(s)) rather than storing the actual data

Re-compute by default: Spark’s cache is “fault-tolerant” --> if any partition of an RDD is lost, it will automatically be recomputed and put it back in cache.

worker failure: recompute the partitions

Driver failure: RDD lineage is gone, Yarn restarts the Driver (Cluster Mode)

### Checkpoint

saves RDD partitions (data) to a reliable storage system (e.g. HDFS)

The checkpoint file won’t be deleted even after the Spark application terminated.

What kind of RDD needs to be checkpointed?
– the computation takes a long time
– the lineage chain is too long
– RDDs that depends on many parent RDDs

# Kafka

Distributed Streaming Platforms

Broker: handles all requests from clients and keeps data replicated within the cluster

`records`: key-value-timestamp

`topic`: stream of records, a topic is broken up into partitions

`partitions`: append-only and immutable, no size limit, a partition is tied to a broker

![image-20211215163310809](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211215163310809.png)

Messages are sent to the partition's leader.

For each partition that has lost a leader, a follower on a remaining node will be promoted to leader.

If a lost Broker come back later, need leader re-election

Consumer Lag: fast producer, slow consumer. solution: add more consumers. increase buffer size.

## Why is Kafka so fast?

Fast writes: No caching in JVM; Append-only

Fast reads: Read cached copy from page cache directly.

# StructuredStreaming

## Basics

a scalable and fault-tolerant streaming processing engine built on the Spark SQL engine

A schema defines the column names and types of a DataFrame

DataFrames can be significantly faster than RDDs

![image-20211214201447919](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211214201447919.png)

Streaming Queries

Batch Queries: query on static data

output sinks: File sink, `Kafka sink`, Foreach sink

Trigger: used to specify how often a streaming query should be activated to produce results

## Micro-batch Execution Model

micro-batch = "job"

micro-batch execution advantage:

1. Dynamic load balancing
2. Fine-grained fault recovery

disadvantage:

1. a higher latency due to the overhead of launching a DAG of tasks

Continuous Processing has smaller latency, but still has limitations:

1. Supported Queries: Only map-like operations; All SQL functions
2. Not Supported: Watermarks; Aggregate operations

## Windows Operations on Event Time

Spark supports 2 types of window operations:

Tumbling Window: Fixed-size, non-overlapping, gap-less windows

Sliding Window: the window will slide

Watermarking: 允许延迟到达一定范围(delay threshold)内的数据加入聚合; only with `Update Output mode`

## Spark and Kafka Integration

## Fault Tolerance

Challenge: Live data, recovery time must be short

Solution: Checkpoints and Write-Ahead Logs (WAL)

write-ahead: note down the operation in a log file first before performing the operation on data.

for checkpoints, each query must have a different checkpoint directory). The query will save all the progress information(offset) and the running aggregates(“state”) to the checkpoint location (replicated on HDFS).

# Virtualization Technique

Docker relies on two major Linux kernel features:

1. Namespaces to isolate an application's view of the operating environment. limits what you can see
2. Control Groups (cgroups): limits what you can use

## CPU controller example

cpu.shares: the `weight` of each group living in the same hierarchy

cpu.cfs_period_us, cpu.cfs_quota_us

## Kubernetes

tasks done by container orchestrator

e.g., allocation of resources between containers, health monitoring of containers and hosts.

Kubernetes Master: runs on a single node in your cluster : analogous to Yarn’s ResourceManager

Kubelet: the primary “node agent” that runs on each node; analogous toYarn’s NodeManager. creating pods and make sure they are all running

Pod: a pod consists of one or more containers. the smallest deployable units in Kubernetes. each pod is assigned a unique IP address.

the system terminated the containerif the container tried to use more resource than its limit.
In Kubernetes, limits are applied to containers, not pods

### Namespace Quota

Namespace Quota: restrict how much of the cluster resources can be consumed across all pods in a namespace.

### Volume

A Pod uses a Volume for storage

Volumes can be used by all containers in pod, but they must be mounted in each container that needs to access them first.

two basic types of Volumes:

1. Ephemeral volume : have a life time of a pod. 当pod exist, Kubernetes destroys ephemeral volumes.
2. Persistent volume: exist beyond the lifetime of a pod. survive even pod crashes or is deleted.

For any kind of volume in a given pod, data is preserved across container (not Pod) restarts.

Kubernetes Volumes:

1. emptyDir
2. hostPath: a pre-existing file or directory on the host machine
