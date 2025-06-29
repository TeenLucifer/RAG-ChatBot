# 基于检索增强的运维知识智能问答项目
## 项目背景介绍
公司内部存在海量的运维技术文档(例如摄像头、录像机、交换机、解码器等监控设备运维技术文档，5G领域运维技术文档等等)，旨在探索利用大模型检索增强技术(RAG)结合领域私有技术文档进行高效的私域知识问答，提升运维效率。

## 技术方案

### 数据预处理模块
运维技术文档多为pdf文档，数据体量大，类别差异大，且包含大量图表信息，我们采用图文分离的策略，文字部分转化为markdown格式，图片表格部分，我们采用多模态大模型(qwen2-vl)进行内容识别，构建技术语料库。

### 检索库搭建
基于语料库，构建索引库，主要两种类型：
- 使用ES构建基于bm25的字面检索库；
- 使用中文bge模型按照段落进行文档向量化，再使用milvus构建向量检索库。

### 问句解析模块
用户问句的口语化严重，且存在意图模糊的现象，而检索库文档内容具有极强的专业性，不适合直接使用原始问句进行内容检索，因此需要对原始问句进行查询扩展和意图识别。
1. 查询扩展：目的在于对原始问句进行语义转换和内容扩充，通过这种方法得到多个问句之后，再进行多路检索召回。这个任务使用qwen2.5-7B进行训练即可。
2. 意图识别：识别出用户的意图,即：用户的问题对应哪个类型的技术文档，识别出之后，可以精准定位，直接检索对应类型的技术文档内容。这个任务使用普通的bert模型即可

### 召回排序模块
用户问句进行问句解析模块之后，会得到多个拓展的问句和意图，接下来分别对每一个问句根据意图进行精准检索(此为多路召回)，之后再对所有召回内容进行汇总，得到召回候选项，进行排序，排序可以使用bge排序模型进行排序，过滤掉一部分无关的内容。

#### 答案生成模块
召回排序之后的得到的背景知识，结合原始问句，基于大模型(qwen2.5-14B)进行最终答案的回答。

### 业务效果
基于检索增强的运维知识智能问答项目上线之后，技术咨询问题解决率达到85%以上，实现了大模型技术在工业场景中的成功落地。

## 环境启动方式
### milvus + attu
* milvus: standalone.bat
* attu: docker run -d -p 8000:3000 -e MILVUS_URL=host.docker.internal:19530 zilliz/attu:v2.5

### elasticsearch

* es账号: elastic
* es密码: 1999jnhw&&wjt

* kibana启动命令: $ docker run -d --name kibana --net es-net -p 5601:5601 kibana:8.6.0

```bash
docker run -d --name kibana \
    -e ELASTICSERACH_HOSTS=http://es:9200 \
    --network=es-net \
    -p 5601:5601 \
    kibana:8.6.0
```

```bash
docker run -d --name es \
    -e "ES_JAVA_OPTS=-Xms1024m -Xmx1024m" \
    -e "discovery.type=single-node" \
    -v D:\Projects\elasticsearch\es-data:/usr/share/elasticsearch/data \
    -v D:\Projects\elasticsearch\es-plugins:/usr/share/elasticsearch/plugins \
    -v D:\Projects\elasticsearch\temp:/usr/temp \
    --privileged \
    --network es-net \
    -p 9200:9200 \
    -p 9300:9300 \
    elasticsearch:8.6.0
```

``` bash
-d                                      #容器后台运行
--name es                               #容器命名
-e "ES_JAVA_OPTS=-Xms1024m -Xmx1024m"   #设置容器最小最大运行内存
-e "discovery.type=single-node"         #设置es的运行模式是单机运行
-v C:\Users\14547\Desktop\lyq\elasticsearch\es-data:/usr/share/elasticsearch/data  #数据卷，方便容器持久化
-v D:\Projects\elasticsearch\es-plugins:/usr/share/elasticsearch/plugins #数据卷，方便容器持久化
-v D:\Projects\elasticsearch\temp:/usr/temp #数据卷，方便容器持久化
--privileged                            #以最大权限运行容器
--network es-net                        #执行容器运行网络，与kibana运行保持在同一网络
-p 9200:9200                            #开放端口
-p 9300:9300                            #开放端口
elasticsearch:8.6.0                     #运行镜像和tag
```
## llama-index闭坑
1. 不支持dashscope的multi-modal embedding, 需要自行改写multi-modal embedding的代码
2. 不支持dashscope的multi-modal model, llama-index官方提供的demo也跑不起来, 需要调用dashscope的sdk完成对话