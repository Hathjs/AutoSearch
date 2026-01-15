
nvcc --version
index_file=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chenqing30/cqproject/searchr1-rag-traindata/PeterJinGo/wiki-18-e5-index/e5_Flat.index
corpus_file=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chenqing30/cqproject/searchr1-rag-traindata/PeterJinGo/wiki-18-corpus/wiki-18.jsonl
retriever_name=e5
retriever_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chenqing30/cqproject/hf-model/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
