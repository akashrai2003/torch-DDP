All benchmarks completed!
Results are saved in the benchmark_results directory
Summary of results:

ResNet50 Performance Comparison:
Configuration        Total Time   Epoch Time   Best Acc   Final Loss
-----------------------------------------------------------------
results/benchmark_resnet 3283.83      1641.91      60.60      2.1058    
results/benchmark_resnet 1093.23      1093.23      36.35      3.9523    
results/benchmark_resnet 329.41       164.59       15.83      4.1461    
results/benchmark_resnet 450.76       450.43       5.11       4.5780    
results/benchmark_resnet 616.10       308.05       5.26       4.4747  








All benchmarks completed!
Results are saved in the benchmark_results directory
Summary of results:

BERT Fine-tuning Performance Comparison:
Configuration        Total Time   Epoch Time   Best Acc   Final Loss
-----------------------------------------------------------------
results/benchmark_nlp 323.27       161.63       88.42      0.2370    
results/benchmark_nlp 185.40       92.70        88.53      0.2810    
results/benchmark_nlp 570.06       285.03       88.53      0.2811    
results/benchmark_nlp 134.10       67.05        87.63      0.3193    
results/benchmark_nlp 302.09       151.04       84.40      0.4335    
results/benchmark_nlp 211.23       105.52       90.41      0.5403    



Explaination of variation:
Based on your benchmark results, I can see some significant performance variations. Here are the key factors explaining the training speed differences:

Main Performance Factors:
1. Communication Overhead
Single node, multiple processes: Fast communication via shared memory
Multi-node setups: Network communication between nodes adds latency
More processes across nodes: Increased synchronization overhead during gradient aggregation
2. Resource Contention
When running 2 processes on a single node:

Both processes compete for the same CPU cores, memory bandwidth, and I/O
Context switching between processes adds overhead
Memory bus saturation when both processes access data simultaneously
3. Batch Size Per Process
Your effective batch size changes with the number of processes:

More processes = smaller batch per process = more gradient updates
Smaller batches can lead to less efficient GPU/CPU utilization
May require more communication rounds for the same amount of data
4. Network Bottlenecks
Multi-node configurations suffer from:

SSH overhead for remote process startup
Network latency between vm-master, vm-worker1, and vm-worker2
Bandwidth limitations during all-reduce operations
Looking at Your Results Pattern:
The slower runs (570.06s, 302.09s) likely correspond to:

Multi-node configurations with communication overhead
Configurations where processes compete for resources
The faster runs (134.10s, 185.40s) are probably:

Single node, single process (no contention)
Optimal resource utilization scenarios
Recommendations:
### Add this to your benchmark script to capture configuration details
echo "Configuration: $nodes nodes, $procs processes per node" >> benchmark_results/config_log.txt
echo "Expected total processes: $((nodes * procs))" >> benchmark_results/config_log.txt

The key takeaway is that distributed training doesn't always mean faster training - it depends on the balance between parallelization benefits and communication/contention costs.





# RAY based DDP:

(pytorch_dist) azureuser@vm-master:~/pytorch-distributed$ python train_llm_ray.py 
2025-06-06 16:55:55,792 INFO worker.py:1694 -- Connecting to existing Ray cluster at address: 10.2.0.4:6379...
2025-06-06 16:55:55,801 INFO worker.py:1879 -- Connected to Ray cluster. View the dashboard at 127.0.0.1:8265 

View detailed results here: /home/azureuser/ray_results/TorchTrainer_2025-06-06_16-55-55
To visualize your results with TensorBoard, run: `tensorboard --logdir /tmp/ray/session_2025-06-06_16-40-09_968565_25572/artifacts/2025-06-06_16-55-55/TorchTrainer_2025-06-06_16-55-55/driver_artifacts`

Training started without custom configuration.
(RayTrainWorker pid=23062, ip=10.2.0.5) Setting up process group for: env:// [rank=0, world_size=2]
(TorchTrainer pid=23004, ip=10.2.0.5) Started distributed worker processes: 
(TorchTrainer pid=23004, ip=10.2.0.5) - (node_id=470dfc784dc5529299e242759ef0792e7c647794c4134136000ba1fe, ip=10.2.0.5, pid=23062) world_rank=0, local_rank=0, node_rank=0
(TorchTrainer pid=23004, ip=10.2.0.5) - (node_id=d93746e82d1a008d55440cd6e1fc6d97b3c43de48fb3d7c28fafacad, ip=10.2.0.6, pid=24444) world_rank=1, local_rank=0, node_rank=1
Map:   0%|          | 0/4358 [00:00<?, ? examples/s]
Map:  23%|██▎       | 1000/4358 [00:00<00:01, 2146.65 examples/s]
Map:  46%|████▌     | 2000/4358 [00:00<00:00, 2417.23 examples/s]
Generating test split:   0%|          | 0/4358 [00:00<?, ? examples/s]
Map:  69%|██████▉   | 3000/4358 [00:01<00:00, 2396.25 examples/s]
Generating test split: 100%|██████████| 4358/4358 [00:00<00:00, 146577.31 examples/s]
Generating train split: 100%|██████████| 36718/36718 [00:00<00:00, 1538111.14 examples/s]
Generating validation split: 100%|██████████| 3760/3760 [00:00<00:00, 1144953.03 examples/s]
Map:  92%|█████████▏| 4000/4358 [00:01<00:00, 2621.97 examples/s]
Map: 100%|██████████| 4358/4358 [00:01<00:00, 2551.64 examples/s]
Map:   0%|          | 0/36718 [00:00<?, ? examples/s] [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
Map:   8%|▊         | 3000/36718 [00:01<00:13, 2506.19 examples/s] [repeated 14x across cluster]
Map:  92%|█████████▏| 4000/4358 [00:01<00:00, 2651.43 examples/s]
Map: 100%|██████████| 4358/4358 [00:01<00:00, 2567.81 examples/s]
Map:  49%|████▉     | 18000/36718 [00:06<00:06, 3045.80 examples/s] [repeated 31x across cluster]
Map:  93%|█████████▎| 34000/36718 [00:11<00:00, 2970.81 examples/s]
Map:  95%|█████████▌| 35000/36718 [00:11<00:00, 3058.61 examples/s]
Map:  98%|█████████▊| 36000/36718 [00:12<00:00, 3000.91 examples/s]
Map: 100%|██████████| 36718/36718 [00:12<00:00, 3024.45 examples/s]
Map: 100%|██████████| 36718/36718 [00:12<00:00, 2980.33 examples/s]
Map:   0%|          | 0/3760 [00:00<?, ? examples/s]
Map:  27%|██▋       | 1000/3760 [00:00<00:01, 2530.67 examples/s] [repeated 25x across cluster]
Map: 100%|██████████| 3760/3760 [00:01<00:00, 2502.72 examples/s]
(RayTrainWorker pid=23062, ip=10.2.0.5) [2025-06-06 16:56:34,263] [WARNING] [real_accelerator.py:209:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
(RayTrainWorker pid=23062, ip=10.2.0.5) [2025-06-06 16:56:34,264] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cpu (auto detect)
Map: 100%|██████████| 36718/36718 [00:12<00:00, 2966.72 examples/s] [repeated 5x across cluster]
  0%|          | 0/125 [00:00<?, ?it/s]/home/azureuser/pytorch_dist/lib/python3.10/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
(RayTrainWorker pid=23062, ip=10.2.0.5)   warnings.warn(warn_msg)
Map:   0%|          | 0/3760 [00:00<?, ? examples/s]
Map:  80%|███████▉  | 3000/3760 [00:01<00:00, 2474.77 examples/s] [repeated 5x across cluster]
Map: 100%|██████████| 3760/3760 [00:01<00:00, 2511.57 examples/s]
(RayTrainWorker pid=24444, ip=10.2.0.6) /home/azureuser/pytorch_dist/lib/python3.10/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
(RayTrainWorker pid=24444, ip=10.2.0.6) `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
(RayTrainWorker pid=23062, ip=10.2.0.5) `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
  1%|          | 1/125 [00:01<03:07,  1.52s/it]
  2%|▏         | 2/125 [00:02<02:51,  1.39s/it]
  2%|▏         | 3/125 [00:04<02:43,  1.34s/it]
  3%|▎         | 4/125 [00:05<02:41,  1.34s/it]
(RayTrainWorker pid=24444, ip=10.2.0.6)   warnings.warn(warn_msg)
(RayTrainWorker pid=23062, ip=10.2.0.5) `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
(RayTrainWorker pid=24444, ip=10.2.0.6) `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
  4%|▍         | 5/125 [00:06<02:38,  1.32s/it]
  5%|▍         | 6/125 [00:07<02:34,  1.30s/it]
  6%|▌         | 7/125 [00:09<02:32,  1.29s/it]
  6%|▋         | 8/125 [00:10<02:29,  1.28s/it]
  7%|▋         | 9/125 [00:11<02:28,  1.28s/it]
  8%|▊         | 10/125 [00:13<02:26,  1.27s/it]
  9%|▉         | 11/125 [00:14<02:24,  1.27s/it]
 10%|▉         | 12/125 [00:15<02:23,  1.27s/it]
 10%|█         | 13/125 [00:16<02:21,  1.27s/it]
 11%|█         | 14/125 [00:18<02:21,  1.27s/it]
 12%|█▏        | 15/125 [00:19<02:19,  1.27s/it]
 13%|█▎        | 16/125 [00:20<02:18,  1.27s/it]
 14%|█▎        | 17/125 [00:21<02:16,  1.27s/it]
 14%|█▍        | 18/125 [00:23<02:15,  1.26s/it]
 15%|█▌        | 19/125 [00:24<02:13,  1.26s/it]
 16%|█▌        | 20/125 [00:25<02:12,  1.26s/it]
 17%|█▋        | 21/125 [00:26<02:11,  1.26s/it]
 18%|█▊        | 22/125 [00:28<02:10,  1.27s/it]
 18%|█▊        | 23/125 [00:29<02:08,  1.26s/it]
 19%|█▉        | 24/125 [00:30<02:07,  1.26s/it]
 20%|██        | 25/125 [00:32<02:05,  1.26s/it]
 21%|██        | 26/125 [00:33<02:04,  1.26s/it]
 22%|██▏       | 27/125 [00:34<02:03,  1.26s/it]
 22%|██▏       | 28/125 [00:35<02:02,  1.26s/it]
 23%|██▎       | 29/125 [00:37<02:00,  1.26s/it]
 24%|██▍       | 30/125 [00:38<01:59,  1.25s/it]
 25%|██▍       | 31/125 [00:39<01:58,  1.26s/it]
 26%|██▌       | 32/125 [00:40<01:56,  1.26s/it]
 26%|██▋       | 33/125 [00:42<01:55,  1.26s/it]
 27%|██▋       | 34/125 [00:43<01:54,  1.26s/it]
 28%|██▊       | 35/125 [00:44<01:53,  1.26s/it]
 29%|██▉       | 36/125 [00:45<01:51,  1.26s/it]
 30%|██▉       | 37/125 [00:47<01:50,  1.26s/it]
 30%|███       | 38/125 [00:48<01:48,  1.25s/it]
 31%|███       | 39/125 [00:49<01:47,  1.25s/it]
 32%|███▏      | 40/125 [00:50<01:46,  1.25s/it]
 33%|███▎      | 41/125 [00:52<01:44,  1.25s/it]
 34%|███▎      | 42/125 [00:53<01:43,  1.25s/it]
 34%|███▍      | 43/125 [00:54<01:42,  1.25s/it]
 35%|███▌      | 44/125 [00:55<01:41,  1.25s/it]
 36%|███▌      | 45/125 [00:57<01:40,  1.25s/it]
 37%|███▋      | 46/125 [00:58<01:39,  1.25s/it]
 38%|███▊      | 47/125 [00:59<01:37,  1.25s/it]
 38%|███▊      | 48/125 [01:00<01:36,  1.25s/it]
 39%|███▉      | 49/125 [01:02<01:35,  1.25s/it]
 40%|████      | 50/125 [01:03<01:34,  1.25s/it]
 41%|████      | 51/125 [01:04<01:32,  1.25s/it]
 42%|████▏     | 52/125 [01:05<01:31,  1.25s/it]
 42%|████▏     | 53/125 [01:07<01:29,  1.25s/it]
 43%|████▎     | 54/125 [01:08<01:28,  1.25s/it]
 44%|████▍     | 55/125 [01:09<01:27,  1.25s/it]
 45%|████▍     | 56/125 [01:10<01:26,  1.25s/it]
 46%|████▌     | 57/125 [01:12<01:24,  1.25s/it]
 46%|████▋     | 58/125 [01:13<01:24,  1.25s/it]
 47%|████▋     | 59/125 [01:14<01:22,  1.25s/it]
 48%|████▊     | 60/125 [01:15<01:21,  1.25s/it]
 49%|████▉     | 61/125 [01:17<01:19,  1.25s/it]
 50%|████▉     | 62/125 [01:18<01:18,  1.25s/it]
 50%|█████     | 63/125 [01:19<01:17,  1.25s/it]
 51%|█████     | 64/125 [01:20<01:16,  1.25s/it]
 52%|█████▏    | 65/125 [01:22<01:14,  1.25s/it]
 53%|█████▎    | 66/125 [01:23<01:13,  1.25s/it]
 54%|█████▎    | 67/125 [01:24<01:12,  1.25s/it]
 54%|█████▍    | 68/125 [01:25<01:11,  1.25s/it]
 55%|█████▌    | 69/125 [01:27<01:10,  1.25s/it]
 56%|█████▌    | 70/125 [01:28<01:08,  1.25s/it]
 57%|█████▋    | 71/125 [01:29<01:07,  1.25s/it]
 58%|█████▊    | 72/125 [01:30<01:06,  1.25s/it]
 58%|█████▊    | 73/125 [01:32<01:05,  1.25s/it]
 59%|█████▉    | 74/125 [01:33<01:03,  1.25s/it]
 60%|██████    | 75/125 [01:34<01:02,  1.25s/it]
 61%|██████    | 76/125 [01:35<01:01,  1.25s/it]
 62%|██████▏   | 77/125 [01:37<00:59,  1.25s/it]
 62%|██████▏   | 78/125 [01:38<00:58,  1.25s/it]
 63%|██████▎   | 79/125 [01:39<00:57,  1.24s/it]
 64%|██████▍   | 80/125 [01:40<00:56,  1.26s/it]
 65%|██████▍   | 81/125 [01:42<00:55,  1.26s/it]
 66%|██████▌   | 82/125 [01:43<00:54,  1.26s/it]
 66%|██████▋   | 83/125 [01:44<00:52,  1.26s/it]
 67%|██████▋   | 84/125 [01:45<00:51,  1.26s/it]
 68%|██████▊   | 85/125 [01:47<00:50,  1.26s/it]
 69%|██████▉   | 86/125 [01:48<00:49,  1.26s/it]
 70%|██████▉   | 87/125 [01:49<00:47,  1.26s/it]
 70%|███████   | 88/125 [01:50<00:46,  1.26s/it]
 71%|███████   | 89/125 [01:52<00:45,  1.25s/it]
 72%|███████▏  | 90/125 [01:53<00:43,  1.25s/it]
 73%|███████▎  | 91/125 [01:54<00:42,  1.25s/it]
 74%|███████▎  | 92/125 [01:55<00:41,  1.25s/it]
 74%|███████▍  | 93/125 [01:57<00:40,  1.26s/it]
 75%|███████▌  | 94/125 [01:58<00:38,  1.26s/it]
 76%|███████▌  | 95/125 [01:59<00:37,  1.25s/it]
 77%|███████▋  | 96/125 [02:00<00:36,  1.25s/it]
 78%|███████▊  | 97/125 [02:02<00:35,  1.25s/it]
 78%|███████▊  | 98/125 [02:03<00:33,  1.25s/it]
 79%|███████▉  | 99/125 [02:04<00:32,  1.25s/it]
 80%|████████  | 100/125 [02:05<00:31,  1.25s/it]
 81%|████████  | 101/125 [02:07<00:30,  1.25s/it]
 82%|████████▏ | 102/125 [02:08<00:28,  1.25s/it]
 82%|████████▏ | 103/125 [02:09<00:27,  1.25s/it]
 83%|████████▎ | 104/125 [02:10<00:26,  1.25s/it]
 84%|████████▍ | 105/125 [02:12<00:25,  1.26s/it]
 85%|████████▍ | 106/125 [02:13<00:23,  1.26s/it]
 86%|████████▌ | 107/125 [02:14<00:22,  1.26s/it]
 86%|████████▋ | 108/125 [02:16<00:21,  1.26s/it]
 87%|████████▋ | 109/125 [02:17<00:20,  1.26s/it]
 88%|████████▊ | 110/125 [02:18<00:18,  1.26s/it]
 89%|████████▉ | 111/125 [02:19<00:17,  1.26s/it]
 90%|████████▉ | 112/125 [02:21<00:16,  1.26s/it]
 90%|█████████ | 113/125 [02:22<00:15,  1.26s/it]
 91%|█████████ | 114/125 [02:23<00:13,  1.25s/it]
 92%|█████████▏| 115/125 [02:24<00:12,  1.25s/it]
 93%|█████████▎| 116/125 [02:26<00:11,  1.25s/it]
 94%|█████████▎| 117/125 [02:27<00:10,  1.25s/it]
 94%|█████████▍| 118/125 [02:28<00:08,  1.26s/it]
 95%|█████████▌| 119/125 [02:29<00:07,  1.27s/it]
 96%|█████████▌| 120/125 [02:31<00:06,  1.26s/it]
 97%|█████████▋| 121/125 [02:32<00:05,  1.26s/it]
 98%|█████████▊| 122/125 [02:33<00:03,  1.26s/it]
 98%|█████████▊| 123/125 [02:34<00:02,  1.26s/it]
 99%|█████████▉| 124/125 [02:36<00:01,  1.26s/it]
100%|██████████| 125/125 [02:37<00:00,  1.26s/it]
100%|██████████| 125/125 [02:38<00:00,  1.27s/it]
(RayTrainWorker pid=23062, ip=10.2.0.5) {'train_runtime': 158.136, 'train_samples_per_second': 6.324, 'train_steps_per_second': 0.79, 'train_loss': 1.606440673828125, 'epoch': 1.0}
(RayTrainWorker pid=24444, ip=10.2.0.6) [2025-06-06 16:56:36,632] [WARNING] [real_accelerator.py:209:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
(RayTrainWorker pid=24444, ip=10.2.0.6) [2025-06-06 16:56:36,791] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cpu (auto detect)

Training completed after 0 iterations at 2025-06-06 16:59:18. Total running time: 3min 22s
2025-06-06 16:59:18,533 INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/home/azureuser/ray_results/TorchTrainer_2025-06-06_16-55-55' in 0.0016s.

(pytorch_dist) azureuser@vm-master:~/pytorch-distributed$ python train_llm_ray.py 
2025-06-06 17:03:59,187 INFO worker.py:1694 -- Connecting to existing Ray cluster at address: 10.2.0.4:6379...
2025-06-06 17:03:59,195 INFO worker.py:1879 -- Connected to Ray cluster. View the dashboard at 127.0.0.1:8265 

View detailed results here: /home/azureuser/ray_results/TorchTrainer_2025-06-06_17-03-59
To visualize your results with TensorBoard, run: `tensorboard --logdir /tmp/ray/session_2025-06-06_16-40-09_968565_25572/artifacts/2025-06-06_17-03-59/TorchTrainer_2025-06-06_17-03-59/driver_artifacts`

Training started without custom configuration.
(RayTrainWorker pid=23385, ip=10.2.0.5) Setting up process group for: env:// [rank=0, world_size=2]
(TorchTrainer pid=23327, ip=10.2.0.5) Started distributed worker processes: 
(TorchTrainer pid=23327, ip=10.2.0.5) - (node_id=470dfc784dc5529299e242759ef0792e7c647794c4134136000ba1fe, ip=10.2.0.5, pid=23385) world_rank=0, local_rank=0, node_rank=0
(TorchTrainer pid=23327, ip=10.2.0.5) - (node_id=d93746e82d1a008d55440cd6e1fc6d97b3c43de48fb3d7c28fafacad, ip=10.2.0.6, pid=24656) world_rank=1, local_rank=0, node_rank=1
Map:   0%|          | 0/4358 [00:00<?, ? examples/s]
Map:  23%|██▎       | 1000/4358 [00:00<00:01, 2159.14 examples/s]
Map:  46%|████▌     | 2000/4358 [00:00<00:00, 2444.93 examples/s]
Map:  69%|██████▉   | 3000/4358 [00:01<00:00, 2480.85 examples/s]
Map:  92%|█████████▏| 4000/4358 [00:01<00:00, 2699.53 examples/s]
Map: 100%|██████████| 4358/4358 [00:01<00:00, 2775.68 examples/s]
Map: 100%|██████████| 4358/4358 [00:01<00:00, 2540.51 examples/s]
Map:   0%|          | 0/36718 [00:00<?, ? examples/s]
Map:   3%|▎         | 1000/36718 [00:00<00:15, 2298.86 examples/s]
Map:   5%|▌         | 2000/36718 [00:00<00:15, 2303.15 examples/s]
Map:   8%|▊         | 3000/36718 [00:01<00:13, 2542.95 examples/s]
Map:  11%|█         | 4000/36718 [00:01<00:12, 2637.42 examples/s]
Map:  14%|█▎        | 5000/36718 [00:01<00:11, 2818.58 examples/s]
Map:  16%|█▋        | 6000/36718 [00:02<00:10, 2948.03 examples/s]
Map:  19%|█▉        | 7000/36718 [00:02<00:10, 2794.28 examples/s]
Map:  22%|██▏       | 8000/36718 [00:02<00:10, 2814.14 examples/s]
Map:   0%|          | 0/4358 [00:00<?, ? examples/s]
Map:  25%|██▍       | 9000/36718 [00:03<00:09, 2887.33 examples/s]
Map:  92%|█████████▏| 4000/4358 [00:01<00:00, 2650.43 examples/s]
Map: 100%|██████████| 4358/4358 [00:01<00:00, 2513.71 examples/s]
Map:   0%|          | 0/36718 [00:00<?, ? examples/s]
Map:  68%|██████▊   | 25000/36718 [00:08<00:03, 3356.98 examples/s] [repeated 26x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
Map:  93%|█████████▎| 34000/36718 [00:11<00:00, 3039.89 examples/s]
Map:  95%|█████████▌| 35000/36718 [00:11<00:00, 3128.17 examples/s]
Map:  98%|█████████▊| 36000/36718 [00:11<00:00, 3072.53 examples/s]
Map: 100%|██████████| 36718/36718 [00:12<00:00, 3098.04 examples/s]
Map: 100%|██████████| 36718/36718 [00:12<00:00, 2850.65 examples/s]
Map:  63%|██████▎   | 23000/36718 [00:07<00:04, 3104.36 examples/s] [repeated 24x across cluster]
Map:   0%|          | 0/3760 [00:00<?, ? examples/s]
Map: 100%|██████████| 3760/3760 [00:01<00:00, 2589.84 examples/s]
Map: 100%|██████████| 3760/3760 [00:01<00:00, 2372.77 examples/s]
(RayTrainWorker pid=23385, ip=10.2.0.5) [2025-06-06 17:04:29,857] [WARNING] [real_accelerator.py:209:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
(RayTrainWorker pid=23385, ip=10.2.0.5) [2025-06-06 17:04:29,858] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cpu (auto detect)
Map:  93%|█████████▎| 34000/36718 [00:11<00:00, 2969.06 examples/s]
Map:  95%|█████████▌| 35000/36718 [00:11<00:00, 3057.86 examples/s]
Map:  98%|█████████▊| 36000/36718 [00:12<00:00, 2999.73 examples/s]
Map: 100%|██████████| 36718/36718 [00:12<00:00, 2976.92 examples/s]
Map:  90%|████████▉ | 33000/36718 [00:11<00:01, 2981.19 examples/s] [repeated 13x across cluster]
Map:   0%|          | 0/3760 [00:00<?, ? examples/s]
Map: 100%|██████████| 3760/3760 [00:01<00:00, 2505.42 examples/s]
  0%|          | 0/125 [00:00<?, ?it/s]/home/azureuser/pytorch_dist/lib/python3.10/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
(RayTrainWorker pid=23385, ip=10.2.0.5)   warnings.warn(warn_msg)
(RayTrainWorker pid=23385, ip=10.2.0.5) `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
(RayTrainWorker pid=24656, ip=10.2.0.6) /home/azureuser/pytorch_dist/lib/python3.10/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
(RayTrainWorker pid=23385, ip=10.2.0.5) `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
Map:  80%|███████▉  | 3000/3760 [00:01<00:00, 2471.28 examples/s] [repeated 3x across cluster]
  1%|          | 1/125 [00:12<26:33, 12.85s/it]
(RayTrainWorker pid=24656, ip=10.2.0.6)   warnings.warn(warn_msg)
(RayTrainWorker pid=24656, ip=10.2.0.6) `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
(RayTrainWorker pid=24656, ip=10.2.0.6) `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
  2%|▏         | 2/125 [00:24<24:57, 12.18s/it]
  2%|▏         | 3/125 [00:36<24:22, 11.99s/it]
  3%|▎         | 4/125 [00:48<23:56, 11.87s/it]
  4%|▍         | 5/125 [00:59<23:42, 11.85s/it]
  5%|▍         | 6/125 [01:11<23:25, 11.81s/it]
  6%|▌         | 7/125 [01:23<23:10, 11.78s/it]
  6%|▋         | 8/125 [01:35<22:57, 11.78s/it]
  7%|▋         | 9/125 [01:46<22:43, 11.75s/it]
  8%|▊         | 10/125 [01:58<22:29, 11.73s/it]
  9%|▉         | 11/125 [02:10<22:16, 11.72s/it]
 10%|▉         | 12/125 [02:21<22:03, 11.71s/it]
 10%|█         | 13/125 [02:33<21:53, 11.73s/it]
 11%|█         | 14/125 [02:45<21:40, 11.72s/it]
 12%|█▏        | 15/125 [02:57<21:31, 11.74s/it]
 13%|█▎        | 16/125 [03:08<21:18, 11.73s/it]
 14%|█▎        | 17/125 [03:20<21:06, 11.73s/it]
 14%|█▍        | 18/125 [03:32<20:55, 11.74s/it]
 15%|█▌        | 19/125 [03:43<20:40, 11.70s/it]
 16%|█▌        | 20/125 [03:55<20:31, 11.73s/it]
 17%|█▋        | 21/125 [04:07<20:19, 11.73s/it]
 18%|█▊        | 22/125 [04:19<20:08, 11.74s/it]
 18%|█▊        | 23/125 [04:30<19:56, 11.73s/it]
 19%|█▉        | 24/125 [04:42<19:44, 11.73s/it]
 20%|██        | 25/125 [04:54<19:34, 11.75s/it]
 21%|██        | 26/125 [05:06<19:22, 11.74s/it]
 22%|██▏       | 27/125 [05:17<19:10, 11.74s/it]
 22%|██▏       | 28/125 [05:29<18:58, 11.74s/it]
 23%|██▎       | 29/125 [05:41<18:46, 11.74s/it]
 24%|██▍       | 30/125 [05:53<18:35, 11.75s/it]
 25%|██▍       | 31/125 [06:04<18:20, 11.71s/it]
 26%|██▌       | 32/125 [06:16<18:09, 11.72s/it]
 26%|██▋       | 33/125 [06:28<17:58, 11.72s/it]
 27%|██▋       | 34/125 [06:40<17:49, 11.75s/it]
 28%|██▊       | 35/125 [06:51<17:35, 11.73s/it]
 29%|██▉       | 36/125 [07:03<17:24, 11.74s/it]
 30%|██▉       | 37/125 [07:15<17:13, 11.74s/it]
 30%|███       | 38/125 [07:26<17:02, 11.75s/it]
 31%|███       | 39/125 [07:38<16:48, 11.72s/it]
 32%|███▏      | 40/125 [07:50<16:37, 11.74s/it]
 33%|███▎      | 41/125 [08:02<16:27, 11.76s/it]
 34%|███▎      | 42/125 [08:13<16:15, 11.76s/it]
 34%|███▍      | 43/125 [08:25<16:03, 11.75s/it]
 35%|███▌      | 44/125 [08:37<15:50, 11.73s/it]
 36%|███▌      | 45/125 [08:49<15:37, 11.72s/it]
 37%|███▋      | 46/125 [09:00<15:25, 11.71s/it]
 38%|███▊      | 47/125 [09:12<15:14, 11.73s/it]
 38%|███▊      | 48/125 [09:24<15:00, 11.70s/it]
 39%|███▉      | 49/125 [09:35<14:49, 11.70s/it]
 40%|████      | 50/125 [09:47<14:37, 11.70s/it]
 41%|████      | 51/125 [09:59<14:27, 11.73s/it]
 42%|████▏     | 52/125 [10:11<14:15, 11.72s/it]
 42%|████▏     | 53/125 [10:22<14:03, 11.72s/it]
 43%|████▎     | 54/125 [10:34<13:51, 11.72s/it]
 44%|████▍     | 55/125 [10:46<13:41, 11.73s/it]
 45%|████▍     | 56/125 [10:58<13:30, 11.75s/it]
 46%|████▌     | 57/125 [11:09<13:17, 11.73s/it]
 46%|████▋     | 58/125 [11:21<13:04, 11.71s/it]
 47%|████▋     | 59/125 [11:33<12:53, 11.72s/it]
 48%|████▊     | 60/125 [11:44<12:40, 11.71s/it]
 49%|████▉     | 61/125 [11:56<12:29, 11.70s/it]
 50%|████▉     | 62/125 [12:08<12:15, 11.68s/it]
 50%|█████     | 63/125 [12:19<12:04, 11.68s/it]
 51%|█████     | 64/125 [12:31<11:54, 11.71s/it]
 52%|█████▏    | 65/125 [12:43<11:40, 11.68s/it]
 53%|█████▎    | 66/125 [12:54<11:29, 11.69s/it]
 54%|█████▎    | 67/125 [13:06<11:18, 11.69s/it]
 54%|█████▍    | 68/125 [13:18<11:07, 11.71s/it]
 55%|█████▌    | 69/125 [13:30<10:57, 11.73s/it]
 56%|█████▌    | 70/125 [13:41<10:44, 11.71s/it]
 57%|█████▋    | 71/125 [13:53<10:33, 11.74s/it]
 58%|█████▊    | 72/125 [14:05<10:20, 11.71s/it]
 58%|█████▊    | 73/125 [14:17<10:10, 11.74s/it]
 59%|█████▉    | 74/125 [14:28<09:58, 11.74s/it]
 60%|██████    | 75/125 [14:40<09:45, 11.71s/it]
 61%|██████    | 76/125 [14:52<09:34, 11.72s/it]
 62%|██████▏   | 77/125 [15:03<09:22, 11.72s/it]
 62%|██████▏   | 78/125 [15:15<09:11, 11.74s/it]
 63%|██████▎   | 79/125 [15:27<09:00, 11.74s/it]
 64%|██████▍   | 80/125 [15:39<08:48, 11.74s/it]
 65%|██████▍   | 81/125 [15:50<08:36, 11.74s/it]
 66%|██████▌   | 82/125 [16:02<08:24, 11.74s/it]
 66%|██████▋   | 83/125 [16:14<08:11, 11.71s/it]
 67%|██████▋   | 84/125 [16:26<08:00, 11.73s/it]
 68%|██████▊   | 85/125 [16:37<07:49, 11.74s/it]
 69%|██████▉   | 86/125 [16:49<07:37, 11.73s/it]
 70%|██████▉   | 87/125 [17:01<07:25, 11.74s/it]
 70%|███████   | 88/125 [17:13<07:14, 11.75s/it]
 71%|███████   | 89/125 [17:24<07:02, 11.73s/it]
 72%|███████▏  | 90/125 [17:36<06:49, 11.71s/it]
 73%|███████▎  | 91/125 [17:48<06:38, 11.72s/it]
 74%|███████▎  | 92/125 [17:59<06:27, 11.74s/it]
 74%|███████▍  | 93/125 [18:11<06:16, 11.76s/it]
 75%|███████▌  | 94/125 [18:23<06:04, 11.76s/it]
 76%|███████▌  | 95/125 [18:35<05:52, 11.76s/it]
 77%|███████▋  | 96/125 [18:47<05:40, 11.76s/it]
 78%|███████▊  | 97/125 [18:58<05:29, 11.76s/it]
 78%|███████▊  | 98/125 [19:10<05:18, 11.78s/it]
 79%|███████▉  | 99/125 [19:22<05:06, 11.77s/it]
 80%|████████  | 100/125 [19:34<04:54, 11.78s/it]
 81%|████████  | 101/125 [19:45<04:41, 11.74s/it]
 82%|████████▏ | 102/125 [19:57<04:30, 11.74s/it]
 82%|████████▏ | 103/125 [20:09<04:18, 11.75s/it]
 83%|████████▎ | 104/125 [20:21<04:06, 11.75s/it]
 84%|████████▍ | 105/125 [20:32<03:55, 11.75s/it]
 85%|████████▍ | 106/125 [20:44<03:42, 11.72s/it]
 86%|████████▌ | 107/125 [20:56<03:31, 11.73s/it]
 86%|████████▋ | 108/125 [21:08<03:20, 11.78s/it]
 87%|████████▋ | 109/125 [21:19<03:08, 11.75s/it]
 88%|████████▊ | 110/125 [21:31<02:56, 11.74s/it]
 89%|████████▉ | 111/125 [21:43<02:44, 11.73s/it]
 90%|████████▉ | 112/125 [21:54<02:32, 11.73s/it]
 90%|█████████ | 113/125 [22:06<02:20, 11.72s/it]
 91%|█████████ | 114/125 [22:18<02:08, 11.72s/it]
 92%|█████████▏| 115/125 [22:30<01:57, 11.70s/it]
 93%|█████████▎| 116/125 [22:41<01:45, 11.70s/it]
 94%|█████████▎| 117/125 [22:53<01:33, 11.72s/it]
 94%|█████████▍| 118/125 [23:05<01:21, 11.69s/it]
 95%|█████████▌| 119/125 [23:16<01:10, 11.72s/it]
 96%|█████████▌| 120/125 [23:28<00:58, 11.78s/it]
 97%|█████████▋| 121/125 [23:40<00:47, 11.75s/it]
 98%|█████████▊| 122/125 [23:52<00:35, 11.75s/it]
 98%|█████████▊| 123/125 [24:03<00:23, 11.73s/it]
 99%|█████████▉| 124/125 [24:15<00:11, 11.71s/it]
100%|██████████| 125/125 [24:27<00:00, 11.74s/it]
(RayTrainWorker pid=23385, ip=10.2.0.5) {'train_runtime': 1476.4048, 'train_samples_per_second': 0.677, 'train_steps_per_second': 0.085, 'train_loss': 1.1289842529296874, 'epoch': 1.0}
(RayTrainWorker pid=24656, ip=10.2.0.6) [2025-06-06 17:04:34,705] [WARNING] [real_accelerator.py:209:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
(RayTrainWorker pid=24656, ip=10.2.0.6) [2025-06-06 17:04:34,705] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cpu (auto detect)
100%|██████████| 125/125 [24:36<00:00, 11.81s/it]

Training completed after 0 iterations at 2025-06-06 17:29:15. Total running time: 25min 16s
2025-06-06 17:29:15,483 INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to '/home/azureuser/ray_results/TorchTrainer_2025-06-06_17-03-59' in 0.0015s.
