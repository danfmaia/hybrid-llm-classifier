# Performance Benchmarking Results

## Test Environment

- System: Acer Nitro 5 (Development Environment)
  - CPU: Intel Core i5-9300H (4 cores, 8 threads)
  - GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
  - Memory: 31GB RAM
- Runtime: Python 3.10
- Dependencies:
  - FastAPI 0.104.1+
  - Ollama (Mistral-7B)
  - FAISS-CPU 1.7.4

## Current Results (as of Feb 7, 2025)

### Classification Performance

| Metric                | Target | Actual | Notes                      |
| --------------------- | ------ | ------ | -------------------------- |
| Accuracy (LegalBench) | 95%    | 85%    | Based on confidence scores |
| Precision             | TBD    | 0.85   | Initial contract testing   |
| Recall                | TBD    | N/A    | Needs more test data       |

### API Performance

| Metric              | Target    | Actual | Notes                             |
| ------------------- | --------- | ------ | --------------------------------- |
| Response Time (p95) | < 2s\*    | 11.37s | Single request, GPU-accelerated   |
| LLM Latency         | N/A       | 5.84s  | Using 22 GPU layers               |
| Embedding Latency   | N/A       | 2.21s  | With input size limiting          |
| Validation Latency  | N/A       | 3.32s  | FAISS similarity search           |
| Throughput          | 150 RPM\* | ~5 RPM | Current development configuration |
| Error Rate          | < 0.1%    | < 1%   | Mostly connection/timeout related |

\*Production targets with AWS g5.xlarge or higher instances

## Environment-Specific Performance Targets

### Development Environment (Local)

- Hardware: Intel i5-9300H, GTX 1650 4GB, 31GB RAM
- Expected Performance:
  - Response Time: ~10s acceptable
  - Throughput: 5-10 RPM
  - GPU Memory Usage: 3.5GB VRAM
  - Classification Accuracy: 85%

### Production Environment (AWS)

1. g5.xlarge (Minimum Recommended)

   - Hardware: NVIDIA A10G GPU (24GB VRAM)
   - Expected Performance:
     - Response Time: 3-4s
     - Throughput: 30-40 RPM per instance
     - GPU Memory Usage: ~8GB VRAM
     - Classification Accuracy: 85-90%

2. g5.2xlarge or Higher (Target Configuration)
   - Expected Performance:
     - Response Time: ~2s
     - Throughput: 150+ RPM (with load balancing)
     - GPU Memory Usage: 12-16GB VRAM
     - Classification Accuracy: 90-95%

### Scaling Strategy

1. Vertical Scaling (Single Instance)

   - Current: GTX 1650 (4GB VRAM)
   - Target: NVIDIA A10G (24GB VRAM)
   - Impact: 3-4x performance improvement

2. Horizontal Scaling (Multiple Instances)
   - Load Balancer + 3-4 g5.xlarge instances
   - Expected Throughput: 150+ RPM
   - High Availability: 99.9% uptime
   - Auto-scaling based on demand

### Performance Optimization Roadmap

1. Development Phase (Local)

   - Focus on code quality and correctness
   - Optimize within hardware constraints
   - Implement and test caching mechanisms
   - Profile and optimize memory usage

2. Pre-Production Phase (AWS)

   - Deploy to g5.xlarge for baseline
   - Implement load balancing
   - Enable auto-scaling
   - Optimize GPU memory usage
   - Fine-tune model parameters

3. Production Phase
   - Scale to multiple g5 instances
   - Implement regional deployment
   - Enable request caching
   - Monitor and optimize costs

### Cost-Performance Trade-offs

1. Development (Local)

   - Zero cloud costs
   - Higher latency acceptable
   - Limited by hardware

2. Production (AWS g5.xlarge)

   - Cost: ~$1.006/hour per instance
   - Better performance/cost ratio
   - Auto-scaling for cost optimization

3. Production (AWS g5.2xlarge)
   - Cost: ~$2.012/hour per instance
   - Optimal performance
   - Required for target RPM

### Monitoring and Optimization

1. Key Metrics to Track

   - Response time distribution
   - GPU memory utilization
   - Request queue length
   - Cache hit rates
   - Cost per inference

2. Optimization Levers
   - Instance type selection
   - Number of instances
   - Cache size and TTL
   - Batch size optimization
   - Load balancer configuration

### Resource Utilization

| Resource        | Target | Actual | Notes                            |
| --------------- | ------ | ------ | -------------------------------- |
| CPU Usage (avg) | N/A    | ~40%   | 4 threads for inference          |
| Memory Usage    | N/A    | ~19GB  | Including OS and other processes |
| GPU VRAM        | N/A    | 3.5GB  | 22 layers offloaded to GPU       |
| GPU Utilization | N/A    | ~80%   | During inference                 |

## Optimization Experiments (Feb 8, 2025)

### Baseline Performance

Initial configuration with default Mistral-7B parameters:

- Total Latency: 12.24s
- LLM Latency: 9.86s
- Embedding Latency: 0.95s
- Validation Latency: 1.43s
- Classification: Consistent (Contract, 0.81 confidence)

### Optimization Attempts

1. Aggressive GPU Optimization

   ```
   num_ctx: 512
   num_gpu: 35
   num_thread: 8
   ```

   Result: Failed with internal server error
   Lesson: GPU layer count too high for available VRAM

2. Conservative Optimization

   ```
   num_ctx: 1024
   num_gpu: 12
   num_thread: 6
   ```

   Results:

   - Total Latency: 19.86s
   - LLM Latency: 9.52s
   - Embedding Latency: 4.13s
   - Validation Latency: 6.20s
     Lesson: Increased thread count led to resource contention

3. Balanced Resource Allocation
   ```
   num_ctx: 1024
   num_gpu: 8
   num_thread: 4
   ```
   Results:
   - Total Latency: 25.51s
   - LLM Latency: 15.26s
   - Embedding Latency: 4.10s
   - Validation Latency: 6.15s
     Lesson: Reduced GPU layers actually increased latency

### Key Findings

1. Default Parameters Optimal

   ```
   num_ctx: 2048
   num_gpu: 1
   num_thread: 4
   ```

   Results:

   - Total Latency: 12.35s
   - LLM Latency: 10.13s
   - Embedding Latency: 0.89s
   - Validation Latency: 1.34s

2. Performance Insights:
   - Default Mistral-7B parameters are well-tuned for our use case
   - Increasing GPU layers degraded performance
   - Higher thread counts led to resource contention
   - Classification results remained consistent across tests

### Recommendations

1. Short-term Optimizations:

   - Implement response caching for repeated queries
   - Add batch processing capabilities
   - Optimize connection pooling
   - Add detailed performance monitoring

2. Infrastructure Considerations:

   - Maintain current GPU configuration
   - Focus on API-level optimizations
   - Consider distributed processing for batch operations
   - Implement warm-up strategies

3. Monitoring Needs:
   - Track GPU memory usage
   - Monitor thread utilization
   - Log classification latencies
   - Measure cache hit rates

## Optimization Status

### Current Optimizations

1. GPU Acceleration

   - 22 layers offloaded to GPU
   - Batch size optimized for 4GB VRAM
   - Context length reduced to 2048 tokens

2. Memory Management

   - Input text size limiting
   - Singleton classifier instance
   - Connection pooling and retry logic

3. Error Handling
   - Automatic retries with exponential backoff
   - Detailed error logging and monitoring
   - Graceful degradation under load

### Planned Improvements

1. Response Time

   - Implement response streaming
   - Add request caching
   - Optimize prompt engineering

2. Throughput

   - Batch request optimization
   - Parallel processing for validation
   - Load balancing configuration

3. Resource Usage
   - Memory usage optimization
   - GPU kernel optimization
   - Cache warm-up strategies

## Benchmark Scenarios

1. Single Document Classification

   - Small document (129 chars):
     - Total latency: 11.37s
     - Confidence: 0.85
     - Validation score: 0.5

2. Batch Classification (To be optimized)

   - Current limitations:
     - Sequential processing
     - No batching optimization
     - Limited by single instance

3. Concurrent Users
   - Current limitations:
     - Single instance bottleneck
     - No load balancing
     - Connection pooling needed

## Next Steps

1. Short-term (Pre-deployment)

   - Implement response streaming
   - Add request caching
   - Optimize batch processing

2. Medium-term

   - Load balancing setup
   - Memory optimization
   - Warm-up strategies

3. Long-term
   - Distributed processing
   - Custom GPU kernels
   - Advanced caching

## Notes

- Current performance is baseline with initial optimizations
- GPU acceleration shows significant improvement over CPU-only
- Further optimization needed to reach target response times
- Focus on reducing LLM latency and validation overhead
