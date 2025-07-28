from benchmarks import VQABenchmark, HallucinationMed, HallucinationGen, ImageClassification, BoundingBox

def run_all_benchmarks():
    benchmarks = [
        VQABenchmark(),
        HallucinationMed(),
        HallucinationGen(),
        ImageClassification(),
        BoundingBox()
    ]
    all_results = []
    for benchmark in benchmarks:
        result = benchmark.run()
        all_results.append(result)
        print(f"Completed benchmark: {benchmark.__class__.__name__}")
    return all_results
    
if __name__ == "__main__":
    results = run_all_benchmarks()

    import json
    print("\n\nFINAL BENCHMARK RESULTS:")
    print(json.dumps(results, indent=2))

