# CSL7110 — Machine Learning with Big Data
## Assignment 4: Clustering, Web Search & PageRank

**Name:** Shreyas Gaikwad
**Roll No:** M25DE1042
**Program:** M.Tech Data Engineering

---

## Repository Structure

```
mlbd_a4/
│
├── M25DE1042_CSL7110_Assignment4.ipynb   # Main notebook (all 3 parts)
│
├── data/
│   ├── spambase.data                     # UCI Spam dataset (Part 1)
│   ├── webpages/                         # Webpage text files (Part 2)
│   ├── actions.txt                       # Search queries (Part 2)
│   ├── answers.txt                       # Expected answers (Part 2)
│   ├── small.txt                         # Small graph 100 nodes (Part 3)
│   └── whole.txt                         # Full graph 1000 nodes (Part 3)
│
└── README.md
```
However, later was done by locally uploading files to Colab.
---

## Requirements

```
Python 3.x
PySpark 4.0.2
NumPy
Google Colab (recommended) or local Spark setup
```

Install dependencies:
```bash
pip install pyspark numpy
```

---

## How to Run

1. Open `M25DE1042_CSL7110_Assignment4.ipynb` in Google Colab
2. Upload the dataset files to `/content/sample_data/`
3. Run all cells top to bottom in order
4. Each part is clearly marked and self-contained

---

## Part 1 — Clustering (UCI Spambase Dataset)

### What it does
Implements two clustering center-selection algorithms on a 4601-point, 58-dimensional spam email dataset and compares their quality using the k-Means objective function.

### Algorithms
- **Farthest-First Traversal (k-center):** Greedy algorithm that repeatedly picks the point farthest from all existing centers. Guarantees a 2-approximation and runs in O(|P| × k).
- **k-Means++:** Probabilistic initialization that picks each new center with probability proportional to its squared distance from the nearest existing center. Runs in O(|P| × k).
- **kmeansObj:** Measures clustering quality as the average squared distance of every point to its nearest center. Lower is better.

### Steps
1. Install PySpark and initialize Spark session
2. Load dataset using `readVectorsSeq` → 4601 vectors of dimension 58
3. Run `kcenter(P, k=10)` and record time
4. Run `kmeansPP(P, k=10)` and compute `kmeansObj`
5. Run `kcenter(P, k1=50)` to get coreset X, then `kmeansPP(X, k=10)`, then `kmeansObj`

### Results

| Step | Method | Time | kmeansObj |
|---|---|---|---|
| 1 | kcenter(P, k=10) | 0.2481 s | — |
| 2 | kmeansPP(P, k=10) | 0.3072 s | 133,742,683.59 |
| 3 | kcenter(P,k1=50) → kmeansPP(X,k=10) | 1.5576 s | 5,149,794,765.93 |

### Key Observation
Direct k-Means++ on the full dataset gives a much lower objective than the coreset approach because the 50 farthest-first points are extreme outliers by design and miss the dense interior of the data. Increasing k1 would improve the coreset objective.

---

## Part 2 — Web Search: Inverted Index

### What it does
Builds an inverted index over a collection of webpages and answers search queries including word lookup and position finding.

### Classes Implemented
| Class | Purpose |
|---|---|
| `Position` | Stores (page, word_index) tuple |
| `MySet` | Set of pages with union and intersection |
| `WordEntry` | All positions of a word + term frequency |
| `PageIndex` | Per-page word → positions map |
| `PageEntry` | Reads file, tokenizes, builds PageIndex |
| `MyHashTable` | Custom polynomial rolling hash with chaining |
| `InvertedPageIndex` | Global word → positions across all pages |
| `SearchEngine` | Parses and executes action strings |

### Assumptions
- Stop words and punctuation are exactly as listed in the problem statement
- Word positions are 1-indexed; stop words count toward position but are not stored
- Plural-singular mapping covers only the three pairs given: stack/stacks, structure/structures, application/applications

### Steps
1. Define stop words, punctuation set, and plural map
2. Build tokenizer that cleans, splits, and assigns positions
3. Implement all 7 classes in order
4. Initialize `SearchEngine` and run all lines from `actions.txt`
5. Compare output against `answers.txt`

### Results
Output matched `answers.txt` exactly — all 11 lines correct.

```
No webpage contains word delhi
stack_datastructure_wiki
stack_datastructure_wiki
Webpage stack_datastructure_wiki does not contain word magazines
No webpage contains word allain
stack_cprogramming
stack_cprogramming
stack_cprogramming
stack_oracle
stack_cprogramming, stack_datastructure_wiki, stackoverflow
stackmagazine
```

---

## Part 3 — PageRank on Spark

### What it does
Implements the iterative PageRank algorithm on a directed graph to rank nodes by importance using β = 0.8 and 40 iterations.

### Formula
```
r(t) = (1 - β) / n  +  β × M × r(t-1)
```
where M is the column-stochastic transition matrix and r is the rank vector.

### Steps
1. Initialize Spark context
2. Load graph using Spark RDD → parse, remove self-loops, deduplicate with `.distinct()`
3. Collect edges to driver, build NumPy transition matrix M
4. Detect dangling nodes (nodes with no outgoing edges)
5. Run 40 iterations with dangling node correction
6. Verify on small graph (expected top score ≈ 0.036)
7. Run on full graph and report top 5 and bottom 5 nodes

### Dangling Node Fix
The dataset contains nodes that receive links but have no outgoing edges. Without correction, rank flows into them and never comes back out, causing the rank sum to drop below 1.0. Fix: at each iteration, the total rank held by dangling nodes is redistributed uniformly to all nodes — equivalent to replacing zero columns of M with (1/n) × ones.

### Results

**Small Graph (100 nodes, 950 edges)**

| Rank | Node | Score |
|---|---|---|
| #1 | 53 | 0.035731 |
| #2 | 14 | 0.034171 |
| #3 | 40 | 0.033630 |
| #4 | 1 | 0.030006 |
| #5 | 27 | 0.029720 |

Top score = 0.035731 (expected ≈ 0.036) ✓
Sum of ranks = 1.000000 ✓

**Whole Graph (1000 nodes, 8161 edges)**

Top 5 nodes: **263, 537, 965, 243, 285**
Bottom 5 nodes: **408, 424, 62, 93, 558**
Sum of ranks = 1.000000 ✓
Runtime = 0.030 seconds

---

## Notes
- Spark was used for file loading and edge deduplication (RDD `.distinct()`) as required
- Iterative computation uses NumPy for speed — running the loop inside Spark RDD joins caused 15+ minute runtimes on Colab due to task scheduling overhead
- All outputs verified against expected values provided in the assignment
