# RAG Evaluation Demonstrations

This folder contains three demonstration scripts that show different approaches to evaluating your RAG application.

## Prerequisites

1. Make sure your `.env` file is properly configured with API keys
2. Ensure your RAG pipeline is working and has indexed documents
3. Install required dependencies

## Evaluation Scripts

### 1. Basic Evaluation (`basic_evaluation.py`)

**Purpose**: Demonstrates fundamental RAG evaluation with simple metrics

**What it does**:
- Tests basic quality metrics (answer length, source availability, keyword relevance)
- Evaluates a small set of questions
- Provides a foundation for understanding RAG evaluation

**Run with**:
```bash
python evals/basic_evaluation.py
```

**Key Metrics**:
- Answer length and quality
- Source attribution
- Keyword relevance to expected answers
- Overall quality score

### 2. Comparative Evaluation (`comparative_evaluation.py`)

**Purpose**: Compares different RAG configurations to find the best setup

**What it does**:
- Tests multiple pipeline configurations side by side
- Measures performance and quality differences
- Ranks configurations by overall performance
- Provides detailed comparison analysis

**Run with**:
```bash
python evals/2_comparative_evaluation.py
```

**Key Metrics**:
- Response time comparison
- Quality score differences
- Source retrieval effectiveness
- Overall ranking and recommendations

### 3. Continuous Evaluation (`3_continuous_evaluation.py`)

**Purpose**: Simulates production monitoring for ongoing quality assurance

**What it does**:
- Monitors RAG performance over simulated user sessions
- Detects performance degradation and issues
- Tracks trends and generates alerts
- Provides continuous monitoring insights

**Run with**:
```bash
python evals/3_continuous_evaluation.py
```

**Key Features**:
- Session-based evaluation
- Alert system for performance issues
- Trend analysis over time
- Production-ready monitoring concepts

## Output

All scripts generate results in the `evals/results/` directory:
- `basic_evaluation_results.json` - Basic evaluation metrics
- `comparative_evaluation_results.json` - Configuration comparison data  
- `continuous_evaluation_results.json` - Monitoring and trend data

## Customization

You can customize these scripts by:
- Modifying the test questions in each script
- Adjusting evaluation metrics and thresholds
- Adding new pipeline configurations to compare
- Changing alert thresholds for continuous monitoring

## Integration with Your Application

These demonstrations show patterns you can integrate into your production RAG application:
- Use basic evaluation for development testing
- Use comparative evaluation when choosing configurations
- Use continuous evaluation for production monitoring