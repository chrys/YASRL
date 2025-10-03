"""
Continuous RAG Evaluation Demo

This script demonstrates how to set up continuous evaluation for monitoring
RAG performance over time. It simulates a production monitoring scenario
where you track quality metrics and detect performance degradation.
"""

import asyncio
import os
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from yasrl.pipeline import RAGPipeline
from yasrl.models import QueryResult

class ContinuousEvaluator:
    """
    Evaluator for continuous monitoring of RAG performance
    """
    
    def __init__(self, baseline_metrics: Dict[str, float] = None):
        self.baseline_metrics = baseline_metrics or {}
        self.alert_thresholds = {
            "response_time": 10.0,  # Alert if response time > 10 seconds
            "quality_score": 0.6,   # Alert if quality drops below 0.6
            "source_availability": 0.8,  # Alert if <80% of queries have sources
            "error_rate": 0.1       # Alert if error rate > 10%
        }
    
    def evaluate_session(self, session_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a session of RAG interactions
        
        Args:
            session_results: List of individual query results
            
        Returns:
            Session evaluation metrics
        """
        total_queries = len(session_results)
        successful_queries = [r for r in session_results if "error" not in r]
        failed_queries = [r for r in session_results if "error" in r]
        
        if not successful_queries:
            return {
                "total_queries": total_queries,
                "error_rate": 1.0,
                "alerts": ["CRITICAL: All queries failed"]
            }
        
        # Calculate metrics
        avg_response_time = sum(r["response_time"] for r in successful_queries) / len(successful_queries)
        avg_quality_score = sum(r["quality_score"] for r in successful_queries) / len(successful_queries)
        source_availability = sum(1 for r in successful_queries if r["has_sources"]) / len(successful_queries)
        error_rate = len(failed_queries) / total_queries
        
        # Check for alerts
        alerts = []
        if avg_response_time > self.alert_thresholds["response_time"]:
            alerts.append(f"HIGH_LATENCY: Average response time {avg_response_time:.2f}s exceeds threshold")
        
        if avg_quality_score < self.alert_thresholds["quality_score"]:
            alerts.append(f"LOW_QUALITY: Quality score {avg_quality_score:.2f} below threshold")
        
        if source_availability < self.alert_thresholds["source_availability"]:
            alerts.append(f"SOURCE_ISSUES: Only {source_availability:.1%} queries have sources")
        
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(f"HIGH_ERROR_RATE: {error_rate:.1%} of queries failed")
        
        # Compare with baseline if available
        baseline_comparison = {}
        if self.baseline_metrics:
            for metric in ["avg_response_time", "avg_quality_score", "source_availability"]:
                if metric in self.baseline_metrics:
                    current_value = locals()[metric]
                    baseline_value = self.baseline_metrics[metric]
                    change_pct = ((current_value - baseline_value) / baseline_value) * 100
                    baseline_comparison[metric] = {
                        "current": current_value,
                        "baseline": baseline_value,
                        "change_percent": change_pct
                    }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": total_queries,
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "avg_quality_score": avg_quality_score,
            "source_availability": source_availability,
            "alerts": alerts,
            "baseline_comparison": baseline_comparison,
            "status": "HEALTHY" if not alerts else "DEGRADED" if len(alerts) <= 2 else "CRITICAL"
        }
    
    def evaluate_single_query(self, question: str, result: QueryResult, response_time: float) -> Dict[str, Any]:
        """
        Evaluate a single query for continuous monitoring
        
        Args:
            question: The question asked
            result: The QueryResult from the pipeline
            response_time: Time taken to get the response
            
        Returns:
            Query evaluation metrics
        """
        # Quality indicators
        has_sources = len(result.source_chunks) > 0
        answer_length = len(result.answer.split())
        is_substantial = answer_length >= 10
        is_not_error_response = not any(phrase in result.answer.lower() 
                                      for phrase in ["sorry", "i don't know", "error", "failed"])
        
        # Calculate quality score
        quality_indicators = [has_sources, is_substantial, is_not_error_response]
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        return {
            "question": question,
            "answer_preview": result.answer[:100] + "..." if len(result.answer) > 100 else result.answer,
            "response_time": response_time,
            "quality_score": quality_score,
            "has_sources": has_sources,
            "source_count": len(result.source_chunks),
            "answer_length": answer_length,
            "timestamp": datetime.now().isoformat()
        }

async def simulate_user_session(pipeline: RAGPipeline, evaluator: ContinuousEvaluator, session_name: str) -> Dict[str, Any]:
    """
    Simulate a user session with multiple queries
    
    Args:
        pipeline: The RAG pipeline to evaluate
        evaluator: The continuous evaluator
        session_name: Name for this session
        
    Returns:
        Session evaluation results
    """
    print(f"\n🎭 Simulating session: {session_name}")
    
    # Define different types of queries users might ask
    query_types = {
        "basic_info": [
            "What is Happy Payments?",
            "Tell me about ISO 8583",
            "What services does Happy Payments provide?"
        ],
        "technical": [
            "How does ISO 8583 message format work?",
            "What are the key components of payment processing?",
            "Explain the payment authorization flow"
        ],
        "specific": [
            "What is field 3 in ISO 8583?",
            "How does Happy Payments handle transaction routing?",
            "What security measures are used in payment processing?"
        ]
    }
    
    # Randomly select queries for this session (simulating real user behavior)
    all_queries = []
    for queries in query_types.values():
        all_queries.extend(queries)
    
    # Simulate 3-7 queries per session
    session_size = random.randint(3, 7)
    selected_queries = random.sample(all_queries, min(session_size, len(all_queries)))
    
    session_results = []
    
    for i, question in enumerate(selected_queries, 1):
        print(f"  Query {i}/{len(selected_queries)}: {question[:50]}...")
        
        try:
            # Add some variability to simulate real-world conditions
            # Sometimes queries might be slower due to system load
            artificial_delay = random.uniform(0, 2) if random.random() < 0.3 else 0
            if artificial_delay > 0:
                await asyncio.sleep(artificial_delay)
            
            start_time = time.time()
            result = await pipeline.ask(question)
            response_time = time.time() - start_time + artificial_delay
            
            # Evaluate the query
            query_eval = evaluator.evaluate_single_query(question, result, response_time)
            session_results.append(query_eval)
            
            print(f"    ✅ Response time: {response_time:.2f}s, Quality: {query_eval['quality_score']:.2f}")
            
        except Exception as e:
            print(f"    ❌ Query failed: {e}")
            session_results.append({
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Evaluate the entire session
    session_eval = evaluator.evaluate_session(session_results)
    session_eval["session_name"] = session_name
    session_eval["queries"] = session_results
    
    return session_eval



async def run_continuous_evaluation():
    """
    Main function that demonstrates continuous RAG evaluation
    """
    print("📊 Continuous RAG Evaluation Demo")
    print("=" * 50)
    print("This demo simulates continuous monitoring of RAG performance")
    print("over multiple user sessions, tracking quality and detecting issues.")

    # Load projects from projects.json
    projects_path = Path(os.getenv("PROJECTS_FILE", "projects.json"))
    if not projects_path.exists():
        print(f"❌ Could not find projects file at {projects_path}")
        return

    with open(projects_path, "r") as f:
        projects_data = json.load(f)

    if not projects_data:
        print("❌ No projects found in projects.json.")
        return

    # List available projects
    print("Available projects:")
    project_choices = []
    for pid, info in projects_data.items():
        display = f"{info['name']} | {pid[:8]}"
        project_choices.append((display, pid))
        print(f"  {len(project_choices)}. {display}")

    # Prompt user to select a project
    while True:
        try:
            selection = int(input(f"Select a project [1-{len(project_choices)}]: "))
            if 1 <= selection <= len(project_choices):
                break
            else:
                print("Invalid selection. Try again.")
        except Exception:
            print("Please enter a valid number.")

    selected_display, selected_pid = project_choices[selection - 1]
    selected_project = projects_data[selected_pid]
    print(f"\n✅ Selected project: {selected_display}")

    try:
        # Initialize the pipeline for the selected project
        print("\n1. Initializing RAG Pipeline for selected project...")
        from yasrl.vector_store import VectorStoreManager

        # Use the same table naming logic as in the UI/app
        project_name = selected_project.get("name", "").strip()
        sanitized_name = "".join(c.lower() if c.isalnum() else "_" for c in project_name)
        sanitized_name = "_".join(filter(None, sanitized_name.split("_")))
        table_prefix = f"yasrl_{sanitized_name or selected_pid}"

        db_manager = VectorStoreManager(
            postgres_uri=os.getenv("POSTGRES_URI") or "",
            vector_dimensions=768,
            table_prefix=table_prefix
        )

        pipeline = await RAGPipeline.create(
            llm=selected_project.get("llm", "gemini"),
            embed_model=selected_project.get("embed_model", "gemini"),
            db_manager=db_manager
        )
        print("✅ Pipeline ready for continuous evaluation")

        # Initialize evaluator
        evaluator = ContinuousEvaluator()

        # Simulate multiple user sessions over time
        print("\n2. Simulating User Sessions...")
        session_types = [
            "Morning Business Users",
            "Afternoon Developers",
            "Evening Support Team",
            "Peak Hours Mixed",
            "Weekend Light Usage"
        ]

        all_sessions = []

        for session_type in session_types:
            session_result = await simulate_user_session(pipeline, evaluator, session_type)
            all_sessions.append(session_result)

            # Print session summary
            print(f"  📈 Session Status: {session_result['status']}")
            print(f"  📈 Queries: {session_result['successful_queries']}/{session_result['total_queries']}")
            print(f"  📈 Avg Quality: {session_result['avg_quality_score']:.2f}")
            print(f"  📈 Avg Response Time: {session_result['avg_response_time']:.2f}s")

            if session_result['alerts']:
                print(f"  ⚠️  Alerts: {len(session_result['alerts'])}")
                for alert in session_result['alerts']:
                    print(f"    - {alert}")

            # Simulate time passing between sessions
            await asyncio.sleep(1)

        # Analyze trends across sessions
        print("\n3. Trend Analysis")
        print("-" * 30)

        # Calculate overall metrics
        total_queries = sum(s['total_queries'] for s in all_sessions)
        total_successful = sum(s['successful_queries'] for s in all_sessions)
        overall_error_rate = 1 - (total_successful / total_queries)

        quality_trend = [s['avg_quality_score'] for s in all_sessions]
        response_time_trend = [s['avg_response_time'] for s in all_sessions]

        print(f"📊 Overall Statistics:")
        print(f"   Total Queries: {total_queries}")
        print(f"   Success Rate: {(total_successful/total_queries):.1%}")
        print(f"   Average Quality: {sum(quality_trend)/len(quality_trend):.2f}")
        print(f"   Average Response Time: {sum(response_time_trend)/len(response_time_trend):.2f}s")

        # Detect trends
        if len(quality_trend) > 2:
            quality_slope = (quality_trend[-1] - quality_trend[0]) / len(quality_trend)
            time_slope = (response_time_trend[-1] - response_time_trend[0]) / len(response_time_trend)

            print(f"\n📈 Trends:")
            if abs(quality_slope) > 0.05:
                direction = "improving" if quality_slope > 0 else "degrading"
                print(f"   Quality: {direction} (slope: {quality_slope:+.3f})")
            else:
                print(f"   Quality: stable")

            if abs(time_slope) > 0.5:
                direction = "faster" if time_slope < 0 else "slower"
                print(f"   Response Time: getting {direction} (slope: {time_slope:+.2f}s)")
            else:
                print(f"   Response Time: stable")

        # Alert summary
        all_alerts = []
        for session in all_sessions:
            all_alerts.extend(session['alerts'])

        if all_alerts:
            print(f"\n⚠️  Alert Summary ({len(all_alerts)} total):")
            alert_counts = {}
            for alert in all_alerts:
                alert_type = alert.split(':')[0]
                alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1

            for alert_type, count in sorted(alert_counts.items()):
                print(f"   {alert_type}: {count} occurrences")
        else:
            print(f"\n✅ No alerts triggered - system performing well!")

        # Save continuous monitoring results
        output_file = Path("./results") / f"continuous_evaluation_results_{sanitized_name or selected_pid}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        monitoring_report = {
            "evaluation_type": "continuous",
            "project": selected_display,
            "monitoring_period": {
                "start": all_sessions[0]['timestamp'],
                "end": all_sessions[-1]['timestamp'],
                "sessions_monitored": len(all_sessions)
            },
            "overall_metrics": {
                "total_queries": total_queries,
                "success_rate": total_successful / total_queries,
                "overall_error_rate": overall_error_rate,
                "average_quality": sum(quality_trend) / len(quality_trend),
                "average_response_time": sum(response_time_trend) / len(response_time_trend)
            },
            "trends": {
                "quality_scores": quality_trend,
                "response_times": response_time_trend
            },
            "alert_summary": {
                "total_alerts": len(all_alerts),
                "alert_types": alert_counts if all_alerts else {}
            },
            "sessions": all_sessions
        }

        with open(output_file, "w") as f:
            json.dump(monitoring_report, f, indent=2)

        print(f"\n💾 Continuous monitoring report saved to: {output_file}")

        # Cleanup
        await pipeline.cleanup()

    except Exception as e:
        print(f"❌ Error during continuous evaluation: {e}")
        raise

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Run the continuous evaluation
    asyncio.run(run_continuous_evaluation())