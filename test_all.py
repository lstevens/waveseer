#!/usr/bin/env python
"""
Comprehensive test script for Waveseer components.

This script tests the entire Waveseer system:
1. API endpoints
2. Pattern detection
3. Database connections
4. Annotation data storage
"""

import os
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, Any


def test_api(base_url: str = "http://localhost:9000") -> Dict[str, Any]:
    """Test all API endpoints."""
    results = {
        "endpoints_tested": 0,
        "endpoints_passed": 0,
        "failures": []
    }

    print("Testing API endpoints...")

    # Test endpoints
    endpoints = [
        {"method": "GET", "url": "/health", "expect_status": 200},
        {"method": "GET", "url": "/models", "expect_status": 200},
        {"method": "GET", "url": "/catalog", "expect_status": 200},
        {"method": "POST", "url": "/match", "data": {"seq": [100, 105, 110, 105, 100], "tf": "1h", "use_ml": False}, "expect_status": 200},
        {"method": "POST", "url": "/batch/match", "data": {"sequences": [[100, 105, 110, 105, 100]], "tf": "1h", "use_ml": False}, "expect_status": 200},
        {"method": "GET", "url": "/test-model/test_model", "expect_status": 200}
    ]

    for endpoint in endpoints:
        results["endpoints_tested"] += 1
        try:
            url = f"{base_url}{endpoint['url']}"
            if endpoint["method"] == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=endpoint["data"], timeout=10)

            if response.status_code == endpoint["expect_status"]:
                print(f"✅ {endpoint['method']} {endpoint['url']} - Status: {response.status_code}")
                results["endpoints_passed"] += 1
            else:
                print(f"❌ {endpoint['method']} {endpoint['url']} - Status: {response.status_code}, Expected: {endpoint['expect_status']}")
                results["failures"].append({
                    "endpoint": endpoint["url"],
                    "method": endpoint["method"],
                    "expected_status": endpoint["expect_status"],
                    "actual_status": response.status_code,
                    "response": str(response.text)[:100]  # Truncate response
                })

        except Exception as e:
            print(f"❌ {endpoint['method']} {endpoint['url']} - Error: {str(e)}")
            results["failures"].append({
                "endpoint": endpoint["url"],
                "method": endpoint["method"],
                "error": str(e)
            })

    results["success_rate"] = results["endpoints_passed"] / results["endpoints_tested"] if results["endpoints_tested"] > 0 else 0

    return results


def test_database(db_path: str = "motifs.db") -> Dict[str, Any]:
    """Test database connectivity and queries."""
    results = {
        "connected": False,
        "tables_found": [],
        "sample_data": {},
        "errors": []
    }

    print(f"\nTesting database at {db_path}...")

    try:
        import duckdb
        conn = duckdb.connect(db_path)
        results["connected"] = True

        # Get list of tables
        tables = conn.execute("SHOW TABLES").fetchall()
        results["tables_found"] = [t[0] for t in tables]

        # Check for patterns table
        if "patterns" in results["tables_found"]:
            patterns = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
            sample = conn.execute("SELECT * FROM patterns LIMIT 3").fetchall()
            results["sample_data"]["patterns"] = {
                "count": patterns,
                "sample": [{"id": row[0], "label": row[1]} for row in sample]
            }
            print(f"✅ Found patterns table with {patterns} records")
        else:
            print("❌ Patterns table not found")
            results["errors"].append("Patterns table not found")

        conn.close()

    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        results["errors"].append(str(e))

    return results


def test_docker_config() -> Dict[str, Any]:
    """Test Docker configuration and container builds."""
    results = {
        "dockerfile_exists": False,
        "docker_compose_exists": False,
        "volumes_ready": False,
        "errors": []
    }

    print("\nTesting Docker configuration...")

    # Check Dockerfile
    if os.path.exists("Dockerfile"):
        results["dockerfile_exists"] = True
        print("✅ Dockerfile exists")
    else:
        print("❌ Dockerfile not found")
        results["errors"].append("Dockerfile not found")

    # Check docker-compose.yml
    if os.path.exists("docker-compose.yml"):
        results["docker_compose_exists"] = True
        print("✅ docker-compose.yml exists")
    else:
        print("❌ docker-compose.yml not found")
        results["errors"].append("docker-compose.yml not found")

    # Check volume directories
    volume_dirs = ["models", "data"]
    missing_dirs = []

    for dir_name in volume_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)

    if not missing_dirs:
        results["volumes_ready"] = True
        print("✅ Volume directories exist")
    else:
        print(f"❌ Missing volume directories: {', '.join(missing_dirs)}")
        results["errors"].append(f"Missing volume directories: {', '.join(missing_dirs)}")

    return results


def test_annotation_interface(api_url: str = "http://localhost:9000") -> Dict[str, Any]:
    """Test the annotation interface."""
    results = {
        "api_connection": False,
        "feedback_file_exists": False,
        "errors": []
    }

    print("\nTesting annotation interface...")

    # Check API connection
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            results["api_connection"] = True
            print(f"✅ API connection successful at {api_url}")
        else:
            print(f"❌ API returned non-200 status: {response.status_code}")
            results["errors"].append(f"API returned non-200 status: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to connect to API: {str(e)}")
        results["errors"].append(f"API connection error: {str(e)}")

    # Check feedback file
    feedback_file = os.path.expanduser("~/.waveseer/data/annotations.json")
    if os.path.exists(feedback_file):
        results["feedback_file_exists"] = True
        print(f"✅ Annotation feedback file exists at {feedback_file}")
    else:
        # Try to find it in the project directory
        alt_path = Path("data/annotations.json")
        if alt_path.exists():
            results["feedback_file_exists"] = True
            print(f"✅ Annotation feedback file exists at {alt_path}")
        else:
            print("❌ Annotation feedback file not found")
            results["errors"].append("Annotation feedback file not found")

    return results


def run_all_tests(api_url: str = "http://localhost:9000") -> Dict[str, Any]:
    """Run all tests and return combined results."""
    start_time = time.time()

    print("=" * 50)
    print("WAVESEER SYSTEM TEST")
    print("=" * 50)

    results = {
        "api": test_api(api_url),
        "database": test_database(),
        "docker": test_docker_config(),
        "annotation": test_annotation_interface(api_url)
    }

    # Calculate overall success rate
    api_success = results["api"]["success_rate"] if "success_rate" in results["api"] else 0
    db_success = 1.0 if results["database"]["connected"] and len(results["database"]["errors"]) == 0 else 0
    docker_success = 1.0 if len(results["docker"]["errors"]) == 0 else 0
    anno_success = 1.0 if len(results["annotation"]["errors"]) == 0 else 0

    results["overall_success"] = (api_success + db_success + docker_success + anno_success) / 4
    results["test_duration"] = time.time() - start_time

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"API Success Rate: {api_success * 100:.2f}%")
    print(f"Database Success: {'✅ Pass' if db_success == 1.0 else '❌ Fail'}")
    print(f"Docker Config: {'✅ Pass' if docker_success == 1.0 else '❌ Fail'}")
    print(f"Annotation Interface: {'✅ Pass' if anno_success == 1.0 else '❌ Fail'}")
    print("-" * 50)
    print(f"Overall Success Rate: {results['overall_success'] * 100:.2f}%")
    print(f"Test Duration: {results['test_duration']:.2f} seconds")
    print("=" * 50)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Waveseer system tests")
    parser.add_argument("--api-url", default="http://localhost:9000", help="API base URL")
    parser.add_argument("--output", help="Output file path for test results (JSON)")

    args = parser.parse_args()

    results = run_all_tests(args.api_url)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nTest results saved to {args.output}")
