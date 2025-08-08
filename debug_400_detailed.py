#!/usr/bin/env python3
"""
Debug the 400 error with detailed error response.
"""

import sys
sys.path.insert(0, '/home/GOD/heaven-base')
import asyncio
import json
import requests
from heaven_base.utils.auto_summarize import AggregationSummarizerAgent

async def debug_400_with_response():
    print("=== Debugging 400 Error with Response Details ===")
    
    # Load the failing payload
    with open('/home/GOD/heaven-base/failing_request_2.json', 'r') as f:
        failing_payload = json.load(f)
    
    # Extract the uni-api request data
    payload = {
        "model": failing_payload["model"],
        "messages": failing_payload["messages"],
        **failing_payload["kwargs"]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-heaven-uni-api-test-12345"
    }
    
    uni_api_url = "http://172.18.0.5:8000/v1/chat/completions"
    
    print(f"Sending request to {uni_api_url}")
    print(f"Payload size: {len(json.dumps(payload))} characters")
    
    try:
        response = requests.post(uni_api_url, headers=headers, json=payload, timeout=120)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Response headers: {dict(response.headers)}")
            print(f"Response text: {response.text}")
            
            # Try to parse as JSON for structured error
            try:
                error_json = response.json()
                print(f"Error JSON: {json.dumps(error_json, indent=2)}")
            except:
                print("Could not parse error response as JSON")
        else:
            print("✅ Request succeeded!")
            
    except Exception as e:
        print(f"Request failed with exception: {e}")

if __name__ == "__main__":
    asyncio.run(debug_400_with_response())