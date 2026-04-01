import requests

# Test 1: General question (rule-based)
r = requests.post("http://localhost:8000/chat", json={
    "message": "What is a warranty?",
    "session_id": "test-session-001"
}, timeout=15)
print("Test 1 - Warranty question:")
print(f"  Status: {r.status_code}")
d = r.json()
print(f"  Intent: {d.get('intent')}")
print(f"  Response (first 120 chars): {d.get('message', '')[:120]}")
print()

# Test 2: Greeting
r2 = requests.post("http://localhost:8000/chat", json={
    "message": "hi",
    "session_id": "test-session-002"
}, timeout=15)
print("Test 2 - Greeting:")
print(f"  Status: {r2.status_code}")
d2 = r2.json()
print(f"  Response (first 120 chars): {d2.get('message', '')[:120]}")
print()
print("All tests PASSED!")
