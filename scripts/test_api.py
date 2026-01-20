"""
Test the API endpoint to verify the fix is working
"""
import requests
import json

def test_api():
    url = "http://localhost:5000/predict"
    
    test_cases = [
        # Good conditions
        {'crop': 'rice', 'rainfall': 800, 'temperature': 26, 'nitrogen': 100, 'phosphorus': 50, 'potassium': 50},
        {'crop': 'tea', 'rainfall': 700, 'temperature': 22, 'nitrogen': 100, 'phosphorus': 50, 'potassium': 50},
        {'crop': 'rubber', 'rainfall': 800, 'temperature': 25, 'nitrogen': 80, 'phosphorus': 40, 'potassium': 50},
        {'crop': 'sugarcane', 'rainfall': 900, 'temperature': 28, 'nitrogen': 150, 'phosphorus': 60, 'potassium': 60},
        {'crop': 'cinnamon', 'rainfall': 800, 'temperature': 28, 'nitrogen': 100, 'phosphorus': 40, 'potassium': 40},
        # Bad conditions
        {'crop': 'rice', 'rainfall': 150, 'temperature': 35, 'nitrogen': 50, 'phosphorus': 20, 'potassium': 20},
        {'crop': 'tea', 'rainfall': 150, 'temperature': 35, 'nitrogen': 50, 'phosphorus': 20, 'potassium': 20},
    ]
    
    print("API TEST RESULTS")
    print("="*70)
    print(f"{'Crop':<12} | {'Rainfall':<8} | {'Temp':<6} | {'Yield (kg/ha)':<15} | {'Status'}")
    print("-"*70)
    
    for tc in test_cases:
        try:
            response = requests.post(url, json=tc, timeout=5)
            if response.status_code == 200:
                result = response.json()
                yield_val = result.get('predicted_yield', 'N/A')
                status = "OK" if yield_val > 0 else "NEGATIVE!"
                print(f"{tc['crop']:<12} | {tc['rainfall']:<8} | {tc['temperature']:<6} | {yield_val:<15.2f} | {status}")
            else:
                print(f"{tc['crop']:<12} | ERROR: {response.status_code}")
        except Exception as e:
            print(f"{tc['crop']:<12} | FAILED: {str(e)[:30]}")
    
    print("="*70)

if __name__ == "__main__":
    test_api()
