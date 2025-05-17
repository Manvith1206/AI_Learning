import os
import json
import requests
import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

def get_google_history(start_date, end_date):
    """
    Retrieves Google search history between specified dates.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        list: Search history entries between the specified dates
    """
    # Convert string dates to datetime objects
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=1)
    
    # OAuth 2.0 scope for Google My Activity data
    SCOPES = ['https://www.googleapis.com/auth/userinfo.profile']
    
    creds = None
    # Check if token file exists with saved credentials
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_info(json.load(open('token.json')))
    
    # If no valid credentials, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for future use
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    # Use Google My Activity API (this is through Google Takeout API)
    service = build('myactivity', 'v1', credentials=creds)
    
    # Get history data
    # Note: This is a simplified approach - actual implementation may vary
    # as Google doesn't have a direct public API for search history
    filePath = "C:/Users/Maneesh/AppData/Local/Temp/01999f66-df95-4f2e-bdab-74459fd25025_takeout-20250516T091613Z-1-001.zip.025/Takeout/Chrome/History.json"
    results = parse_takeout_data(filePath, start_date, end_date)
    
    # In practice, you'll need to use Google Takeout to download your data
    # and then parse the JSON file
    print("Note: Direct API access to search history is limited.")
    print("Please follow these alternative steps:")
    print("1. Go to https://takeout.google.com/")
    print("2. Select only 'My Activity' data")
    print("3. Click 'Next step' and create export")
    print("4. Download the export when ready")
    print("5. Use this function to parse the downloaded JSON file:")
    
    def parse_takeout_data(file_path, start_date, end_date):
        """Parse Google Takeout data file for search history."""
        filtered_history = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            # Check if it's a search activity
            if 'search' in item.get('title', '').lower():
                # Parse the timestamp
                timestamp = item.get('time')
                if timestamp:
                    activity_time = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    # Filter by date range
                    if start_datetime <= activity_time <= end_datetime:
                        filtered_history.append({
                            'query': item.get('title', '').replace('Searched for ', ''),
                            'time': activity_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'details': item.get('details', [])
                        })
        
        return filtered_history
    
    print("\nExample usage after downloading Takeout data:")
    print("history = parse_takeout_data('Takeout/My Activity/Search/MyActivity.json', '2023-01-01', '2023-01-31')")
    
    return results

# Example usage:
history = get_google_history('2023-01-01', '2023-01-31')
for entry in history:
    print(f"{entry['time']}: {entry['query']}")