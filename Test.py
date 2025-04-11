import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import re
from collections import defaultdict

class CricketRAG:
    def __init__(self, data_path: str):
        """
        Initialize the Cricket RAG system with the data source
        
        Args:
            data_path: Path to JSON file containing cricket match data
        """
        self.data = self._load_data(data_path)
        self.players_index = self._build_player_index()
        self.teams_index = self._build_team_index()
        self.matches_index = self._build_matches_index()
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load cricket data from JSON file"""
        try:
            with open(data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
            
    def _build_player_index(self) -> Dict[str, List[int]]:
        """
        Build an index mapping player names to their match IDs
        """
        player_index = defaultdict(list)
        
        for i, match in enumerate(self.data):
            for team in ['team1', 'team2']:
                if team in match:
                    for player in match[team].get('players', []):
                        player_index[player['name'].lower()].append(i)
                        
        return player_index
    
    def _build_team_index(self) -> Dict[str, List[int]]:
        """
        Build an index mapping team names to their match IDs
        """
        team_index = defaultdict(list)
        
        for i, match in enumerate(self.data):
            if 'team1' in match and 'name' in match['team1']:
                team_index[match['team1']['name'].lower()].append(i)
            if 'team2' in match and 'name' in match['team2']:
                team_index[match['team2']['name'].lower()].append(i)
                
        return team_index
    
    def _build_matches_index(self) -> Dict[str, List[int]]:
        """
        Build an index mapping match attributes to match IDs
        """
        matches_index = {
            'venue': defaultdict(list),
            'date': defaultdict(list),
            'format': defaultdict(list)
        }
        
        for i, match in enumerate(self.data):
            if 'venue' in match:
                matches_index['venue'][match['venue'].lower()].append(i)
            if 'date' in match:
                matches_index['date'][match['date']].append(i)
            if 'format' in match:
                matches_index['format'][match['format'].lower()].append(i)
                
        return matches_index
    
    def _extract_player_stats(self, player_name: str, match_ids: List[int]) -> Dict[str, Any]:
        """
        Extract and aggregate statistics for a player across specified matches
        
        Args:
            player_name: Name of the player
            match_ids: List of match IDs to analyze
            
        Returns:
            Dictionary containing aggregated player statistics
        """
        stats = {
            'matches_played': 0,
            'batting': {
                'total_runs': 0,
                'innings': 0,
                'highest_score': 0,
                'fifties': 0,
                'hundreds': 0,
                'not_outs': 0,
                'balls_faced': 0
            },
            'bowling': {
                'wickets': 0,
                'overs': 0,
                'runs_conceded': 0,
                'best_figures': {'wickets': 0, 'runs': 0},
                'innings': 0
            },
            'match_details': []
        }
        
        player_name = player_name.lower()
        
        for match_id in match_ids:
            match = self.data[match_id]
            match_found = False
            match_detail = {
                'match_id': match_id,
                'date': match.get('date', 'Unknown'),
                'venue': match.get('venue', 'Unknown'),
                'teams': []
            }
            
            for team_key in ['team1', 'team2']:
                if team_key in match:
                    team = match[team_key]
                    match_detail['teams'].append(team.get('name', 'Unknown'))
                    
                    for player in team.get('players', []):
                        if player['name'].lower() == player_name:
                            match_found = True
                            # Record batting stats
                            if 'batting' in player:
                                stats['batting']['innings'] += 1
                                runs = player['batting'].get('runs', 0)
                                stats['batting']['total_runs'] += runs
                                stats['batting']['highest_score'] = max(stats['batting']['highest_score'], runs)
                                stats['batting']['balls_faced'] += player['batting'].get('balls_faced', 0)
                                
                                if runs >= 100:
                                    stats['batting']['hundreds'] += 1
                                elif runs >= 50:
                                    stats['batting']['fifties'] += 1
                                    
                                if player['batting'].get('out', True) == False:
                                    stats['batting']['not_outs'] += 1
                            
                            # Record bowling stats
                            if 'bowling' in player:
                                stats['bowling']['innings'] += 1
                                wickets = player['bowling'].get('wickets', 0)
                                runs_conceded = player['bowling'].get('runs_conceded', 0)
                                
                                stats['bowling']['wickets'] += wickets
                                stats['bowling']['runs_conceded'] += runs_conceded
                                stats['bowling']['overs'] += player['bowling'].get('overs', 0)
                                
                                # Update best bowling figures
                                if wickets > stats['bowling']['best_figures']['wickets'] or (
                                    wickets == stats['bowling']['best_figures']['wickets'] and 
                                    runs_conceded < stats['bowling']['best_figures']['runs']
                                ):
                                    stats['bowling']['best_figures'] = {
                                        'wickets': wickets,
                                        'runs': runs_conceded
                                    }
            
            if match_found:
                stats['matches_played'] += 1
                stats['match_details'].append(match_detail)
        
        # Calculate averages
        if stats['batting']['innings'] > 0:
            outs = stats['batting']['innings'] - stats['batting']['not_outs']
            stats['batting']['average'] = stats['batting']['total_runs'] / max(1, outs)
            if stats['batting']['balls_faced'] > 0:
                stats['batting']['strike_rate'] = (stats['batting']['total_runs'] / stats['batting']['balls_faced']) * 100
        
        if stats['bowling']['runs_conceded'] > 0 and stats['bowling']['wickets'] > 0:
            stats['bowling']['average'] = stats['bowling']['runs_conceded'] / stats['bowling']['wickets']
            
        return stats
    
    def _extract_team_stats(self, team_name: str, match_ids: List[int]) -> Dict[str, Any]:
        """
        Extract and aggregate statistics for a team across specified matches
        
        Args:
            team_name: Name of the team
            match_ids: List of match IDs to analyze
            
        Returns:
            Dictionary containing aggregated team statistics
        """
        stats = {
            'matches_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'highest_score': 0,
            'lowest_score': float('inf'),
            'match_details': []
        }
        
        team_name = team_name.lower()
        
        for match_id in match_ids:
            match = self.data[match_id]
            match_found = False
            team_score = 0
            opponent_name = ""
            opponent_score = 0
            result = ""
            
            for team_idx, team_key in enumerate(['team1', 'team2']):
                if team_key in match and match[team_key].get('name', '').lower() == team_name:
                    match_found = True
                    team_score = match[team_key].get('score', 0)
                    
                    # Get opponent details
                    opponent_key = 'team2' if team_key == 'team1' else 'team1'
                    if opponent_key in match:
                        opponent_name = match[opponent_key].get('name', 'Unknown')
                        opponent_score = match[opponent_key].get('score', 0)
                    
                    # Determine match result
                    if 'result' in match:
                        if match['result'].lower() == 'draw':
                            stats['draws'] += 1
                            result = "Draw"
                        elif match['result'].lower() == team_name:
                            stats['wins'] += 1
                            result = "Win"
                        else:
                            stats['losses'] += 1
                            result = "Loss"
            
            if match_found:
                stats['matches_played'] += 1
                stats['highest_score'] = max(stats['highest_score'], team_score)
                if team_score > 0:  # Only update lowest score if it's a valid score
                    stats['lowest_score'] = min(stats['lowest_score'], team_score)
                
                stats['match_details'].append({
                    'match_id': match_id,
                    'date': match.get('date', 'Unknown'),
                    'venue': match.get('venue', 'Unknown'),
                    'opponent': opponent_name,
                    'team_score': team_score,
                    'opponent_score': opponent_score,
                    'result': result
                })
        
        # Fix lowest score if no matches were found
        if stats['lowest_score'] == float('inf'):
            stats['lowest_score'] = 0
            
        # Calculate win percentage
        if stats['matches_played'] > 0:
            stats['win_percentage'] = (stats['wins'] / stats['matches_played']) * 100
            
        return stats
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a natural language query about cricket statistics
        
        Args:
            query_text: Natural language query about cricket statistics
            
        Returns:
            Dictionary containing query results
        """
        query_text = query_text.lower()
        
        # Define query patterns
        player_stats_pattern = r"(stats|statistics|performance|record|runs|wickets|batting|bowling).+?(of|by|for)\s+([a-z\s]+)"
        team_stats_pattern = r"(stats|statistics|performance|record).+?(of|by|for)\s+([a-z\s]+)\s+(team)"
        
        # Check for player statistics query
        player_match = re.search(player_stats_pattern, query_text)
        if player_match:
            player_name = player_match.group(3).strip()
            if player_name in self.players_index:
                match_ids = self.players_index[player_name]
                return {
                    "query_type": "player_statistics",
                    "entity": player_name,
                    "results": self._extract_player_stats(player_name, match_ids)
                }
        
        # Check for team statistics query
        team_match = re.search(team_stats_pattern, query_text)
        if team_match:
            team_name = team_match.group(3).strip()
            if team_name in self.teams_index:
                match_ids = self.teams_index[team_name]
                return {
                    "query_type": "team_statistics",
                    "entity": team_name,
                    "results": self._extract_team_stats(team_name, match_ids)
                }
        
        # Check for direct player name mention
        for player_name in self.players_index:
            if player_name in query_text:
                match_ids = self.players_index[player_name]
                return {
                    "query_type": "player_statistics",
                    "entity": player_name,
                    "results": self._extract_player_stats(player_name, match_ids)
                }
        
        # Check for direct team name mention
        for team_name in self.teams_index:
            if team_name in query_text and "team" in query_text:
                match_ids = self.teams_index[team_name]
                return {
                    "query_type": "team_statistics",
                    "entity": team_name,
                    "results": self._extract_team_stats(team_name, match_ids)
                }
        
        # Handle more specific queries for stats
        if "total runs" in query_text or "run count" in query_text:
            for player_name in self.players_index:
                if player_name in query_text:
                    match_ids = self.players_index[player_name]
                    stats = self._extract_player_stats(player_name, match_ids)
                    return {
                        "query_type": "specific_statistic",
                        "entity": player_name,
                        "statistic": "total_runs",
                        "value": stats['batting']['total_runs'],
                        "full_stats": stats
                    }
        
        if "wickets" in query_text:
            for player_name in self.players_index:
                if player_name in query_text:
                    match_ids = self.players_index[player_name]
                    stats = self._extract_player_stats(player_name, match_ids)
                    return {
                        "query_type": "specific_statistic",
                        "entity": player_name,
                        "statistic": "wickets",
                        "value": stats['bowling']['wickets'],
                        "full_stats": stats
                    }
        
        # If no specific pattern matches, return a generic response
        return {
            "query_type": "unknown",
            "message": "Could not determine the specific cricket statistics you're looking for. Please try again with a more specific query about a player or team."
        }

# Example usage
def main():
    # Sample data structure
    sample_data = [
        {
            "match_id": "M001",
            "date": "2023-06-15",
            "venue": "Lords",
            "format": "Test",
            "team1": {
                "name": "India",
                "score": 345,
                "players": [
                    {
                        "name": "Virat Kohli",
                        "batting": {"runs": 120, "balls_faced": 190, "out": True},
                        "bowling": {"overs": 5, "wickets": 1, "runs_conceded": 30}
                    },
                    {
                        "name": "Rohit Sharma",
                        "batting": {"runs": 83, "balls_faced": 145, "out": True}
                    }
                ]
            },
            "team2": {
                "name": "England",
                "score": 290,
                "players": [
                    {
                        "name": "Joe Root",
                        "batting": {"runs": 95, "balls_faced": 160, "out": True}
                    },
                    {
                        "name": "James Anderson",
                        "batting": {"runs": 12, "balls_faced": 35, "out": True},
                        "bowling": {"overs": 22, "wickets": 4, "runs_conceded": 62}
                    }
                ]
            },
            "result": "India"
        },
        {
            "match_id": "M002",
            "date": "2023-06-22",
            "venue": "Oval",
            "format": "Test",
            "team1": {
                "name": "England",
                "score": 375,
                "players": [
                    {
                        "name": "Joe Root",
                        "batting": {"runs": 105, "balls_faced": 180, "out": False}
                    },
                    {
                        "name": "James Anderson",
                        "batting": {"runs": 8, "balls_faced": 25, "out": True},
                        "bowling": {"overs": 24, "wickets": 3, "runs_conceded": 75}
                    }
                ]
            },
            "team2": {
                "name": "India",
                "score": 410,
                "players": [
                    {
                        "name": "Virat Kohli",
                        "batting": {"runs": 78, "balls_faced": 142, "out": True},
                        "bowling": {"overs": 4, "wickets": 0, "runs_conceded": 15}
                    },
                    {
                        "name": "Rohit Sharma",
                        "batting": {"runs": 127, "balls_faced": 200, "out": True}
                    }
                ]
            },
            "result": "India"
        }
    ]
    
    # Write sample data to a temporary file
    with open('sample_cricket_data.json', 'w') as f:
        json.dump(sample_data, f)
    
    # Initialize RAG system
    cricket_rag = CricketRAG('sample_cricket_data.json')
    
    # Test queries
    queries = [
        "What are the total runs scored by Virat Kohli?",
        "Show me the statistics of Joe Root",
        "How many wickets has James Anderson taken?",
        "Performance of India team"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = cricket_rag.query(query)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()