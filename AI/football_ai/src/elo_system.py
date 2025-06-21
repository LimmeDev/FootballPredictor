#!/usr/bin/env python3
"""
Comprehensive ELO Rating System for Football
Team ELO, Player ELO, Position vs Position Analysis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data"

class FootballELOSystem:
    def __init__(self):
        self.team_elo = {}
        self.player_elo = {}
        self.position_elo = {}
        self.historical_elo = []
        
        # ELO parameters
        self.initial_elo = 1500
        self.k_factor_team = 32
        self.k_factor_player = 20
        self.k_factor_position = 16
        
        # Position mappings
        self.positions = {
            'GK': 'Goalkeeper',
            'CB': 'Centre-Back', 'LB': 'Left-Back', 'RB': 'Right-Back',
            'CDM': 'Defensive Midfielder', 'CM': 'Central Midfielder', 'CAM': 'Attacking Midfielder',
            'LM': 'Left Midfielder', 'RM': 'Right Midfielder',
            'LW': 'Left Winger', 'RW': 'Right Winger',
            'CF': 'Centre Forward', 'ST': 'Striker'
        }
        
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using ELO formula."""
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update_elo(self, old_rating: float, actual_score: float, expected_score: float, k_factor: float) -> float:
        """Update ELO rating."""
        return old_rating + k_factor * (actual_score - expected_score)
    
    def get_team_elo(self, team: str) -> float:
        """Get team ELO rating."""
        return self.team_elo.get(team, self.initial_elo)
    
    def process_match_result(self, home_team: str, away_team: str, 
                           home_goals: int, away_goals: int, 
                           match_date: str = None) -> Dict:
        """Process match result and update ELO ratings."""
        
        # Get current ratings
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        
        # Calculate expected scores
        home_expected = self.calculate_expected_score(home_elo, away_elo)
        away_expected = 1 - home_expected
        
        # Calculate actual scores
        if home_goals > away_goals:
            home_actual, away_actual = 1.0, 0.0
        elif home_goals < away_goals:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5
        
        # Goal difference bonus/penalty
        goal_diff = abs(home_goals - away_goals)
        multiplier = math.log(max(goal_diff, 1)) + 1
        
        # Update ratings
        k_factor = self.k_factor_team * multiplier
        new_home_elo = self.update_elo(home_elo, home_actual, home_expected, k_factor)
        new_away_elo = self.update_elo(away_elo, away_actual, away_expected, k_factor)
        
        # Store new ratings
        self.team_elo[home_team] = new_home_elo
        self.team_elo[away_team] = new_away_elo
        
        # Store historical data
        self.historical_elo.append({
            'date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_elo_before': home_elo,
            'away_elo_before': away_elo,
            'home_elo_after': new_home_elo,
            'away_elo_after': new_away_elo,
            'home_expected': home_expected,
            'away_expected': away_expected
        })
        
        return {
            'home_elo_change': new_home_elo - home_elo,
            'away_elo_change': new_away_elo - away_elo,
            'home_elo_new': new_home_elo,
            'away_elo_new': new_away_elo
        }
    
    def create_player_elo_simulation(self, teams: List[str]) -> Dict:
        """Create simulated player ELO ratings for teams."""
        print("👥 Creating player ELO simulation...")
        
        players_per_team = 25
        positions_per_team = {
            'GK': 3, 'CB': 4, 'LB': 2, 'RB': 2,
            'CDM': 2, 'CM': 4, 'CAM': 2,
            'LM': 1, 'RM': 1, 'LW': 2, 'RW': 2,
            'ST': 2
        }
        
        player_elo = {}
        
        for team in teams:
            team_base_elo = self.get_team_elo(team)
            team_players = {}
            
            for position, count in positions_per_team.items():
                for i in range(count):
                    # Vary player ELO around team ELO
                    variation = np.random.normal(0, 100)  # ±100 ELO variation
                    position_bonus = {
                        'GK': -50, 'CB': -20, 'LB': 0, 'RB': 0,
                        'CDM': 10, 'CM': 20, 'CAM': 40,
                        'LM': 30, 'RM': 30, 'LW': 50, 'RW': 50,
                        'ST': 60
                    }.get(position, 0)
                    
                    player_name = f"{team}_{position}_{i+1}"
                    player_rating = max(1000, team_base_elo + variation + position_bonus)
                    
                    team_players[player_name] = {
                        'elo': player_rating,
                        'position': position,
                        'team': team
                    }
            
            player_elo[team] = team_players
        
        self.player_elo = player_elo
        print(f"✅ Created {sum(len(players) for players in player_elo.values())} player ratings")
        return player_elo
    
    def calculate_positional_matchups(self, home_team: str, away_team: str) -> Dict:
        """Calculate position vs position ELO matchups."""
        if not self.player_elo:
            return {}
        
        home_players = self.player_elo.get(home_team, {})
        away_players = self.player_elo.get(away_team, {})
        
        matchups = {}
        
        # Direct position matchups
        position_battles = {
            'GK_vs_ST': ('GK', 'ST'),
            'CB_vs_ST': ('CB', 'ST'),
            'LB_vs_RW': ('LB', 'RW'),
            'RB_vs_LW': ('RB', 'LW'),
            'CDM_vs_CAM': ('CDM', 'CAM'),
            'CM_vs_CM': ('CM', 'CM'),
            'LW_vs_RB': ('LW', 'RB'),
            'RW_vs_LB': ('RW', 'LB'),
            'ST_vs_CB': ('ST', 'CB')
        }
        
        for battle_name, (home_pos, away_pos) in position_battles.items():
            # Get best players in each position
            home_pos_players = [p for p in home_players.values() if p['position'] == home_pos]
            away_pos_players = [p for p in away_players.values() if p['position'] == away_pos]
            
            if home_pos_players and away_pos_players:
                home_best = max(home_pos_players, key=lambda x: x['elo'])
                away_best = max(away_pos_players, key=lambda x: x['elo'])
                
                # Calculate advantage
                elo_diff = home_best['elo'] - away_best['elo']
                expected_home = self.calculate_expected_score(home_best['elo'], away_best['elo'])
                
                matchups[battle_name] = {
                    'home_elo': home_best['elo'],
                    'away_elo': away_best['elo'],
                    'elo_difference': elo_diff,
                    'home_advantage': expected_home,
                    'battle_intensity': abs(elo_diff) / 100  # 0-10 scale
                }
        
        return matchups
    
    def calculate_team_composition_strength(self, team: str) -> Dict:
        """Calculate team composition and positional strength."""
        if team not in self.player_elo:
            return {}
        
        players = self.player_elo[team]
        composition = {}
        
        # Calculate positional strengths
        for position in self.positions.keys():
            pos_players = [p for p in players.values() if p['position'] == position]
            if pos_players:
                elos = [p['elo'] for p in pos_players]
                composition[f'{position}_strength'] = np.mean(elos)
                composition[f'{position}_depth'] = len(elos)
                composition[f'{position}_best'] = max(elos)
        
        # Overall team metrics
        all_elos = [p['elo'] for p in players.values()]
        composition['team_avg_elo'] = np.mean(all_elos)
        composition['team_elo_std'] = np.std(all_elos)
        composition['team_depth'] = len(all_elos)
        composition['star_player_elo'] = max(all_elos)
        composition['weakest_player_elo'] = min(all_elos)
        
        # Formation analysis (simplified)
        def_players = sum(1 for p in players.values() if p['position'] in ['CB', 'LB', 'RB'])
        mid_players = sum(1 for p in players.values() if p['position'] in ['CDM', 'CM', 'CAM', 'LM', 'RM'])
        att_players = sum(1 for p in players.values() if p['position'] in ['LW', 'RW', 'ST', 'CF'])
        
        composition['formation_def_strength'] = def_players
        composition['formation_mid_strength'] = mid_players
        composition['formation_att_strength'] = att_players
        
        return composition

def build_comprehensive_elo_dataset():
    """Build comprehensive ELO dataset for all teams."""
    print("🏆 BUILDING COMPREHENSIVE ELO SYSTEM")
    print("=" * 60)
    
    # Initialize ELO system
    elo_system = FootballELOSystem()
    
    # Load match data
    openfootball_file = PROCESSED_DIR / "openfootball_features.parquet"
    if not openfootball_file.exists():
        print("❌ No match data found! Run parse_openfootball.py first.")
        return False
    
    df = pd.read_parquet(openfootball_file)
    print(f"📊 Processing {len(df)} matches for ELO calculation...")
    
    # Sort by date to process chronologically
    df = df.sort_values('Date')
    
    # Process all matches to build ELO history
    for idx, row in df.iterrows():
        elo_system.process_match_result(
            row['HomeTeam'], row['AwayTeam'],
            row['Home_Goals'], row['Away_Goals'],
            str(row['Date'])
        )
    
    print(f"✅ Processed {len(elo_system.historical_elo)} matches")
    print(f"🏆 Created ELO ratings for {len(elo_system.team_elo)} teams")
    
    # Get all teams
    teams = list(elo_system.team_elo.keys())
    
    # Create player ELO simulation
    elo_system.create_player_elo_simulation(teams)
    
    # Save ELO data
    elo_data = {
        'team_elo': elo_system.team_elo,
        'player_elo': elo_system.player_elo,
        'historical_elo': elo_system.historical_elo[-1000:],  # Last 1000 matches
        'parameters': {
            'initial_elo': elo_system.initial_elo,
            'k_factor_team': elo_system.k_factor_team,
            'k_factor_player': elo_system.k_factor_player
        }
    }
    
    elo_file = RAW_DIR / "elo_ratings.json"
    with open(elo_file, 'w') as f:
        json.dump(elo_data, f, indent=2, default=str)
    
    print(f"💾 Saved ELO data to: {elo_file}")
    
    # Show top teams
    print("\n🏆 TOP 10 TEAMS BY ELO:")
    sorted_teams = sorted(elo_system.team_elo.items(), key=lambda x: x[1], reverse=True)
    for i, (team, elo) in enumerate(sorted_teams[:10], 1):
        print(f"   {i:2d}. {team}: {elo:.0f}")
    
    # Show ELO distribution
    elos = list(elo_system.team_elo.values())
    print(f"\n📊 ELO Distribution:")
    print(f"   Mean: {np.mean(elos):.0f}")
    print(f"   Std:  {np.std(elos):.0f}")
    print(f"   Range: {min(elos):.0f} - {max(elos):.0f}")
    
    return elo_system

if __name__ == "__main__":
    elo_system = build_comprehensive_elo_dataset()
    if elo_system:
        print("\n🎉 ELO system built successfully!")
        print("Ready for mega feature engineering!")
    else:
        print("❌ Failed to build ELO system")
        exit(1) 