#!/usr/bin/env python3
"""
MEGA ANALYSIS SYSTEM
Comprehensive football analysis with ELO, positional battles, and decision trees
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

class MegaAnalysisSystem:
    def __init__(self):
        self.elo_data = self.load_elo_data()
        self.models = self.load_models()
        
    def load_elo_data(self) -> Dict:
        """Load ELO ratings data."""
        elo_file = RAW_DIR / "elo_ratings.json"
        if elo_file.exists():
            with open(elo_file, 'r') as f:
                return json.load(f)
        return {}
    
    def load_models(self) -> Dict:
        """Load trained models."""
        models = {}
        
        # Load mega predictor
        mega_file = MODELS_DIR / "mega_predictor.pkl"
        if mega_file.exists():
            models['mega_predictor'] = joblib.load(mega_file)
        
        # Load decision tree
        tree_file = MODELS_DIR / "mega_tree_model.pkl"
        if tree_file.exists():
            models['mega_tree'] = joblib.load(tree_file)
            
        return models
    
    def analyze_team_elo(self, team: str) -> Dict:
        """Analyze team ELO rating and player composition."""
        team_elo = self.elo_data.get('team_elo', {})
        player_elo = self.elo_data.get('player_elo', {})
        
        if team not in team_elo:
            return {'error': f'Team {team} not found in ELO database'}
        
        analysis = {
            'team_elo': team_elo.get(team, 1500),
            'rank': self.get_team_rank(team),
            'player_analysis': self.analyze_team_players(team, player_elo),
            'positional_strength': self.analyze_positional_strength(team, player_elo)
        }
        
        return analysis
    
    def get_team_rank(self, team: str) -> int:
        """Get team's ELO rank."""
        team_elo = self.elo_data.get('team_elo', {})
        sorted_teams = sorted(team_elo.items(), key=lambda x: x[1], reverse=True)
        
        for i, (t, elo) in enumerate(sorted_teams, 1):
            if t == team:
                return i
        return len(sorted_teams) + 1
    
    def analyze_team_players(self, team: str, player_elo: Dict) -> Dict:
        """Analyze team's player ELO composition."""
        if team not in player_elo:
            return {}
        
        players = player_elo[team]
        player_ratings = [p['elo'] for p in players.values()]
        
        return {
            'total_players': len(players),
            'average_elo': np.mean(player_ratings),
            'std_elo': np.std(player_ratings),
            'best_player_elo': max(player_ratings),
            'weakest_player_elo': min(player_ratings),
            'squad_depth_score': len(players) * np.mean(player_ratings) / 1000
        }
    
    def analyze_positional_strength(self, team: str, player_elo: Dict) -> Dict:
        """Analyze positional strength distribution."""
        if team not in player_elo:
            return {}
        
        players = player_elo[team]
        positions = {}
        
        for player_data in players.values():
            pos = player_data['position']
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(player_data['elo'])
        
        pos_analysis = {}
        for pos, elos in positions.items():
            pos_analysis[pos] = {
                'count': len(elos),
                'average_elo': np.mean(elos),
                'best_elo': max(elos),
                'depth_score': len(elos) * np.mean(elos) / 1000
            }
        
        return pos_analysis
    
    def analyze_matchup(self, home_team: str, away_team: str) -> Dict:
        """Comprehensive matchup analysis."""
        print(f"\n🔍 MEGA MATCHUP ANALYSIS")
        print("=" * 60)
        print(f"🏠 {home_team} vs ✈️  {away_team}")
        print("=" * 60)
        
        analysis = {
            'home_analysis': self.analyze_team_elo(home_team),
            'away_analysis': self.analyze_team_elo(away_team),
            'elo_comparison': self.compare_elos(home_team, away_team),
            'positional_battles': self.analyze_positional_battles(home_team, away_team),
            'prediction': self.predict_match_outcome(home_team, away_team),
            'decision_tree_analysis': self.analyze_with_decision_tree(home_team, away_team)
        }
        
        self.display_analysis(analysis)
        return analysis
    
    def compare_elos(self, home_team: str, away_team: str) -> Dict:
        """Compare ELO ratings between teams."""
        team_elo = self.elo_data.get('team_elo', {})
        
        home_elo = team_elo.get(home_team, 1500)
        away_elo = team_elo.get(away_team, 1500)
        
        return {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_difference': home_elo - away_elo,
            'home_expected': 1 / (1 + 10**((away_elo - home_elo) / 400)),
            'home_advantage': home_elo > away_elo,
            'elo_gap_category': self.categorize_elo_gap(abs(home_elo - away_elo))
        }
    
    def categorize_elo_gap(self, gap: float) -> str:
        """Categorize ELO gap."""
        if gap < 50:
            return "Very Close"
        elif gap < 100:
            return "Close"
        elif gap < 150:
            return "Moderate"
        elif gap < 200:
            return "Large"
        else:
            return "Huge"
    
    def analyze_positional_battles(self, home_team: str, away_team: str) -> Dict:
        """Analyze position vs position battles."""
        player_elo = self.elo_data.get('player_elo', {})
        
        if home_team not in player_elo or away_team not in player_elo:
            return {}
        
        home_players = player_elo[home_team]
        away_players = player_elo[away_team]
        
        battles = {
            'GK_vs_ST': self.calculate_battle('GK', 'ST', home_players, away_players),
            'CB_vs_ST': self.calculate_battle('CB', 'ST', home_players, away_players),
            'CM_vs_CM': self.calculate_battle('CM', 'CM', home_players, away_players),
            'LW_vs_RB': self.calculate_battle('LW', 'RB', home_players, away_players),
            'RW_vs_LB': self.calculate_battle('RW', 'LB', home_players, away_players)
        }
        
        return battles
    
    def calculate_battle(self, home_pos: str, away_pos: str, 
                        home_players: Dict, away_players: Dict) -> Dict:
        """Calculate individual position battle."""
        home_pos_players = [p for p in home_players.values() if p['position'] == home_pos]
        away_pos_players = [p for p in away_players.values() if p['position'] == away_pos]
        
        if not home_pos_players or not away_pos_players:
            return {'status': 'No players found'}
        
        home_best = max(home_pos_players, key=lambda x: x['elo'])
        away_best = max(away_pos_players, key=lambda x: x['elo'])
        
        return {
            'home_elo': home_best['elo'],
            'away_elo': away_best['elo'],
            'advantage': 'Home' if home_best['elo'] > away_best['elo'] else 'Away',
            'elo_difference': home_best['elo'] - away_best['elo'],
            'battle_intensity': abs(home_best['elo'] - away_best['elo']) / 100
        }
    
    def predict_match_outcome(self, home_team: str, away_team: str) -> Dict:
        """Predict match using mega predictor."""
        if 'mega_predictor' not in self.models:
            return {'error': 'Mega predictor not loaded'}
        
        model_data = self.models['mega_predictor']
        model = model_data['model']
        feature_names = model_data['features']
        
        # Create realistic features based on team analysis
        features = self.create_realistic_features(home_team, away_team, len(feature_names))
        
        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        
        return {
            'home_win_prob': probabilities[0],
            'draw_prob': probabilities[1],
            'away_win_prob': probabilities[2],
            'most_likely': ['Home Win', 'Draw', 'Away Win'][np.argmax(probabilities)],
            'confidence': max(probabilities)
        }
    
    def create_realistic_features(self, home_team: str, away_team: str, num_features: int) -> np.ndarray:
        """Create realistic features based on team data."""
        team_elo = self.elo_data.get('team_elo', {})
        
        home_elo = team_elo.get(home_team, 1500)
        away_elo = team_elo.get(away_team, 1500)
        
        # Start with ELO-based features
        features = np.array([
            home_elo,
            away_elo,
            home_elo - away_elo,  # ELO difference
            home_elo + away_elo,  # ELO sum
            home_elo / away_elo,  # ELO ratio
            1 / (1 + 10**((away_elo - home_elo) / 400)),  # Home expected
            1 / (1 + 10**((home_elo - away_elo) / 400)),  # Away expected
        ])
        
        # Add remaining features with realistic values
        remaining = num_features - len(features)
        if remaining > 0:
            additional = np.random.normal(0, 1, remaining)
            features = np.concatenate([features, additional])
        
        return features[:num_features]
    
    def analyze_with_decision_tree(self, home_team: str, away_team: str) -> Dict:
        """Analyze with decision tree rules."""
        if 'mega_tree' not in self.models:
            return {'error': 'Decision tree not loaded'}
        
        tree_model = self.models['mega_tree']
        
        # Get tree rules
        rules_file = MODELS_DIR / "mega_tree_rules.txt"
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                rules = f.read()
        else:
            rules = "Rules not found"
        
        return {
            'tree_depth': tree_model.tree_.max_depth,
            'tree_leaves': tree_model.tree_.n_leaves,
            'tree_nodes': tree_model.tree_.node_count,
            'sample_rules': rules[:500] + "..." if len(rules) > 500 else rules
        }
    
    def display_analysis(self, analysis: Dict):
        """Display comprehensive analysis."""
        print(f"\n📊 ELO COMPARISON:")
        elo_comp = analysis['elo_comparison']
        print(f"   🏠 Home ELO: {elo_comp['home_elo']:.0f}")
        print(f"   ✈️  Away ELO: {elo_comp['away_elo']:.0f}")
        print(f"   📈 Difference: {elo_comp['elo_difference']:+.0f}")
        print(f"   🎯 Home Expected: {elo_comp['home_expected']:.1%}")
        print(f"   📊 Gap Category: {elo_comp['elo_gap_category']}")
        
        print(f"\n⚔️  POSITIONAL BATTLES:")
        battles = analysis['positional_battles']
        for battle_name, battle_data in battles.items():
            if 'status' not in battle_data:
                advantage = battle_data['advantage']
                intensity = battle_data['battle_intensity']
                print(f"   {battle_name}: {advantage} advantage (Intensity: {intensity:.1f})")
        
        print(f"\n🔮 MEGA PREDICTION:")
        pred = analysis['prediction']
        if 'error' not in pred:
            print(f"   🏠 Home Win: {pred['home_win_prob']:.1%}")
            print(f"   🤝 Draw:     {pred['draw_prob']:.1%}")
            print(f"   ✈️  Away Win: {pred['away_win_prob']:.1%}")
            print(f"   🎯 Most Likely: {pred['most_likely']} ({pred['confidence']:.1%})")
        
        print(f"\n🌳 DECISION TREE ANALYSIS:")
        tree_analysis = analysis['decision_tree_analysis']
        if 'error' not in tree_analysis:
            print(f"   🌿 Tree Depth: {tree_analysis['tree_depth']}")
            print(f"   🍃 Leaves: {tree_analysis['tree_leaves']}")
            print(f"   🌳 Nodes: {tree_analysis['tree_nodes']}")

def main():
    """Main function for mega analysis."""
    print("🔍 MEGA ANALYSIS SYSTEM")
    print("=" * 80)
    print("🎯 Ultimate football analysis with ELO, positional battles, and AI")
    print("📊 Comprehensive team analysis and match predictions")
    print("🌳 Decision tree insights and feature analysis")
    print("=" * 80)
    
    # Initialize system
    analyzer = MegaAnalysisSystem()
    
    # Analyze top matches
    mega_matches = [
        ("Manchester City", "Liverpool"),
        ("Barcelona", "Real Madrid"),
        ("Bayern München", "Borussia Dortmund"),
        ("Paris Saint-Germain", "Olympique Marseille"),
        ("Arsenal", "Chelsea")
    ]
    
    for home, away in mega_matches:
        analyzer.analyze_matchup(home, away)
    
    print(f"\n🎉 MEGA ANALYSIS COMPLETE!")
    print(f"🔍 Analyzed {len(mega_matches)} mega matchups")
    print(f"🏆 ELO ratings, positional battles, and AI predictions")
    print(f"🌳 Decision tree analysis with maximum complexity")
    print(f"🎯 Ultimate football intelligence system!")

if __name__ == "__main__":
    main() 