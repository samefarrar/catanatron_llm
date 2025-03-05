import time
from typing import List, Dict, Tuple, Any, Optional
import json
import random
from enum import Enum

from catanatron.models.player import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.board import Board
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES, SETTLEMENT, CITY
from catanatron.models.map import CatanMap
from catanatron.models.decks import freqdeck_count
from catanatron.state_functions import (
    get_player_buildings,
    player_key,
    player_num_resource_cards,
    player_num_dev_cards,
    get_player_freqdeck,
    get_longest_road_length,
    get_largest_army,
)
from catanatron.state import State
from catanatron_experimental.cli.cli_players import register_player

# Constants for pretty printing
RESOURCE_EMOJI = {
    "WOOD": "🌲",
    "BRICK": "🧱",
    "SHEEP": "🐑",
    "WHEAT": "🌾",
    "ORE": "⛏️",
    None: "🏜️",
}

BUILDING_EMOJI = {
    "SETTLEMENT": "🏠",
    "CITY": "🏙️",
    "ROAD": "🛣️",
}

COSTS = {
    "ROAD": {"WOOD": 1, "BRICK": 1},
    "SETTLEMENT": {"WOOD": 1, "BRICK": 1, "WHEAT": 1, "SHEEP": 1},
    "CITY": {"WHEAT": 2, "ORE": 3},
    "DEVELOPMENT_CARD": {"SHEEP": 1, "WHEAT": 1, "ORE": 1},
}

DEV_CARD_DESCRIPTIONS = {
    "KNIGHT": "Move the robber and steal a card from a player adjacent to the new location",
    "YEAR_OF_PLENTY": "Take any 2 resources from the bank",
    "MONOPOLY": "Take all resources of one type from all other players",
    "ROAD_BUILDING": "Build 2 roads for free",
    "VICTORY_POINT": "Worth 1 victory point",
}

class LLMPlayer(Player):
    """LLM-powered player that presents game state in a way that Claude can understand."""
    # Class properties to use instead of instance variables
    debug_mode = True

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """Present game state and options to Claude in a readable format and choose an action.

        Args:
            game (Game): Complete game state (read-only)
            playable_actions (List[Action]): Available actions to choose from

        Returns:
            action (Action): Chosen action from playable_actions
        """
        state = game.state
        if self.debug_mode:
            print(f"Game ID: {game.id}")
            self._print_game_header(game)
            self._print_board_state(state)
            self._print_player_states(state)
            self._print_available_actions(playable_actions)
        # In a real implementation, this would connect to Claude via an API
        # For now, we'll randomly select an action
        return self._select_action(playable_actions, state)

    def _select_action(self, playable_actions: List[Action], state: State) -> Action:
        """Select an action from the available options.

        In a real implementation, this would call Claude to analyze the game state
        and choose an action based on the visible information.
        """
        # For demonstration, we'll pick a reasonable action
        # (This is where Claude's decision would be integrated)

        # If we're placing initial settlements, choose a decent spot (resource diversity)
        if state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
            # This would have Claude's strategy for placing initial settlements
            return playable_actions[0]  # Just pick first for now

        # If we're placing initial roads, connect to our settlement
        elif state.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
            # This would have Claude's strategy for placing initial roads
            return playable_actions[0]  # Just pick first for now

        # If we need to discard, try to keep a balanced hand
        elif state.current_prompt == ActionPrompt.DISCARD:
            # This would have Claude's strategy for discarding
            return playable_actions[0]  # Just pick first for now

        # If we're moving the robber, avoid our own hexes and target leading players
        elif state.current_prompt == ActionPrompt.MOVE_ROBBER:
            # This would have Claude's strategy for moving the robber
            return playable_actions[0]  # Just pick first for now

        # For regular turns
        elif state.current_prompt == ActionPrompt.PLAY_TURN:
            # Must roll if we haven't yet
            roll_actions = [a for a in playable_actions if a.action_type == ActionType.ROLL]
            if roll_actions:
                return roll_actions[0]

            # Prioritize based on strategy (building, dev cards, etc.)
            build_city_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_CITY]
            if build_city_actions:
                return build_city_actions[0]

            build_settlement_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
            if build_settlement_actions:
                return build_settlement_actions[0]

            # Default to ending our turn if nothing else useful
            end_turn_actions = [a for a in playable_actions if a.action_type == ActionType.END_TURN]
            if end_turn_actions:
                return end_turn_actions[0]

        # For anything else, choose randomly
        print("\nChoosing randomly from available actions.")
        return random.choice(playable_actions)

    def _print_game_header(self, game: Game) -> None:
        """Print game status header with turn information."""
        state = game.state
        print("\n" + "=" * 80)
        print(f"TURN {state.num_turns} | Player: {state.current_color().name} | Action: {state.current_prompt.name}")
        print("=" * 80)

    def _print_board_state(self, state: State) -> None:
        """Print the board state in a human-readable format."""
        board = state.board

        # Print current board information
        print("\n🏆 Game Status:")

        # Current longest road
        print(f"  Longest Road: {board.road_color.name if board.road_color else 'None'} ({board.road_length} segments)")

        # Current largest army
        largest_army_color, largest_army_size = get_largest_army(state)
        print(f"  Largest Army: {largest_army_color.name if largest_army_color else 'None'} ({largest_army_size or 0} knights)")

        # Robber location
        print(f"  Robber Location: {board.robber_coordinate}")

        # Print a simplified map visualization
        print("\n🗺️ Board Map:")
        print("  (Simplified visualization - coordinates shown as 'Q,R')")

        # Print tiles with resources
        print("  Tiles:")
        tiles_by_row = {}
        for coord, tile in board.map.land_tiles.items():
            row = coord[1]  # Get the R coordinate
            if row not in tiles_by_row:
                tiles_by_row[row] = []
            resource_emoji = RESOURCE_EMOJI[tile.resource]
            number = tile.number if tile.number is not None else ""
            robber = "🦹" if coord == board.robber_coordinate else ""
            tiles_by_row[row].append(f"({coord[0]},{coord[1]}) {resource_emoji}{number}{robber}")

        # Print the tiles row by row
        for row in sorted(tiles_by_row.keys()):
            padding = "  " * (3 + row)  # Adjust this for proper hexagonal layout
            print(f"{padding}{' '.join(tiles_by_row[row])}")

        # Print buildings on the board
        print("\n  Buildings:")
        for node_id, (color, building_type) in board.buildings.items():
            print(f"    Node {node_id:2d}: {color.name} {BUILDING_EMOJI[building_type]} ({building_type})")

        # Print roads
        print("\n  Roads:")
        roads_printed = set()
        for edge, color in board.roads.items():
            edge_tuple = tuple(sorted(edge))
            if edge_tuple in roads_printed:
                continue
            roads_printed.add(edge_tuple)
            print(f"    Edge {edge}: {color.name} {BUILDING_EMOJI['ROAD']}")

        # Print ports
        print("\n  Ports:")
        for resource, node_ids in board.map.port_nodes.items():
            resource_name = resource or "3:1"
            print(f"    {resource_name}: at nodes {node_ids}")

    def _print_player_states(self, state: State) -> None:
        """Print the status of all players in the game."""
        print("\n👥 Player Status:")
        for color in state.colors:
            hand = get_player_freqdeck(state, color)
            key = player_key(state, color)
            vp = state.player_state.get(f"{key}_VICTORY_POINTS", 0)
            longest_road = get_longest_road_length(state, color)
            settlements = get_player_buildings(state, color, SETTLEMENT)
            cities = get_player_buildings(state, color, CITY)

            # Display player info with emoji
            print(f"\n  {color.name} Player:")
            print(f"    Victory Points: {vp}")
            # print(f"    Resources ({player_num_resource_cards(state, color)} cards):")
            # for resource in RESOURCES:
            #     count = freqdeck_count(hand, resource)
            #     if count > 0:
            #         print(f"      {RESOURCE_EMOJI[resource]} {resource}: {count}")

            # Development cards
            # dev_card_count = player_num_dev_cards(state, color)
            # if dev_card_count > 0:
            #     print(f"    Development Cards ({dev_card_count} total):")
            #     dev_card_types = ["KNIGHT", "MONOPOLY", "ROAD_BUILDING", "YEAR_OF_PLENTY", "VICTORY_POINT"]
            #     for card_type in dev_card_types:
            #         count = state.player_state.get(f"{key}_{card_type}_IN_HAND", 0)
            #         if count > 0:
            #             desc = DEV_CARD_DESCRIPTIONS.get(card_type, "")
            #             print(f"      {card_type}: {count} - {desc}")

            # Building information
            print(f"    Buildings:")
            print(f"      🏠 Settlements: {len(settlements)} (Remaining: {state.player_state.get(f'{key}_SETTLEMENTS_AVAILABLE', 0)})")
            print(f"      🏙️ Cities: {len(cities)} (Remaining: {state.player_state.get(f'{key}_CITIES_AVAILABLE', 0)})")
            print(f"      🛣️ Roads: {state.player_state.get(f'{key}_ROADS_AVAILABLE', 0)}")
            print(f"      Longest Road: {longest_road} segments")

        # # Bank information
        # print("\n  🏦 Bank:")
        # # Resource freqdeck is a list where indexes represent resources: [WOOD, BRICK, SHEEP, WHEAT, ORE]
        # for i, resource in enumerate(RESOURCES):
        #     print(f"    {RESOURCE_EMOJI[resource]} {resource}: {state.resource_freqdeck[i]}")
        # print(f"    Development Cards remaining: {len(state.development_listdeck)}")

    def _print_available_actions(self, playable_actions: List[Action]) -> None:
        """Print available actions in a way Claude can understand."""
        print("\n🎮 Available Actions:")
        for i, action in enumerate(playable_actions):
            action_description = self._get_action_description(action)
            print(f"  {i}: {action_description}")

        print("\n❓ What action would you like to take? (Number from the options above)")

    def _get_action_description(self, action: Action) -> str:
        """Get a human-readable description of an action."""
        action_type = action.action_type
        value = action.value
        color = action.color

        descriptions = {
            ActionType.ROLL: "Roll the dice",
            ActionType.END_TURN: "End your turn",
            ActionType.BUY_DEVELOPMENT_CARD: f"Buy a development card (Cost: {self._format_cost('DEVELOPMENT_CARD')})",
            ActionType.PLAY_KNIGHT_CARD: "Play Knight card - Move the robber and steal a resource",
            ActionType.PLAY_YEAR_OF_PLENTY: f"Play Year of Plenty card - Take two resources: {value}",
            ActionType.PLAY_MONOPOLY: f"Play Monopoly card - Take all {value} from other players",
            ActionType.PLAY_ROAD_BUILDING: "Play Road Building card - Build two roads for free",
        }

        # Return from dictionary if action type is in it
        if action_type in descriptions:
            return descriptions[action_type]

        if action_type == ActionType.BUILD_ROAD:
            return f"Build a road at edge {value} (Cost: {self._format_cost('ROAD')})"
        elif action_type == ActionType.BUILD_SETTLEMENT:
            return f"Build a settlement at node {value} (Cost: {self._format_cost('SETTLEMENT')})"
        elif action_type == ActionType.BUILD_CITY:
            return f"Upgrade settlement to city at node {value} (Cost: {self._format_cost('CITY')})"
        elif action_type == ActionType.MOVE_ROBBER:
            target_color = value[1]
            target_str = f" and steal from {target_color.name}" if target_color else ""
            return f"Move robber to {value[0]}{target_str}"
        elif action_type == ActionType.MARITIME_TRADE:
            trade_resources = value
            offering = []

            # Check if trade_resources contains string values (resource names)
            if isinstance(trade_resources[0], str):
                # Count occurrences of each resource
                resource_counts = {}
                for resource in trade_resources[:4]:
                    if resource not in resource_counts:
                        resource_counts[resource] = 0
                    resource_counts[resource] += 1

                # Create the offering text
                for resource, count in resource_counts.items():
                    offering.append(f"{count} {resource}")

                # Determine what's being received
                receiving = trade_resources[4] if len(trade_resources) > 4 else None
            else:
                # Original code for when trade_resources contains integer counts
                for i, count in enumerate(trade_resources[:4]):
                    if isinstance(count, int) and count > 0:
                        offering.append(f"{count} {RESOURCES[i]}")
                receiving = RESOURCES[4] if len(trade_resources) > 4 and trade_resources[4] > 0 else None

            return f"Trade {', '.join(offering)} for 1 {receiving}"
        elif action_type == ActionType.DISCARD:
            if value is None:
                return "Discard half your cards (will be prompted for specifics)"
            else:
                discard_str = ", ".join([f"{count} {res}" for res, count in zip(RESOURCES, value) if count > 0])
                return f"Discard: {discard_str}"
        elif action_type == ActionType.ACCEPT_TRADE:
            offering = ", ".join([f"{count} {res}" for res, count in zip(RESOURCES, value[:5]) if count > 0])
            receiving = ", ".join([f"{count} {res}" for res, count in zip(RESOURCES, value[5:]) if count > 0])
            return f"Accept trade: Give {offering}, Receive {receiving}"
        elif action_type == ActionType.REJECT_TRADE:
            return "Reject the proposed trade"
        else:
            return f"{action_type.name}: {value}"

    def _format_cost(self, item_type: str) -> str:
        """Format the cost of an item in a readable way."""
        cost_items = []
        for resource, count in COSTS[item_type].items():
            cost_items.append(f"{count} {RESOURCE_EMOJI[resource]}{resource}")
        return ", ".join(cost_items)


# Manually register the LLMPlayer with the CLI system
register_player("LLM")(LLMPlayer)
