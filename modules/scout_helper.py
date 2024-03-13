from __future__ import annotations
from typing import TYPE_CHECKING

from modules.extra import get_closest
from library import Point2D, Point2DI, PLAYER_ENEMY, UnitType
from modules.py_unit import PyUnit
from functools import cache

# --- Types ---
if TYPE_CHECKING:
    from agents.basic_agent import BasicAgent
    from modules.potential_flow.regions import Region


@cache
def get_enemy_base_location(agent: BasicAgent) -> Point2DI:
    return agent.base_location_manager.get_player_starting_base_location(
        PLAYER_ENEMY
    ).position


@cache
def get_enemy_info(agent: BasicAgent):
    """Tuple containing the enemy start position and the enemy start region."""
    enemy_base_pos = get_enemy_base_location(agent)
    return enemy_base_pos, agent.region_manager.get_region(enemy_base_pos.as_tile())


def get_closest_choke_pos(agent: BasicAgent, py_unit: PyUnit) -> Point2DI:
    chokepoints_as_centers = agent.region_manager.chokepoints_as_centers
    return get_closest(chokepoints_as_centers, py_unit.position)


def get_closest_choke_to_position(self, position):
    return get_closest(
        self.agent.region_manager.chokepoints, position, lambda c: c.center
    )


def get_enemy_expansion_info(agent: BasicAgent) -> tuple[Point2D, Region]:
    """Returns the target position and region for the given agent."""
    expansion_pos = get_enemy_expansion_pos(agent)
    target_region = agent.region_manager.get_region(expansion_pos.as_tile())
    return expansion_pos, target_region


def get_enemy_expansion_pos(agent: BasicAgent) -> Point2D:
    return agent.base_location_manager.get_next_expansion(PLAYER_ENEMY).position


def is_indestructable(type: UnitType):
    return type.is_geyser or type.is_mineral or type.is_geyser


def is_building(enemy: PyUnit, type: UnitType):
    return (
        type.is_building
        and not enemy.is_flying
        and (not type.is_combat_unit or enemy.is_being_constructed)
        and type.unit_typeid != UNIT_TYPEID.TERRAN_BUNKER
    )

def enemy_targeting_scout(enemy: PyUnit, scout_unit: PyUnit):
    return enemy.can_attack and not enemy.unit_type.is_worker or enemy.get_target() == scout_unit