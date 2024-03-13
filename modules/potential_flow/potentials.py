from __future__ import annotations
import math
from typing import TYPE_CHECKING
from library import Color, Point2D, Point2DI, UNIT_TYPEID
from config import DEBUG_SCOUT, OLD_ENEMIES_ENABLED
from modules.extra import (
    get_closest,
    get_enemies_in_neighbouring_tiles,
    get_enemies_in_radius,
)
from config import DEBUG_SCOUT
TILE_SIZE = 32
SMALL = 0.01
ON = 1
OFF = 0
EXTEND = 1

from modules.potential_flow.flows import (
    enemy_pf,
    obstacle_potential,
    region_pf,
    source_potential,
    vortex_potential,
)
from modules.potential_flow.regions import Region
from modules.potential_flow.vector import Vector
from modules.scout_helper import *

from modules.py_unit import PyUnit

if TYPE_CHECKING:
    from tasks.pf_scout import PFscout

def region_pval(scout: PFscout, scout_unit: PyUnit, target_region: Region) -> Vector:
    """Start from center of the region and the combine of source and vortex potential flow"""
    cur_reg = scout.agent.region_manager.get_region(scout_unit.tile_position)
    d2_center = cur_reg.center.distance(scout_unit.position)

    source_correction = ON if cur_reg == target_region else OFF
    vortex_correction = (
        ON
        if cur_reg == target_region
        else (SMALL if d2_center < scout.DISTANCE_TO_SWITCH_SOURCE_SINK else SMALL)
    )
    scout.DISTANCE_TO_SWITCH_SOURCE_SINK = scout_unit.unit_type.sight_range + EXTEND

    if DEBUG_SCOUT:
        scout.region_potentials.clear()
        _vrtx_potential = (
            vortex_potential(cur_reg.center, scout_unit.position)
            * scout.CENTER_VORTEX
            * vortex_correction
        )
        scout.region_potentials.append(_vrtx_potential)

        _src_potential = (
            source_potential(cur_reg.center, scout_unit.position)
            * scout.CENTER_SOURCE_SINK
            * source_correction
        )

        scout.region_potentials.append(
            _src_potential
            if d2_center < scout.DISTANCE_TO_SWITCH_SOURCE_SINK
            else -_src_potential
        )

    return region_pf(
        cur_reg.center,
        scout_unit.position,
        d2_center,
        scout,
        vortex_correction,
        source_correction,
        scout.DISTANCE_TO_SWITCH_SOURCE_SINK,
    )


def border_pval(
    scout: PFscout,
    scout_unit: PyUnit,
    cur_region: Region,
    target_reg: Region,
    is_different_region: bool,
):
    detail_border: frozenset[Point2DI] = cur_region.border
    scout_position = scout_unit.position

    border_co = len(detail_border) / (math.pi * 14)
    scout.DISTANCE_TO_ACTIVE_BORDER_FLOW = max(border_co, 3)
    src_correction = (1 if scout.agent.region_manager.get_region(scout_unit.tile_position) == target_reg else 0)
    chokepoint = get_closest(scout.agent.region_manager.chokepoints_as_centers, scout_position)
    inactive_border = (scout_position.distance(chokepoint) < scout.DISTANCE_TO_ACTIVE_BORDER_FLOW + 4)

    num_border = 0
    border_pval = Vector()

    for border_tile in detail_border:
        if scout_position.distance(border_tile) < scout.DISTANCE_TO_ACTIVE_BORDER_FLOW:
            if is_different_region and inactive_border:
                continue

            vrtx_pval = vortex_potential(border_tile, scout_position) * scout.BORDER_VORTEX
            src = (
                source_potential(border_tile, scout_position)
                * scout.BORDER_SOURCE
                * src_correction
            )

            border_pval += vrtx_pval + src
            num_border += 1
    return border_pval


def attract_point_pval(scout: PFscout, scout_unit: PyUnit):
    pval = Vector()
    pos = scout_unit.position
    if scout.USE_EXTRA_ATTR:
        if scout.attract_points is None:
            scout.attract_points.add(scout.agent.region_manager.get_region(scout_unit.tile_position).center)
    for point in scout.attract_points:
        pval += source_potential(point, pos) * (-TILE_SIZE)
    return pval



def unit_pval(scout: PFscout, enemy: PyUnit, scout_unit: PyUnit):
    """
    Unit's (building and enemy) emitted potential flow
    Rotate an (-alpha) angle. Input is the alpha angle
    """
    enemy_type = enemy.unit_type
    scout_pos = scout_unit.position
    region = scout.agent.region_manager.get_region(scout_unit.tile_position)
    center = region.center if region else None
    enemy_radius = enemy.radius

    # Mineral and other indestructive obstacle
    if is_indestructable(enemy_type):  # is invincible
        return obstacle_potential(scout, enemy.position, scout_pos, center, enemy_radius * enemy_radius)
    
    elif is_building(enemy, enemy_type):
        return obstacle_potential(scout, enemy.position, scout_pos, center, enemy_radius * enemy_radius)
    
    # If is an attacking enemy unit or worker who aim at our scout's position
    elif (
        enemy.can_attack
        and enemy.position.distance(scout_pos) < enemy_type.attack_range + 4
        and ((enemy_target := enemy.get_target()) == scout_unit or not enemy_type.is_worker)
    ):
        attack_range = enemy_type.attack_range + 1
        if enemy_target == scout_unit:
            return enemy_pf(scout.ENEMY_NEEDLE, scout_pos, enemy, enemy_target, attack_range)
        else:
            return (
                source_potential(enemy.position, scout_pos)
                * scout.ENEMY_NEEDLE
                * (enemy.unit_type.attack_range - 0.5)
                * (1.0 / enemy.radius)
            )
    elif not (enemy_type.is_combat_unit) and (enemy.is_flying or enemy.is_burrowed):
        return Vector()
    return obstacle_potential(scout, enemy.position, scout_pos, center, enemy_radius * enemy_radius)


def calculate_pval(scout: PFscout, scout_unit: PyUnit):
    cur_region = scout.agent.region_manager.get_region(scout_unit.tile_position)

    next_vector = Vector() # the potential value:

    enemy_direction = Vector()
    enemy_num = 0
    obstacle_val = Vector()
    obstacle_num = 0
    units = []

    enemies = get_enemies_in_neighbouring_tiles(
        scout.agent,
        scout_unit.tile_position,
        fast=False,
        dist=scout_unit.unit_type.sight_range + 2,
    )
    scout.enemies = enemies

    # Calculate unitPVal
    for enemy in scout.get_enemies():
        enemy_pval = unit_pval(scout, enemy, scout_unit)
        if enemy_targeting_scout(enemy, scout_unit):
            enemy_direction += enemy_pval
            enemy_num += 1
            next_vector += enemy_pval
        else:
            # obstacle
            obstacle_num += 1
            obstacle_val += enemy_pval
            if enemy_pval:
                units.append(enemy_pval)

    # Add back obstacle value after averaged
    if obstacle_num:
        next_vector += obstacle_val * (1.0 / obstacle_num)

    # Calculate regionPVal
    region_val = region_pval(scout, scout_unit, scout.target_region)
    next_vector += region_val

    # Calculate borderPVal
    curr_border_pval = border_pval(
        scout,
        scout_unit,
        cur_region,
        scout.target_region,
        cur_region != scout.target_region,
    )
    next_vector += curr_border_pval

    enemy_infront(scout, enemy_direction)

    return next_vector


def enemy_infront(scout: PFscout, enemy_direction):
    if enemy_direction.length() < 0:
        return
    cos_enemy = enemy_direction.cos(region_val)
    coses = [cos_enemy]
    for cos_enemy in coses:
        if (
            (cos_enemy < -0.5 and enemy_num >= 3)
            or (cos_enemy < -0.85)
            or curr_border_pval.cos(att_tmp) < -0.5
        ):
            scout.last_reverse_frame = scout.agent.current_frame
            scout.forward *= -1