from __future__ import annotations
from typing import TYPE_CHECKING

from config import (
    TIME_KEEP_ENEMY,
    TIME_KEEP_ENEMY_BUILDING,
    DEBUG_SCOUT,
    FRAME_SKIP_SCOUT,
    OLD_ENEMIES_ENABLED,
    FRAME_SKIP_SCOUT,
)
from modules.extra import get_closest

from modules.potential_flow.regions import Region
from modules.potential_flow.potentials import calculate_pval
from modules.potential_flow.vector import Vector
from tasks.task import Status

# from modules.scout_manager import ScoutManager
from tasks.scout import Scout
from modules.py_unit import PyUnit
from modules.extra import (
    get_enemies_in_base_location,
    get_enemies_in_neighbouring_tiles,
)
from library import Unit, Point2D, Point2DI, Color, PLAYER_ENEMY
from queue import SimpleQueue
from modules.cache_manager import (
    add_expire_instance,
    add_expire_function,
    update_cached_functions,
)
from functools import cache

if TYPE_CHECKING:
    from agents.basic_agent import BasicAgent

from modules.scout_helper import *


SECOND_IN_FRAMES = 16
CHOKE_DIST = 5


class PFscout(Scout):

    def __init__(
        self,
        scout_bases: SimpleQueue[Point2D],
        prio: int,
        agent: BasicAgent,
        # scout_manager: ScoutManager,
    ):
        super().__init__(None, prio, agent, restart_on_fail=False, is_high_freq=True)
        self.target_region: Region = None
        self.forward = 1  # 1: proceed forward, -1: reverse direction
        self.last_reverse_frame = 0
        # self.scout_manager = scout_manager
        self.reset_constants()

        self.USE_ATTRACT_PVAL = True

        self.enemies = set()
        self.attract_points: set[Point2D] = set()

        self.frame_since_switch = 0
        self.should_switch_to_expansion = False
        self.switch_complete = False


        self.USE_EXTRA_ATTR = True

        if DEBUG_SCOUT:
            self.region_potentials: list[Vector] = []
            self.border_potentials = []
            self.all_potentials = []
            self.obstacles_debug = set()
            self.unit_potentials = []

            self.show_object_r = True
            self.show_unit_p = True
            self.show_region_p = True
            self.show_border_p = True
            self.show_all_p = False

        add_expire_instance(self.agent, self)
        add_expire_function(self.agent, self, self.get_next_target, 200)

    def reset_constants(self):
        self.CENTER_VORTEX = 19
        self.CENTER_SOURCE_SINK = 3
        self.DISTANCE_TO_SWITCH_SOURCE_SINK = 4
        self.DISTANCE_TO_ACTIVE_BORDER_FLOW = 3
        self.BORDER_VORTEX = -16
        self.BORDER_SOURCE = 9
        self.BUILDING_OBSTACLE = 5
        self.ENEMY_NEEDLE = 9

    def on_start(self, py_unit: PyUnit) -> Status:
        """
        Start or restart the task.

        :return: Status.DONE if there is a list of targets and the task has been given to at suitable unit. # noqa
        Status.FAIL if unit not suitable.
        """

        self.region_10 = next(
            region for region in self.agent.region_manager.regions if region.id == 10
        )

        if py_unit.unit_type.unit_typeid in self.candidates:
            return Status.DONE
        return Status.FAIL

    def on_step(self, py_unit: PyUnit) -> Status:
        """
        Checks if the task is continuing.

        :return: Status.DONE if unit is finished scouting. Status.NOT_DONE if it keeps scouting.
        Status.FAIL if unit is idle.
        """

        self.reset_constants()
        self.scout_enemy_opening(py_unit)
        update_cached_functions(self.agent, self)

        return Status.NOT_DONE

    def potential_flow(self, py_unit):
        # Control scout's movement using potential flow
        self.move(py_unit)

        if DEBUG_SCOUT:
            self.debug(py_unit)

    def debug(self, py_unit: PyUnit):
        if self.show_object_r:
            if self.target_region:
                cur_center = self.target_region.center
                self.agent.map_tools.draw_text(cur_center, "thres", Color.GRAY)
                self.agent.map_tools.draw_circle(
                    Point2D(cur_center.x, cur_center.y - 1),
                    self.DISTANCE_TO_SWITCH_SOURCE_SINK,
                    Color.GRAY,
                )
            for obj in self.obstacles_debug:
                self.agent.map_tools.draw_circle(obj[0].position, obj[1], Color.TEAL)
                self.agent.map_tools.draw_text(obj[0].position, "o", Color.TEAL)
        scout_pos = py_unit.position
        if self.show_unit_p:
            light_red = Color(255, 204, 203)
            for up in self.unit_potentials:
                self.agent.map_tools.draw_text(up + scout_pos, "u", light_red)
                self.agent.map_tools.draw_line(scout_pos, up + scout_pos, light_red)
                self.agent.map_tools.draw_circle(up + scout_pos, 2, light_red)

        if self.show_region_p:
            for rp in self.region_potentials:
                self.agent.map_tools.draw_line(scout_pos, rp + scout_pos, Color.YELLOW)
                self.agent.map_tools.draw_circle(rp + scout_pos, 2, Color.YELLOW)

        if self.show_border_p:
            for bp in self.border_potentials:
                self.agent.map_tools.draw_line(
                    scout_pos, bp + scout_pos, Color(255, 165, 0)
                )  # orange
                self.agent.map_tools.draw_circle(bp + scout_pos, 2, Color(255, 165, 0))
                self.agent.map_tools.draw_text(bp + scout_pos, "b", Color.WHITE)

        if self.show_all_p:
            for p in self.all_potentials:
                self.agent.map_tools.draw_line(scout_pos, p + scout_pos, Color.BLACK)
                self.agent.map_tools.draw_circle(p + scout_pos, 2, Color.BLACK)

    def near_reach_pos(self, pos1, pos2, dist=1):
        return pos1.distance(pos2) < dist

    def move(self, py_unit: PyUnit):
        """Code for controlling scout's movement using potential flow"""
        
        scout_pos = py_unit.unit.position

        this_target = py_unit.position

        # reverse all flows values if needed
        should_reverse = self.forward
        self.reverse_flows(should_reverse)
        speed = calculate_pval(self, py_unit)
        # reverse back
        self.reverse_flows(should_reverse)

        # Find valid position
        self.forward = 1
        self.scout_target = scout_pos + speed
        ratio = 1 / speed.length()
        seg = speed * ratio
        this_target = (seg * 3) + this_target
        while not self.agent.map_tools.is_walkable(Point2DI(this_target)):
            this_target = seg + this_target

            if self.agent.map_tools.is_valid_position(this_target):
                break

        py_unit.move(this_target)

    def register_enemy_positions(self, enemies: set[Union[Unit, PyUnit]]):
        for enemy in enemies:
            self.enemies.add(enemy)

    def get_enemies(self):
        return self.enemies

    def in_danger(self, py_unit: PyUnit) -> bool:
        """Check if unit is in danger"""
        if get_enemies_in_neighbouring_tiles(self.agent, py_unit.tile_position, dist=6):
            self.USE_ATTRACT_PVAL = False
            return True
        if danger := any(
            enemy.get_target() == py_unit
            for enemy in get_enemies_in_neighbouring_tiles(
                self.agent,
                py_unit.tile_position,
                dist=py_unit.unit_type.sight_range + 2,
            )
        ):
            self.USE_ATTRACT_PVAL = False
            return danger

    def add_attract_point(self, attract_point):
        self.attract_points.add(attract_point)

    def set_scout_target(self, scout_target):
        self.scout_target = scout_target

    def fade_enemies(self):
        enemies_to_remove = {
            enemy
            for enemy in self.enemies
            if self.agent.current_frame - enemy.last_seen > enemy.fade_time
        }
        self.enemies -= enemies_to_remove

    def scout_enemy_opening(self, py_unit):
        """Scout enemy opening"""
        enemy_start_pos, enemy_start_region = get_enemy_info(self.agent)

        waypoint = get_closest(
            self.agent.region_manager.chokepoints_as_centers, enemy_start_pos
        )

        closest_choke_pos = get_closest_choke_pos(self.agent, py_unit)
        cur_region = self.agent.region_manager.get_region(py_unit.tile_position)

        enemy_expansion_pos, enemy_expansion_region = get_enemy_expansion_info(
            self.agent
        )

        if not (
            closest_choke_pos
            and enemy_expansion_pos
            and self.is_far_enough_from_choke(py_unit, closest_choke_pos)
        ):
            self.target_region, next_move_pos = (
                (enemy_expansion_region, enemy_expansion_pos)
                if self.should_switch_to_expansion
                else (enemy_start_region, enemy_start_pos)
            )
            py_unit.move(next_move_pos)
            if DEBUG_SCOUT:
                self.tt = next_move_pos
            return
        if cur_region == enemy_start_region:
            if self.check_if_should_switch_region_to_expansion():
                self.switch_region(enemy_expansion_region, waypoint)
            else:
                if (
                    self.should_switch_to_expansion
                    and cur_region == enemy_expansion_region
                ) or not self.should_switch_to_expansion:
                    self.update_switch_status(waypoint)
                self.potential_flow(py_unit)
        elif cur_region == enemy_expansion_region:
            if (
                self.should_switch_to_expansion
                and self.agent.current_frame
                > self.frame_since_switch + SECOND_IN_FRAMES * 10
                and self.switch_complete
            ):
                self.after_change(enemy_start_region, waypoint)
            else:
                if self.should_switch_to_expansion:
                    self.update_switch_status(waypoint)
                self.potential_flow(py_unit)
        else:
            next_pos = (
                enemy_expansion_pos
                if self.should_switch_to_expansion
                else enemy_start_pos
            )
            self.normal_scout(py_unit, enemy_start_region, waypoint, next_pos)

        if DEBUG_SCOUT:
            self.agent.map_tools.draw_circle(closest_choke_pos, 0.25, Color.PURPLE)
            if enemy_expansion_pos:
                self.agent.map_tools.draw_circle(
                    enemy_expansion_pos, 5, Color(255, 165, 0)
                )
                self.agent.map_tools.draw_text(enemy_expansion_pos, "next", Color.WHITE)

    def reverse_flows(self, go):
        """Reverse flows values of interest if needed"""
        self.CENTER_VORTEX *= go
        self.BORDER_VORTEX *= go
        self.BUILDING_OBSTACLE *= go

    @cache
    def get_next_target_old(self, py_unit: PyUnit) -> Point2D:
        """Get next target position"""
        closest_enemy_base = get_closest(
            self.agent.non_start_bases,
            get_enemy_base_location(self.agent),
            lambda base: base.position,
        )
        if not self.use_old_next_target_method:
            if closest_enemy_base.contains_position(py_unit.position):
                if any(
                    unit.unit_type.is_resource_depot
                    for unit in get_enemies_in_base_location(
                        self.agent, closest_enemy_base
                    )
                ):
                    return closest_enemy_base.position
                else:
                    return None
        return closest_enemy_base.position

    @cache
    def validate_expansion_target(
        self, next_target_pos: Point2D, switch: callable = None
    ) -> bool:
        """Validate expansion target"""
        target_base = get_closest(
            self.agent.non_start_bases, next_target_pos, lambda base: base.position
        )
        if self.agent.map_tools.is_visible(
            int(next_target_pos.x), int(next_target_pos.y)
        ) and target_base not in self.agent.base_location_manager.get_occupied_base_locations(
            PLAYER_ENEMY
        ):
            return False
        return True

    def check_if_should_switch_region_to_expansion(self) -> bool:
        """Should switch region"""
        return (
            (not self.should_switch_to_expansion)
            and self.agent.current_frame
            > self.frame_since_switch + SECOND_IN_FRAMES * 40
            and self.switch_complete
        )

    def should_switch_region(self) -> bool:
        return (
            self.agent.current_frame > self.frame_since_switch + SECOND_IN_FRAMES * 20
            and self.switch_complete
        )

    def is_far_enough_from_choke(
        self, py_unit: PyUnit, closest_choke_pos: Point2DI
    ) -> bool:
        return py_unit.position.square_distance(closest_choke_pos) > CHOKE_DIST

    def normal_scout(self, py_unit, enemy_start_region, waypoint, pos):
        self.target_region = enemy_start_region
        self.add_attract_point(waypoint)
        if self.in_danger(py_unit):
            self.potential_flow(py_unit)
            return
        py_unit.move(pos)
        self.tt = Point2DI(pos)

    def update_switch_status(self, waypoint):
        if self.switch_complete:
            return
        self.attract_points.remove(waypoint)
        self.switch_complete = True
        self.frame_since_switch = self.agent.current_frame

    def after_change(self, enemy_start_region, waypoint):
        self.should_switch_to_expansion = False
        self.switch_complete = False
        self.target_region = enemy_start_region
        self.add_attract_point(waypoint)

    def switch_region(self, region: Region, waypoint):
        self.should_switch_to_expansion = True
        self.switch_complete = False
        self.target_region = region
        self.add_attract_point(waypoint)  # is center
