from __future__ import annotations
import math
from typing import TYPE_CHECKING
from library import Point2D
from modules.potential_flow.vector import Vector

SCALE = 2.5

from modules.py_unit import PyUnit

if TYPE_CHECKING:
    from tasks.pf_scout import PFscout


def enemy_pf(ENEMY_NEEDLE, scout_pos: Point2D, enemy: PyUnit, enemy_target, attack_range):
    """Enemy potential flow
        E(z) = p₅e^(-iα)N(z)
               ps
    """
    return ENEMY_NEEDLE * needle_pval(enemy.position, scout_pos, enemy_target.position,
                                      attack_range * 1.0 / enemy.radius) * SCALE


def needle_pval(source, point, target, bias):
    """Like source/sink potential but the shape change to a direction"""
    if target is None:
        return Vector()
    x = point.x - source.x
    y = point.y - source.y
    r2 = 1.0 * x * x + 1.0 * y * y

    V = Vector(target.x - source.x, target.y - source.y)
    v = Vector(x, y)
    r = Vector(x / r2, y / r2)

    if not V:
        return Vector()

    cosn = 1 / bias
    cost = V.cos(v)

    if cost >= cosn:
        sinn = math.sqrt(1 - cosn * cosn)
        sint = V.sin(v)
        t = 1
        if sint >= 0:
            r *= 1 / (cosn * cost + sinn * sint)
        else:
            r *= 1 / (cosn * cost - sinn * sint)

    return r


def vortex_potential(source, point):
    """
    V(z) = ilog(z-z_start)
    s=curReg_center, p=enemy_position
    """
    x = point.x - source.x  # x - x_start
    y = point.y - source.y  # y - y_start
    r2 = x * x + y * y  # (x-x_start)^2 + (y-y_start)^2
    return Vector(y / r2, -x / r2)  # u, v


def source_potential(source, point):
    """S(z) = log(z-z_s)"""
    x = point.x - source.x
    y = point.y - source.y
    r2 = 1.0 * x * x + y * y
    return Vector(x / r2, y / r2)


def region_pf(region_center, pos, center, scout: PFscout,
              vortex_correction, source_correction, d_r_thres):
    """
    R(z) = p₁V(z) + p₂S(z) if ||z'|| > dᵣ_ₜₕᵣₑₛ
    p₁V(z) - p₂S(z) otherwise
    """
    p1Vz = vortex_potential(region_center, pos) * scout.CENTER_VORTEX * vortex_correction
    p2Sz = source_potential(region_center, pos) * scout.CENTER_SOURCE_SINK * source_correction
    return p1Vz + p2Sz if center < d_r_thres else p1Vz - p2Sz


def obstacle_potential(scout: PFscout, obs_pos, pos, center, a2):
    """Obstacle potential flow
    O(z) = p₁Oᵥ(z) + p₂Oₛ(z) if ||z꜀'|| > dᵣ_ₜₕᵣₑₛ
    p₁Oᵥ(z) - p₂Oₛ(z) otherwise
    """

    Ov = obstacle_vortex_potential(obs_pos, pos, center, a2)
    Os = obstacle_source_potential(obs_pos, pos, center, a2)

    if center.square_distance(pos) < scout.DISTANCE_TO_SWITCH_SOURCE_SINK:
        return Ov * scout.CENTER_VORTEX * 1.2 + Os * scout.CENTER_SOURCE_SINK * 1.2
    else:
        return Ov * scout.CENTER_VORTEX * 1.2 - Os * scout.CENTER_SOURCE_SINK * 1.2


def obstacle_vortex_potential(obs_pos, pos, center, a2):
    """Circle theorem obstacle by a vortex O(z) = -ilog(a^2/(z-Z)-conj(z_c-Z))
    s : obstacle position
    p : considered position
    c: center vortex position
    """

    x = pos.x - obs_pos.x
    y = pos.y - obs_pos.y
    xc = center.x - obs_pos.x
    yc = center.y - obs_pos.y
    x2 = 1.0 * x * x
    y2 = 1.0 * y * y
    r2 = x2 + y2
    deno = r2 * (a2 * a2 - 2 * a2 * (x * xc + y * yc) + r2 * (xc * xc + yc * yc))
    vx = a2 * (a2 * y - 2 * x * y * xc - y2 * yc + x2 * yc) / deno
    vy = -a2 * (a2 * x - 2 * x * y * yc - x2 * xc + y2 * xc) / deno
    return Vector(vx, vy)


def obstacle_source_potential(scout, p, c, a2):
    """O(z) = log(a^2/(z-Z)-conj(z_c-Z))
    s : obstacle position
    p : considered position
    c: center vortex position
    """
    
    x = p.x - scout.x
    y = p.y - scout.y
    xc = c.x - scout.x
    yc = c.y - scout.y
    x2 = 1.0 * x * x
    y2 = 1.0 * y * y
    r2 = x2 + y2

    deno = r2 * (a2 * a2 - 2 * a2 * (x * xc + y * yc) + r2 * (xc * xc + yc * yc))

    vx = -a2 * (a2 * x - 2 * x * y * yc - x2 * xc + y2 * xc) / deno
    vy = -a2 * (a2 * y - 2 * x * y * xc - y2 * yc + x2 * yc) / deno

    return Vector(vx, vy)
