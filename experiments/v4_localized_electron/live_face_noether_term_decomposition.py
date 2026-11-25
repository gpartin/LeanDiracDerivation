"""Decompose native face-cut phase flux by action term."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import live_clifford_bond_action as action
import live_face_noether_continuity_gate as noether


HERE = Path(__file__).resolve().parent
SIDE = 5


def gradients(interval, left, right):
    return torch.func.grad(interval, argnums=(0, 1))(
        left, right, native_time.DT)


def variation(interval, left, right, field, mask):
    gl, gr = gradients(interval, left, right)
    return (torch.dot(gl, noether.generator(left, field, mask))
            + torch.dot(gr, noether.generator(right, field, mask)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", default="live_face_noether_trajectory.npz")
    parser.add_argument("--negative", default="live_face_noether_trajectory_negative.npz")
    parser.add_argument("--output", default="live_face_noether_term_decomposition.json")
    args = parser.parse_args()
    archive = np.load(HERE/args.positive)
    negative = np.load(HERE/args.negative)
    states = [torch.tensor(value, dtype=torch.float64)
              for value in archive["states"]]
    negative_states = [torch.tensor(value, dtype=torch.float64)
                       for value in negative["states"]]
    rest = archive["rest"]
    rotation = archive["rotation"]
    rest_states = [torch.tensor(
        np.linalg.matrix_power(rotation, index-1)@rest,
        dtype=torch.float64) for index in range(len(states))]
    coordinates = torch.arange(SIDE)
    masks = {}
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = SIDE
        masks[("x", "y", "z")[axis]] = (
            coordinates.reshape(shape) <= SIDE//2).to(torch.float64).expand(
                SIDE, SIDE, SIDE)

    rows = []
    intervals = {
        "full": action.interval,
        "inherited": action.incumbent.interval,
    }
    for field in ("Psi", "common"):
        for axis, mask in masks.items():
            samples = {name: [] for name in ("full", "inherited", "face")}
            quadratic_samples = {
                name: [] for name in ("full", "inherited", "face")}
            for index in range(1, len(states)):
                left, right = states[index-1], states[index]
                nl, nr = negative_states[index-1], negative_states[index]
                rl, rr = rest_states[index-1], rest_states[index]
                values = {}
                quadratic = {}
                for name, interval in intervals.items():
                    positive_value = variation(
                        interval, left, right, field, mask)
                    negative_value = variation(
                        interval, nl, nr, field, mask)
                    rest_value = variation(interval, rl, rr, field, mask)
                    values[name] = float(positive_value-rest_value)
                    quadratic[name] = float(
                        (positive_value+negative_value-2*rest_value)/2)
                    samples[name].append(values[name])
                samples["face"].append(values["full"]-values["inherited"])
                quadratic_samples["full"].append(quadratic["full"])
                quadratic_samples["inherited"].append(quadratic["inherited"])
                quadratic_samples["face"].append(
                    quadratic["full"]-quadratic["inherited"])
            rows.append({
                "field_phase": field,
                "cut_axis": axis,
                "mean_flux": {name: float(np.mean(value))
                              for name, value in samples.items()},
                "mean_quadratic_packet_flux": {
                    name: float(np.mean(value))
                    for name, value in quadratic_samples.items()},
                "samples": samples,
                "quadratic_packet_samples": quadratic_samples,
            })
    result = {
        "status": "V4_LIVE_FACE_NOETHER_TERM_DECOMPOSITION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "link_source": action.LINK_SOURCE,
        "action_scale": action.SCALE,
        "launch": "plus copy, negative x packet",
        "rows": rows,
        "verdict": "DIAGNOSTIC_ONLY",
        "claim_boundary": (
            "Action-derived face-cut flux decomposition on the short "
            "nonlinear trajectory; it does not itself establish equality "
            "with Dirac current or a full cell translation."),
    }
    path = HERE/args.output
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
