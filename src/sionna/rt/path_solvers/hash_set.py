#
# SPDX-FileCopyrightText: Copyright (c) 2026 The Sionna contributors.
# SPDX-License-Identifier: Apache-2.0
#
"""Bounded, collision-resolving insertion of 64-bit candidate hashes."""

import drjit as dr
import mitsuba as mi


@dr.syntax
def insert_hash(table: mi.UInt64, zero_seen: mi.UInt,
                overflow: mi.UInt, key: mi.UInt64,
                source: mi.UInt, table_size: int,
                active: mi.Bool, max_probes: int = 128) -> mi.Bool:
    """Insert one key per active lane into a per-source open-addressed set.

    Return true only for the lane that inserted a new key. Equal *full* keys
    are duplicates; different keys with the same initial slot are probed.
    Zero is the empty-slot marker, so key zero uses a separate per-source CAS.
    A bounded probe prevents unbounded work on a full or clustered table. On
    exhaustion, ``overflow[0]`` is set and the caller must reject the result.

    As with ``scatter_cas``, consume the returned mask in the same evaluation
    as the table updates, or materialize all dependent outputs together.
    """
    is_zero = active & (key == 0)
    _, inserted_zero = dr.scatter_cas(
        zero_seen, mi.UInt(0), mi.UInt(1), source, is_zero
    )
    inserted = inserted_zero & is_zero
    pending = active & ~is_zero
    slot = mi.UInt(key % table_size)
    offset = source * table_size
    probe = mi.UInt(0)
    limit = min(max_probes, table_size)

    while dr.hint(pending & (probe < limit),
                 exclude=[table, zero_seen, overflow]):
        previous, claimed = dr.scatter_cas(
            table, mi.UInt64(0), key, offset + slot, pending
        )
        claimed &= pending
        inserted |= claimed
        pending &= ~claimed & (previous != key)
        slot = dr.select(slot + 1 == table_size, mi.UInt(0), slot + 1)
        probe += 1

    dr.scatter_reduce(dr.ReduceOp.Max, overflow, mi.UInt(1), mi.UInt(0),
                      pending)
    return inserted
