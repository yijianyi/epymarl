"""Identity-preserving native TABX -> fixed QMIX entity slots, and inverse maps."""
from dataclasses import dataclass

import numpy as np

LAYOUT_VERSION = "tabx-entity-slots-v2"
SIMULATOR_CONTRACT = "official-tabx-native-v1"
MAX_ALLY, MAX_ENEMY, MAX_ZONE = 10, 10, 10
OWN_FEATURES, OTHER_FEATURES, UNIT_FEATURES, ZONE_FEATURES = 14, 16, 14, 6
OBS_SIZE, STATE_SIZE = 378, 340
OBS_ZONE_OFFSET, STATE_ZONE_OFFSET = 318, 280


@dataclass(frozen=True)
class TABXLayout:
    n_ally: int
    n_enemy: int
    n_zone: int

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (self.n_ally, self.n_enemy, self.n_zone)):
            raise ValueError("Native sizes must be integers")
        if not (1 <= self.n_ally <= MAX_ALLY and 1 <= self.n_enemy <= MAX_ENEMY and 0 <= self.n_zone <= MAX_ZONE):
            raise ValueError("Unsupported native TABX sizes")

    @property
    def n_units(self):
        return self.n_ally + self.n_enemy

    @property
    def native_obs_size(self):
        return OWN_FEATURES + OTHER_FEATURES * (self.n_units - 1) + ZONE_FEATURES * self.n_zone

    @property
    def native_state_size(self):
        return UNIT_FEATURES * self.n_units + ZONE_FEATURES * self.n_zone

    @property
    def unit_slots(self):
        return np.r_[np.arange(self.n_ally), MAX_ALLY + np.arange(self.n_enemy)]

    @property
    def present(self):
        present = np.zeros(MAX_ALLY + MAX_ENEMY, dtype=np.uint8)
        present[self.unit_slots] = 1
        return present

    def metadata(self):
        return dict(layout_version=LAYOUT_VERSION, simulator_contract=SIMULATOR_CONTRACT,
                    native_shape=[self.n_ally, self.n_enemy, self.n_zone],
                    n_ally=self.n_ally, n_enemy=self.n_enemy, n_zone=self.n_zone,
                    unit_slots=self.unit_slots.tolist(), native_obs_size=self.native_obs_size,
                    native_state_size=self.native_state_size)

    def other_slots(self, observer):
        if not 0 <= observer < self.n_units:
            raise ValueError("Observer is not a real native unit")
        # Native others: (observer+1) % N, ..., (observer+N-1) % N.
        entities = (observer + np.arange(1, self.n_units)) % self.n_units
        observer_slot = self.unit_slots[observer]
        return (self.unit_slots[entities] - observer_slot - 1) % (MAX_ALLY + MAX_ENEMY)

    def pack_local(self, native, observer):
        native = np.asarray(native, dtype=np.float32)
        if native.shape != (self.native_obs_size,) or not np.isfinite(native).all():
            raise ValueError("Invalid native local observation")
        fixed = np.zeros(OBS_SIZE, dtype=np.float32)
        fixed[:OWN_FEATURES] = native[:OWN_FEATURES]
        native_zone = OWN_FEATURES + OTHER_FEATURES * (self.n_units - 1)
        fixed[OWN_FEATURES:OBS_ZONE_OFFSET].reshape(19, OTHER_FEATURES)[self.other_slots(observer)] = native[OWN_FEATURES:native_zone].reshape(-1, OTHER_FEATURES)
        fixed[OBS_ZONE_OFFSET:OBS_ZONE_OFFSET + self.n_zone * ZONE_FEATURES] = native[native_zone:]
        return fixed

    def unpack_local(self, fixed, observer):
        fixed = np.asarray(fixed)
        if fixed.shape != (OBS_SIZE,):
            raise ValueError("Invalid fixed local observation")
        others = fixed[OWN_FEATURES:OBS_ZONE_OFFSET].reshape(19, OTHER_FEATURES)[self.other_slots(observer)]
        return np.concatenate((fixed[:OWN_FEATURES], others.reshape(-1),
                               fixed[OBS_ZONE_OFFSET:OBS_ZONE_OFFSET + self.n_zone * ZONE_FEATURES]))

    def pack_observations(self, native):
        native = np.asarray(native)
        if native.shape != (self.n_ally, self.native_obs_size):
            raise ValueError("Invalid allied observation matrix")
        fixed = np.zeros((MAX_ALLY, OBS_SIZE), dtype=np.float32)
        for observer in range(self.n_ally):
            fixed[observer] = self.pack_local(native[observer], observer)
        return fixed

    def pack_state(self, native):
        native = np.asarray(native, dtype=np.float32).reshape(-1)
        if native.shape != (self.native_state_size,) or not np.isfinite(native).all():
            raise ValueError("Invalid native global state")
        fixed = np.zeros(STATE_SIZE, dtype=np.float32)
        end = self.n_units * UNIT_FEATURES
        fixed[:STATE_ZONE_OFFSET].reshape(20, UNIT_FEATURES)[self.unit_slots] = native[:end].reshape(self.n_units, UNIT_FEATURES)
        fixed[STATE_ZONE_OFFSET:STATE_ZONE_OFFSET + self.n_zone * ZONE_FEATURES] = native[end:]
        return fixed

    def unpack_state(self, fixed):
        fixed = np.asarray(fixed)
        if fixed.shape != (STATE_SIZE,):
            raise ValueError("Invalid fixed global state")
        return np.concatenate((fixed[:STATE_ZONE_OFFSET].reshape(20, UNIT_FEATURES)[self.unit_slots].reshape(-1),
                               fixed[STATE_ZONE_OFFSET:STATE_ZONE_OFFSET + self.n_zone * ZONE_FEATURES]))

    def pack_alive(self, native_alive):
        native_alive = np.asarray(native_alive, dtype=np.uint8)
        if native_alive.shape != (self.n_units,):
            raise ValueError("Invalid native alive mask")
        fixed = np.zeros(MAX_ALLY + MAX_ENEMY, dtype=np.uint8)
        fixed[self.unit_slots] = native_alive
        return fixed

    def pack_avail_actions(self, native, alive):
        native, alive = np.asarray(native), np.asarray(alive, dtype=bool)
        if native.shape != (self.n_ally, 8) or alive.shape != (self.n_ally,):
            raise ValueError("Invalid native allied action mask")
        if not np.isin(native, (0, 1)).all():
            raise ValueError("Nonbinary native action mask")
        fixed = np.zeros((MAX_ALLY, 8), dtype=np.int32)
        fixed[:, 7] = 1
        fixed[:self.n_ally][alive] = native[alive]
        if not (fixed.sum(axis=1) > 0).all():
            raise ValueError("Empty legal action set")
        return fixed

    def native_actions(self, fixed):
        fixed = np.asarray(fixed)
        if fixed.shape != (MAX_ALLY,):
            raise ValueError("Expected exactly 10 allied actions")
        return fixed[:self.n_ally].copy()
