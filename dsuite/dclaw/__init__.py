# Copyright 2019 The D'Suite Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gym environment registration for DClaw environments."""

from dsuite.utils.registration import register

#===============================================================================
# Pose tasks
#===============================================================================

# Default number of steps per episode.
_POSE_EPISODE_LEN = 40  # 40*40*2.5ms = 4s

register(
    env_id='DClawPoseStatic-v0',
    class_path='dsuite.dclaw.pose:DClawPoseStatic',
    max_episode_steps=_POSE_EPISODE_LEN)

register(
    env_id='DClawPoseDynamic-v0',
    class_path='dsuite.dclaw.pose:DClawPoseDynamic',
    max_episode_steps=_POSE_EPISODE_LEN)

#===============================================================================
# Turn tasks
#===============================================================================

# Default number of steps per episode.
_TURN_EPISODE_LEN = 40  # 40*40*2.5ms = 4s

register(
    env_id='DClawTurnFixed-v0',
    class_path='dsuite.dclaw.turn:DClawTurnFixed')

register(
    env_id='DClawTurnRandom-v0',
    class_path='dsuite.dclaw.turn:DClawTurnRandom')

register(
    env_id='DClawTurnRandomDynamics-v0',
    class_path='dsuite.dclaw.turn:DClawTurnRandomDynamics')

register(
    env_id='DClawTurnImage-v0',
    class_path='dsuite.dclaw.turn:DClawTurnImage')

register(
    env_id='DClawTurnImageResetFree-v0',
    class_path='dsuite.dclaw.turn:DClawTurnImageResetFree')

register(
    env_id='DClawTurnMultiGoal-v0',
    class_path='dsuite.dclaw.turn:DClawTurnMultiGoal')

register(
    env_id='DClawTurnMultiGoalResetFree-v0',
    class_path='dsuite.dclaw.turn:DClawTurnMultiGoalResetFree')

register(
    env_id='DClawTurnRandomResetSingleGoal-v0',
    class_path='dsuite.dclaw.turn:DClawTurnRandomResetSingleGoal')

#===============================================================================
# Turn Free Object tasks
#===============================================================================

register(
    env_id='DClawTurnFreeValve3Fixed-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3Fixed')

register(
    env_id='DClawTurnFreeValve3Hardware-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3Hardware')

register(
    env_id='DClawTurnFreeValve3RandomReset-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3RandomReset')

register(
    env_id='DClawTurnFreeValve3ResetFree-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3ResetFree')

register(
    env_id='DClawTurnFreeValve3ResetFreeSwapGoal-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3ResetFreeSwapGoal')

register(
    env_id='DClawTurnFreeValve3ResetFreeSwapGoalEval-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3ResetFreeSwapGoalEval')

register(
    env_id='DClawTurnFreeValve3ResetFreeComposedGoals-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3ResetFreeComposedGoals')

register(
    env_id='DClawTurnFreeValve3FixedResetSwapGoal-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3FixedResetSwapGoal')

register(
    env_id='DClawTurnFreeValve3FixedResetSwapGoalEval-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3FixedResetSwapGoalEval')

register(
    env_id='DClawTurnFreeValve3ResetFreeCurriculum-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3ResetFreeCurriculum')

register(
    env_id='DClawTurnFreeValve3ResetFreeCurriculumEval-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3ResetFreeCurriculumEval')

register(
    env_id='DClawTurnFreeValve3Image-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3Image')

register(
    env_id='DClawTurnFreeValve3MultiGoal-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3MultiGoal')

register(
    env_id='DClawTurnFreeValve3MultiGoalResetFree-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTurnFreeValve3MultiGoalResetFree')

#===============================================================================
# Turn Free Object tasks
#===============================================================================

register(
    env_id='DClawTranslatePuckFixed-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTranslatePuckFixed')

register(
    env_id='DClawTranslatePuckResetFree-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTranslatePuckResetFree')

register(
    env_id='DClawTranslatePuckResetFreeSwapGoalEval-v0',
    class_path='dsuite.dclaw.turn_free_object:DClawTranslatePuckResetFreeSwapGoalEval')


#===============================================================================
# Lift Object tasks
#===============================================================================

register(
    env_id='DClawLiftDDFixed-v0',
    class_path='dsuite.dclaw.lift:DClawLiftDDFixed')

register(
    env_id='DClawLiftDDHardware-v0',
    class_path='dsuite.dclaw.lift:DClawLiftDDHardware')

register(
    env_id='DClawLiftDDResetFree-v0',
    class_path='dsuite.dclaw.lift:DClawLiftDDResetFree')

register(
    env_id='DClawLiftDDResetFreeComposedGoals-v0',
    class_path='dsuite.dclaw.lift:DClawLiftDDResetFreeComposedGoals')


#===============================================================================
# Flip Object tasks
#===============================================================================

register(
    env_id='DClawFlipEraserFixed-v0',
    class_path='dsuite.dclaw.flip:DClawFlipEraserFixed')

register(
    env_id='DClawFlipEraserResetFree-v0',
    class_path='dsuite.dclaw.flip:DClawFlipEraserResetFree')

register(
    env_id='DClawFlipEraserResetFreeSwapGoal-v0',
    class_path='dsuite.dclaw.flip:DClawFlipEraserResetFreeSwapGoal')

register(
    env_id='DClawFlipEraserResetFreeSwapGoalEval-v0',
    class_path='dsuite.dclaw.flip:DClawFlipEraserResetFreeSwapGoalEval')

register(
    env_id='DClawFlipEraserResetFreeComposedGoals-v0',
    class_path='dsuite.dclaw.flip:DClawFlipEraserResetFreeComposedGoals')

#===============================================================================
# Screw tasks
#===============================================================================

# Default number of steps per episode.
_SCREW_EPISODE_LEN = 80  # 80*40*2.5ms = 8s

register(
    env_id='DClawScrewFixed-v0',
    class_path='dsuite.dclaw.screw:DClawScrewFixed',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandom-v0',
    class_path='dsuite.dclaw.screw:DClawScrewRandom',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandomDynamics-v0',
    class_path='dsuite.dclaw.screw:DClawScrewRandomDynamics',
    max_episode_steps=_SCREW_EPISODE_LEN)

#===============================================================================
# Slide Object tasks
#===============================================================================

register(
    env_id='DClawSlideBeadsFixed-v0',
    class_path='dsuite.dclaw.slide:DClawSlideBeadsFixed')

register(
    env_id='DClawSlideBeadsResetFree-v0',
    class_path='dsuite.dclaw.slide:DClawSlideBeadsResetFree')

register(
    env_id='DClawSlideBeadsResetFreeEval-v0',
    class_path='dsuite.dclaw.slide:DClawSlideBeadsResetFreeEval')
