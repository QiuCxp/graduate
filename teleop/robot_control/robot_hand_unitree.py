# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_  # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, Array, Value, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(
        self,
        left_hand_array_in,
        right_hand_array_in,
        left_trigger_value_in,
        right_trigger_value_in,
        dual_hand_data_lock=None,
        dual_hand_state_array_out=None,
        dual_hand_action_array_out=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        """
        left_hand_array_in / right_hand_array_in:
            25x3 hand skeleton positions from XR (multiprocessing.Array length=75)

        left_trigger_value_in / right_trigger_value_in:
            trigger value from controller (multiprocessing.Value float), expected [0,1]

        dual_hand_state_array_out / dual_hand_action_array_out:
            optional shared outputs: 14 dims (7 left + 7 right)
        """
        logger_mp.info("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        else:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test)

        # DDS pub/sub
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array("d", Dex3_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Dex3_Num_Motors, lock=True)

        # subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # wait initial state
        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        # control thread (avoid DDS issues across processes)
        hand_control_thread = threading.Thread(
            target=self.control_process,
            args=(
                left_hand_array_in,
                right_hand_array_in,
                left_trigger_value_in,
                right_trigger_value_in,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array_out,
                dual_hand_action_array_out,
            ),
            daemon=True,
        )
        hand_control_thread.start()

        logger_mp.info("Initialize Dex3_1_Controller OK!")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                for idx, jid in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[jid].q
                for idx, jid in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[jid].q
            time.sleep(0.002)

    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F
            self.status = status & 0x07
            self.timeout = timeout & 0x01

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        for idx, jid in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[jid].q = float(left_q_target[idx])
        for idx, jid in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[jid].q = float(right_q_target[idx])

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)

    def control_process(
        self,
        left_hand_array_in,
        right_hand_array_in,
        left_trigger_value_in,
        right_trigger_value_in,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array_out=None,
        dual_hand_action_array_out=None,
    ):
        self.running = True

        # Ensure DDS is initialized in this subprocess
        try:
            ChannelFactoryInitialize(1 if self.simulation_mode else 0)
        except Exception as e:
            logger_mp.warning(f"[Dex3_1_Controller] DDS init in subprocess failed: {e}")

        left_q_target = np.full(Dex3_Num_Motors, 0.0)
        right_q_target = np.full(Dex3_Num_Motors, 0.0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # init cmd msg
        self.left_msg = unitree_hg_msg_dds__HandCmd_()
        for jid in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id=jid, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[jid].mode = motor_mode
            self.left_msg.motor_cmd[jid].q = q
            self.left_msg.motor_cmd[jid].dq = dq
            self.left_msg.motor_cmd[jid].tau = tau
            self.left_msg.motor_cmd[jid].kp = kp
            self.left_msg.motor_cmd[jid].kd = kd

        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for jid in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id=jid, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[jid].mode = motor_mode
            self.right_msg.motor_cmd[jid].q = q
            self.right_msg.motor_cmd[jid].dq = dq
            self.right_msg.motor_cmd[jid].tau = tau
            self.right_msg.motor_cmd[jid].kp = kp
            self.right_msg.motor_cmd[jid].kd = kd
        
        # [Fix] Re-initialize DDS publishers in the subprocess to ensure valid handles
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        # close poses (same as your OK version)
        # Left order after re-permute: T0,T1,T2,I0,I1,M0,M1
        left_close_pose = np.array([0.0, 0.8, 1.5, -1.5, -1.5, -1.5, -1.5])
        # Right enum order: T0,T1,T2,I0,I1,M0,M1
        right_close_pose = np.array([0.0, -0.8, -1.5, 1.5, 1.5, 1.5, 1.5])

        try:
            while self.running:
                start_time = time.time()

                # Reset targets to open pose (0.0) each frame
                left_q_target = np.zeros(Dex3_Num_Motors)
                right_q_target = np.zeros(Dex3_Num_Motors)

                # read hand skeleton
                with left_hand_array_in.get_lock():
                    left_hand_data = np.array(left_hand_array_in[:]).reshape(25, 3).copy()
                with right_hand_array_in.get_lock():
                    right_hand_data = np.array(right_hand_array_in[:]).reshape(25, 3).copy()

                # read state
                state_data = np.concatenate(
                    (np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:]))
                )

                # === retarget only if hand data is valid/initialized ===
                if (
                    not np.all(right_hand_data == 0.0)
                    and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15]))
                ):
                    ref_left_value = (
                        left_hand_data[self.hand_retargeting.left_indices[1, :]]
                        - left_hand_data[self.hand_retargeting.left_indices[0, :]]
                    )
                    ref_right_value = (
                        right_hand_data[self.hand_retargeting.right_indices[1, :]]
                        - right_hand_data[self.hand_retargeting.right_indices[0, :]]
                    )

                    left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
                        self.hand_retargeting.left_dex_retargeting_to_hardware
                    ]
                    # Permute Left Hand Retargeting (T,T,T,M,M,I,I) -> (T,T,T,I,I,M,M)
                    left_q_target = left_q_target[[0, 1, 2, 5, 6, 3, 4]]

                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
                        self.hand_retargeting.right_dex_retargeting_to_hardware
                    ]

                # === Trigger Override Logic (the key you want) ===
                left_trigger = float(left_trigger_value_in.value)
                right_trigger = float(right_trigger_value_in.value)

                if left_trigger > 0.001:
                    left_q_target = (1.0 - left_trigger) * left_q_target + left_trigger * left_close_pose
                if right_trigger > 0.001:
                    right_q_target = (1.0 - right_trigger) * right_q_target + right_trigger * right_close_pose

                # output
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array_out is not None and dual_hand_action_array_out is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_action_array_out[:] = action_data

                # publish
                self.ctrl_dual_hand(left_q_target, right_q_target)

                # rate control
                time_elapsed = time.time() - start_time
                time.sleep(max(0.0, (1.0 / self.fps) - time_elapsed))
        finally:
            logger_mp.info("Dex3_1_Controller has been closed.")


class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandIndex0 = 3
    kLeftHandIndex1 = 4
    kLeftHandMiddle0 = 5
    kLeftHandMiddle1 = 6


class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"


class Dex1_1_Gripper_Controller:
    def __init__(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        dual_gripper_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
        filter=True,
        fps=200.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        logger_mp.info("Initialize Dex1_1_Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.gripper_sub_ready = False
        self.simulation_mode = simulation_mode

        if filter and not self.simulation_mode:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)
        else:
            self.smooth_filter = None

        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()

        self.left_gripper_state_value = Value("d", 0.0, lock=True)
        self.right_gripper_state_value = Value("d", 0.0, lock=True)

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while not self.gripper_sub_ready:
            time.sleep(0.01)
            logger_mp.warning("[Dex1_1_Gripper_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex1_1_Gripper_Controller] Subscribe dds ok.")

        self.gripper_control_thread = threading.Thread(
            target=self.control_thread,
            args=(
                left_gripper_value_in,
                right_gripper_value_in,
                self.left_gripper_state_value,
                self.right_gripper_state_value,
                dual_gripper_data_lock,
                dual_gripper_state_out,
                dual_gripper_action_out,
            ),
        )
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize Dex1_1_Gripper_Controller OK!")

    def _subscribe_gripper_state(self):
        while True:
            left_gripper_msg = self.LeftGripperState_subscriber.Read()
            right_gripper_msg = self.RightGripperState_subscriber.Read()
            self.gripper_sub_ready = True
            if left_gripper_msg is not None and right_gripper_msg is not None:
                self.left_gripper_state_value.value = left_gripper_msg.states[0].q
                self.right_gripper_state_value.value = right_gripper_msg.states[0].q
            time.sleep(0.002)

    def ctrl_dual_gripper(self, dual_gripper_action):
        self.left_gripper_msg.cmds[0].q = float(dual_gripper_action[0])
        self.right_gripper_msg.cmds[0].q = float(dual_gripper_action[1])
        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)

    def control_thread(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        left_gripper_state_value,
        right_gripper_state_value,
        dual_hand_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
    ):
        self.running = True
        DELTA_GRIPPER_CMD = 0.18
        THUMB_INDEX_DISTANCE_MIN = 5.0
        THUMB_INDEX_DISTANCE_MAX = 7.0
        LEFT_MAPPED_MIN = 0.0
        RIGHT_MAPPED_MIN = 0.0
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 5.40
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 5.40
        left_target_action = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05

        self.left_gripper_msg = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp = kp
        self.left_gripper_msg.cmds[0].kd = kd

        self.right_gripper_msg.cmds[0].dq = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp = kp
        self.right_gripper_msg.cmds[0].kd = kd

        try:
            while self.running:
                start_time = time.time()

                with left_gripper_value_in.get_lock():
                    left_gripper_value = float(left_gripper_value_in.value)
                with right_gripper_value_in.get_lock():
                    right_gripper_value = float(right_gripper_value_in.value)

                dual_gripper_state = np.array(
                    [left_gripper_state_value.value, right_gripper_state_value.value]
                )

                if left_gripper_value != 0.0 or right_gripper_value != 0.0:
                    left_target_action = np.interp(
                        left_gripper_value,
                        [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX],
                        [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX],
                    )
                    right_target_action = np.interp(
                        right_gripper_value,
                        [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX],
                        [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX],
                    )

                if not self.simulation_mode:
                    left_actual_action = np.clip(
                        left_target_action,
                        dual_gripper_state[0] - DELTA_GRIPPER_CMD,
                        dual_gripper_state[0] + DELTA_GRIPPER_CMD,
                    )
                    right_actual_action = np.clip(
                        right_target_action,
                        dual_gripper_state[1] - DELTA_GRIPPER_CMD,
                        dual_gripper_state[1] + DELTA_GRIPPER_CMD,
                    )
                else:
                    left_actual_action = left_target_action
                    right_actual_action = right_target_action

                dual_gripper_action = np.array([left_actual_action, right_actual_action])

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out is not None and dual_gripper_action_out is not None:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array(
                            [LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN]
                        )
                        dual_gripper_action_out[:] = dual_gripper_action - np.array(
                            [LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN]
                        )

                self.ctrl_dual_gripper(dual_gripper_action)

                time_elapsed = time.time() - start_time
                time.sleep(max(0.0, (1.0 / self.fps) - time_elapsed))
        finally:
            logger_mp.info("Dex1_1_Gripper_Controller has been closed.")


class Gripper_JointIndex(IntEnum):
    kGripper = 0


if __name__ == "__main__":
    import argparse
    from televuer import TeleVuerWrapper
    from teleimager import ImageClient

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xr-mode",
        type=str,
        choices=["hand", "controller"],
        default="hand",
        help="Select XR device tracking source",
    )
    parser.add_argument(
        "--ee",
        type=str,
        choices=["dex1", "dex3", "inspire1", "brainco"],
        help="Select end effector controller",
    )
    args = parser.parse_args()
    logger_mp.info(f"args:{args}\n")

    ChannelFactoryInitialize(1)  # 0 for real robot, 1 for simulation

    img_client = ImageClient(host="127.0.0.1")  # host='192.168.123.164'
    if not img_client.has_head_cam():
        logger_mp.error("Head camera is required. Please enable head camera on the image server side.")
    head_img_shape = img_client.get_head_shape()
    tv_binocular = img_client.head_is_binocular()

    tv_wrapper = TeleVuerWrapper(
        binocular=tv_binocular,
        use_hand_tracking=args.xr_mode == "hand",
        img_shape=head_img_shape,
        return_hand_rot_data=False,
    )

    # end-effector
    if args.ee == "dex3":
        left_hand_pos_array = Array("d", 75, lock=True)
        right_hand_pos_array = Array("d", 75, lock=True)

        left_trigger_value = Value("d", 0.0, lock=True)
        right_trigger_value = Value("d", 0.0, lock=True)

        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array("d", 14, lock=False)
        dual_hand_action_array = Array("d", 14, lock=False)

        hand_ctrl = Dex3_1_Controller(
            left_hand_pos_array,
            right_hand_pos_array,
            left_trigger_value,
            right_trigger_value,
            dual_hand_data_lock,
            dual_hand_state_array,
            dual_hand_action_array,
        )

    elif args.ee == "dex1":
        left_gripper_value = Value("d", 0.0, lock=True)
        right_gripper_value = Value("d", 0.0, lock=True)

        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array("d", 2, lock=False)
        dual_gripper_action_array = Array("d", 2, lock=False)

        gripper_ctrl = Dex1_1_Gripper_Controller(
            left_gripper_value,
            right_gripper_value,
            dual_gripper_data_lock,
            dual_gripper_state_array,
            dual_gripper_action_array,
        )

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == "s":
        while True:
            head_img, head_img_fps = img_client.get_head_frame()
            tv_wrapper.set_display_image(head_img)
            tele_data = tv_wrapper.get_tele_data()

            if args.ee == "dex3" and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()

                # optional: if you still want trigger to override even in hand mode:
                with left_trigger_value.get_lock():
                    left_trigger_value.value = tele_data.left_ctrl_triggerValue
                with right_trigger_value.get_lock():
                    right_trigger_value.value = tele_data.right_ctrl_triggerValue

            elif args.ee == "dex3" and args.xr_mode == "controller":
                # controller mode: drive only by trigger (hand arrays stay zeros, that's OK)
                with left_trigger_value.get_lock():
                    left_trigger_value.value = tele_data.left_ctrl_triggerValue
                with right_trigger_value.get_lock():
                    right_trigger_value.value = tele_data.right_ctrl_triggerValue

            elif args.ee == "dex1" and args.xr_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue

            elif args.ee == "dex1" and args.xr_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue

            time.sleep(0.01)

