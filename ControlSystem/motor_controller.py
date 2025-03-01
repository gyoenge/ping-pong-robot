import queue
import threading
import dynamixel_sdk as dxl
import time


class MC:
    def __init__(self, port="COM3"):
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_PROFILE_VELOCITY = 112
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.port = port
        self.baud = 57600
        self.ids = [1, 2, 3]
        self.protocol = 2.0
        self.init_pos = [180, 180, 180]
        self.curr_pos = [180, 180, 180]
        self.portHandler = dxl.PortHandler(self.port)
        self.packetHandler = dxl.PacketHandler(self.protocol)
        self.groupSyncWrite = dxl.GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, 4)

        if self.portHandler.openPort():
            print("Succeeded to open the port.")
        else:
            print("Failed to open the port.")
            quit()

        if self.portHandler.setBaudRate(self.baud):
            print("Succeeded to set the baudrate.")
        else:
            print("Failed to set the baudrate.")
            quit()

        self.set_torque(1, 1)
        self.set_torque(2, 1)
        self.set_torque(3, 1)
        self.reset_velocity(70)

    def degrees_to_dxl_value(self, angle_degrees):
        return int(angle_degrees * 4096.0 / 360.0)

    def set_velocity(self, id, velocity):
        profile_velocity = velocity

        self.dxl_addparam_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, id, self.ADDR_PROFILE_VELOCITY, profile_velocity)

    def set_torque(self, motor_id, enable):
        self.dxl_comm_result, self.dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_id, self.ADDR_TORQUE_ENABLE, enable)
        if self.dxl_comm_result != dxl.COMM_SUCCESS:
            print(
                f"Failed to communicate with motor {motor_id}. Error: {self.dxl_error}")

    def single_move(self, motor_id, angle):
        self.set_torque(motor_id, 1)
        self.goal_position = self.degrees_to_dxl_value(angle)
        self.dxl_comm_result, self.dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, motor_id, self.ADDR_GOAL_POSITION, self.goal_position)
        if self.dxl_comm_result == dxl.COMM_SUCCESS:
            print(f"Motor moved to target angle: {angle} degrees.")
        else:
            print(f"Failed to move the motor. Error: {self.dxl_error}")

    def move(self, angles):
        self.goal_positions = [
            self.degrees_to_dxl_value(angle) for angle in angles]

        for motor_id, goal_position in zip(self.ids, self.goal_positions):
            param = [dxl.DXL_LOBYTE(dxl.DXL_LOWORD(goal_position)),
                     dxl.DXL_HIBYTE(dxl.DXL_LOWORD(goal_position)),
                     dxl.DXL_LOBYTE(dxl.DXL_HIWORD(goal_position)),
                     dxl.DXL_HIBYTE(dxl.DXL_HIWORD(goal_position))]

            self.dxl_addparam_result = self.groupSyncWrite.addParam(
                motor_id, param)
            if self.dxl_addparam_result != True:
                print("Failed to addParam for sync write.")
                quit()

        self.groupSyncWrite.txPacket()
        self.groupSyncWrite.clearParam()

        print("All motors moved to target angles.")

    def position(self, motor_id):
        self.dxl_present_position, self.dxl_comm_result, self.dxl_error = self.packetHandler.read4ByteTxRx(
            self.portHandler, motor_id, self.ADDR_PRESENT_POSITION)
        if self.dxl_comm_result != dxl.COMM_SUCCESS:
            print(
                f"Failed to communicate with motor {motor_id}. Error: {self.dxl_error}")
            return None
        return self.dxl_present_position

    def position_all(self):
        print("Angles: ", self.position(1) * 360.0 /
              4096.0, self.position(2) * 360.0 / 4096.0, self.position(3) * 360.0 / 4096.0)

    def reset_velocity(self, velocity=70):
        self.set_velocity(1, velocity)
        self.set_velocity(2, velocity)
        self.set_velocity(3, velocity)

    def close(self):
        self.set_torque(1, 0)
        time.sleep(0.1)
        self.set_torque(2, 0)
        time.sleep(0.1)
        self.set_torque(3, 0)
        time.sleep(0.1)
        self.portHandler.closePort()

    def ready_mid(self):
        self.move([215, 120, 210])
        pass

    def ready_low(self):
        self.single_move(3, 270)
        time.sleep(0.4)
        self.move([180, 100, 270])
        pass

    def flip_mid(self, delay=0):
        time.sleep(delay)
        self.reset_velocity(100)
        self.set_velocity(3, 700)
        self.move([190, 180, 230])
        time.sleep(0.4)
        self.reset_velocity()
        self.ready_mid()
        pass

    def flip_low(self, delay=0):
        time.sleep(delay)
        self.reset_velocity(120)
        self.set_velocity(3, 700)
        self.single_move(2, 130)
        time.sleep(0.1)
        self.move([160, 170, 290])
        time.sleep(0.4)
        self.reset_velocity(100)
        self.ready_low()
        pass


class MCThread(threading.Thread):
    def __init__(self, port="COM3"):
        threading.Thread.__init__(self)
        self.mc = MC(port)
        self.mc.reset()
        self.queue = queue.Queue()

    def add_move(self, action, message=None):
        self.queue.put((action, message))

    def run(self):
        while True:
            action, message = self.queue.get()
            if action == "move":
                if message is None:
                    break
                self.mc.move(message)
                self.queue.task_done()
            elif action == "reset":
                self.mc.reset()
                self.queue.task_done()
            elif action == "position":
                self.mc.position_all()
                self.queue.task_done()
            elif action == "close":
                self.mc.close()
                self.queue.task_done()
            elif action == "motion":
                if message == 1:
                    self.mc.motion_1()
                elif message == 2:
                    self.mc.motion_2()
                elif message == 3:
                    if message is None:
                        break
                    self.mc.motion_3(message)
                    self.queue.task_done()
                elif message == 4:
                    self.mc.motion_4()
                elif message == 5:
                    self.mc.motion_5()
                self.queue.task_done()

    def stop(self):
        self.add_move(None)
