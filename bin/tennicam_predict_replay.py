import pathlib
import sys

import o80
import pam_mujoco
import signal_handler
import tennicam_client
from lightargs import BrightArgs, FileExists

TENNICAM_CLIENT_DEFAULT_FOLDER = pathlib.Path("/tmp")
TENNICAM_CLIENT_MUJOCO_ID = "tennicam_predict_replay"


def _get_default_file(
    directory: pathlib.Path, file_prefix="tennicam_*"
) -> pathlib.Path:
    """
    Returns one of the file "/tmp/tennicam_*"
    Return empty string if no such file.
    """

    for filename in directory.glob(file_prefix):
        if filename.is_file():
            return directory / filename

    return ""


def _configure() -> BrightArgs:
    """
    Configuration dialog
    """

    global TENNICAM_CLIENT_DEFAULT_FOLDER
    config = BrightArgs("tennicam replay")
    config.add_option(
        "filepath",
        str(_get_default_file(TENNICAM_CLIENT_DEFAULT_FOLDER)),
        "absolute path of the log file to replay",
        str,
        [FileExists],
    )
    change_all = False
    config.dialog(change_all, sys.argv[1:])
    print()
    return config


def _get_handle() -> pam_mujoco.MujocoHandle:
    """
    Configures mujoco to have a controllable ball
    and returns the handle
    """

    global TENNICAM_CLIENT_MUJOCO_ID
    ball = pam_mujoco.MujocoItem(
        "ball", control=pam_mujoco.MujocoItem.CONSTANT_CONTROL, color=(1, 0, 0, 1)
    )
    graphics = True
    accelerated_time = False
    handle = pam_mujoco.MujocoHandle(
        TENNICAM_CLIENT_MUJOCO_ID,
        table=True,
        balls=(ball,),
        graphics=graphics,
        accelerated_time=accelerated_time,
    )
    return handle


def run(filepath: pathlib.Path):
    """
    Parse filepath and plays the corresponding
    ball trajectory in mujoco (via an o80 frontend)
    """

    # configuring and staring mujoco
    handle = _get_handle()

    # getting a frontend for ball control
    ball = handle.frontends["ball"]

    # parsing file
    ball_infos = list(tennicam_client.parse(filepath))

    # duration between 2 ball observations
    duration = o80.Duration_us.nanoseconds(ball_infos[1][1] - ball_infos[0][1])

    # playing the ball trajectory
    signal_handler.init()  # for detecting ctrl+c
    for ball_info in ball_infos:
        _, _, position, velocity = ball_info
        ball.add_command(position, velocity, duration, o80.Mode.OVERWRITE)
        ball.pulse_and_wait()
        if signal_handler.has_received_sigint():
            break


if __name__ == "__main__":
    config = _configure()
    run(pathlib.Path(config.filepath))
