import logging

from numpy import array, ndarray, where
from scipy.interpolate import InterpolatedUnivariateSpline, make_interp_spline
from scipy.signal import decimate


class PredictionFilter:
    def __init__(self, filter_name: str) -> None:
        self.loaded_filter = self.load_filter[filter_name]

    def load_filter(self, filter_name):
        classes = {"omni": OmniSpace, "plane": VirtualPlane, "box": VirtualBox}
        if filter_name in classes:
            return classes[filter_name]
        else:
            raise ValueError(f"Invalid class name: {filter_name}")


class VirtualPlane:
    def __init__(self, pred_config) -> None:
        self.axis = pred_config.virtual_plane.axis
        self.offset = pred_config.virtual_plane.offset
        self.spline_deg = pred_config.virtual_plane.spline_deg

    def filter(self, t_pred, q_t_pred):
        q_t_pred = array(q_t_pred)

        if len(q_t_pred) <= self.spline_deg:
            logging.debug(
                "Sample number to low.\n"
                f"Sample degree: {self.spline_deg}.\n"
                f"Number of samples: {len(q_t_pred)}.\n"
                "Spline fitting requires more samples."
            )
            return [], []

        p_t_pred = q_t_pred[:, 0:3]
        p_t_offset = p_t_pred[:, self.axis] - self.offset
        univariate_spline = InterpolatedUnivariateSpline(t_pred, p_t_offset)

        crossing_times = list(univariate_spline.roots())

        spline = make_interp_spline(t_pred, p_t_pred, axis=0)

        t_hitpoints = []
        hitpoints = []

        for t in crossing_times:
            t_hitpoints.append(t)
            hitpoints.append(spline(t))

        return t_hitpoints, hitpoints


class VirtualBox:
    def __init__(self, pred_config) -> None:
        center = pred_config.virtual_box.center

        depth = pred_config.virtual_box.depth  # x-axis
        width = pred_config.virtual_box.width  # y-axis
        height = pred_config.virtual_box.height  # z-axis

        self.xlim = (center[0] - depth / 2, center[0] + depth / 2)
        self.ylim = (center[1] - width / 2, center[1] + width / 2)
        self.zlim = (center[2] - height / 2, center[2] + height / 2)

        self.downsample = pred_config.setting.downsample
        f_predictor = pred_config.setting.f_predictor
        f_downsample = pred_config.setting.f_downsample

        self.filter_order = pred_config.setting.filter_order
        self.decimate_factor = int(f_predictor / f_downsample)

    def filter(self, t_pred, q_t_pred) -> ndarray:
        t_pred = array(t_pred)
        q_t_pred = array(q_t_pred)

        p_t_pred = q_t_pred[:, 0:3]

        min_x, max_x = self.xlim
        min_y, max_y = self.ylim
        min_z, max_z = self.zlim

        mask_x = (p_t_pred[:, 0] >= min_x) & (p_t_pred[:, 0] <= max_x)
        mask_y = (p_t_pred[:, 1] >= min_y) & (p_t_pred[:, 1] <= max_y)
        mask_z = (p_t_pred[:, 2] >= min_z) & (p_t_pred[:, 2] <= max_z)
        mask = mask_x & mask_y & mask_z
        indices = where(mask)[0]

        t_filtered = []
        p_t_filtered = []

        if indices.size != 0:
            t_filtered = t_pred[indices]
            p_t_filtered = p_t_pred[indices]

        if self.downsample:
            t_filtered = decimate(t_filtered, self.decimate_factor, self.filter_order)
            p_t_filtered = decimate(
                p_t_filtered, self.decimate_factor, self.filter_order, axis=0
            )

        return t_filtered, p_t_filtered


class OmniSpace:
    def __init__(self, pred_config) -> None:
        self.downsample = pred_config.setting.downsample
        f_predictor = pred_config.setting.f_predictor
        f_downsample = pred_config.setting.f_downsample

        self.filter_order = pred_config.setting.filter_order
        self.decimate_factor = int(f_predictor / f_downsample)

    def filter(self, t_pred, q_t_pred) -> ndarray:
        if self.downsample:
            t_pred = decimate(t_pred, self.decimate_factor, self.filter_order)
            q_t_pred = decimate(
                q_t_pred, self.decimate_factor, self.filter_order, axis=0
            )

        return t_pred, q_t_pred
