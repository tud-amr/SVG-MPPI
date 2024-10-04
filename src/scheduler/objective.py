import torch


class Objective:
    def __init__(self, u_min, u_max, device):
        self._waypoints = torch.zeros((1, 2), device=device)

        self.u_min = torch.tensor(u_min, device=device)
        self.u_max = torch.tensor(u_max, device=device)

        self.device = device

        self.alpha = 0
        self.dingo = 18     # number of bodies in Dingo

        self.w_distance = 0
        self.w_progress = 0

        self.w_rotation = 0

        self.w_contact = 0
        self.w_control = torch.diag(torch.tensor([0., 0., 0.], device=device))

        self._cumulative_contact_force = None

    def reset(self):
        self._cumulative_contact_force = None

    def compute_cost(self, sim, u, t, T):
        self._update_contact_force(sim)

        cost = 0.

        if t == T - 1:
            indices, cost_closest_distance = self._distance_closest_waypoint(
                sim)
            cost += self.w_distance * cost_closest_distance

            cost_path_progress = self._relative_progress(indices)
            cost += self.w_progress * cost_path_progress

            cost_rotatation_to_goal = self._rotation_to_goal(sim, indices)
            cost += self.w_rotation * cost_rotatation_to_goal

            cost_contact_force = self._contact_forces_to_goal()
            cost += self.w_contact * cost_contact_force

        return cost

    def _cost_control_vec(self, sim, u):
        cost_control_vec_per_env = torch.zeros((u.shape[0]))

        sampled_rob_vel = torch.cat((sim.dof_state[:, 1].unsqueeze(1),
                                     sim.dof_state[:, 3].unsqueeze(1),
                                     sim.dof_state[:, 5].unsqueeze(1)), 1)

        normalized_u = torch.abs(sampled_rob_vel - u) / \
            torch.max(self.u_max, torch.abs(self.u_min))
        cost_control_vec_per_env = torch.sum(
            (normalized_u @ self.w_control) * normalized_u, dim=1)

        return cost_control_vec_per_env

    def _distance_closest_waypoint(self, sim, m=1):
        sampled_rob_pos = torch.cat((sim.dof_state[:, 0].unsqueeze(
            1), sim.dof_state[:, 2].unsqueeze(1)), dim=1)
        distances = torch.linalg.norm(self.waypoints.unsqueeze(
            0) - sampled_rob_pos.unsqueeze(1), dim=2)

        next_indices = distances.argmin(dim=1) + m
        next_indices = torch.clamp(
            next_indices, min=0, max=len(self.waypoints) - 1)

        normalized_distances = distances[torch.arange(
            sim.num_envs), next_indices] / self.alpha
        return next_indices, normalized_distances

    def _relative_progress(self, next_indices):
        min_index = next_indices.min()
        max_index = next_indices.max()

        range_index = max_index - min_index
        if range_index == 0:
            return torch.zeros_like(next_indices, device=self.device)

        normalized_difference = (next_indices - min_index) / range_index
        reverse_normal_difference = 1.0 - normalized_difference
        return reverse_normal_difference

    def _rotation_to_goal(self, sim, indices, n=1):
        rotation_goal_indices = torch.clamp(
            indices + n, max=len(self.waypoints) - 1)
        rotation_goal = self.waypoints[rotation_goal_indices.squeeze()]

        delta_x = rotation_goal[:, 0] - sim.dof_state[:, 0]
        delta_y = rotation_goal[:, 1] - sim.dof_state[:, 2]

        tar_yaw = torch.atan2(delta_y, delta_x)
        rob_yaw = sim.dof_state[:, n]

        yaw_diff = torch.abs(tar_yaw - rob_yaw)

        rotation_norm_cost_per_env = yaw_diff / torch.pi
        return rotation_norm_cost_per_env

    def _update_contact_force(self, sim):
        if self._cumulative_contact_force is None:
            self._cumulative_contact_force = torch.zeros(
                (sim.num_envs), device=self.device)

        xy_contact_forces = torch.abs(torch.cat((sim.net_contact_force[:, 0].unsqueeze(
            1), sim.net_contact_force[:, 1].unsqueeze(1)), 1))
        xy_contact_forces = torch.sum(xy_contact_forces, dim=1)

        number_env_bodies = int(xy_contact_forces.size(dim=0) / sim.num_envs)
        xy_force_per_body = xy_contact_forces.reshape(
            [sim.num_envs, number_env_bodies])

        self._cumulative_contact_force += torch.sum(
            xy_force_per_body[:, :self.dingo], dim=1)

    def _contact_forces_to_goal(self, epsilon=1e-3):
        interaction_norm_per_env = self._cumulative_contact_force / \
            (torch.max(self._cumulative_contact_force) + epsilon)
        return interaction_norm_per_env

    @property
    def waypoints(self):
        return self._waypoints

    @waypoints.setter
    def waypoints(self, new_waypoints):
        self._waypoints = new_waypoints
