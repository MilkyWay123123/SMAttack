import torch


class ThinPlateSpline:
    def __init__(self, source_points, target_points, reg_lambda=1e-6, device='cuda'):
        self.device = device

        self.source_points = source_points.to(self.device)
        self.target_points = target_points.to(self.device)
        self.N = source_points.shape[0]
        self.M = source_points.shape[1]
        self.C = source_points.shape[2]
        self.reg_lambda = reg_lambda
        self.weights = self._compute_weights()

    def _compute_weights(self):
        K = self._pairwise_tps_kernel(self.source_points, self.source_points)  # Shape: [N, M, M]
        P = torch.cat(
            [self.source_points, torch.ones(self.N, self.M, 1, device=self.device)], dim=2
        )

        L_top = torch.cat([K, P], dim=2)
        L_bottom = torch.cat(
            [P.transpose(1, 2), torch.zeros(self.N, self.C + 1, self.C + 1, device=self.device)], dim=2
        )
        L = torch.cat([L_top, L_bottom], dim=1)

        L_reg = L + self.reg_lambda * torch.eye(L.shape[-1], device=self.device)

        Y = torch.cat(
            [self.target_points, torch.zeros(self.N, self.C + 1, self.C, device=self.device)], dim=1
        )

        try:
            weights = torch.linalg.solve(L_reg, Y)
        except RuntimeError:
            print("Warning: Solver failed, using pseudoinverse instead.")
            weights = torch.linalg.pinv(L_reg) @ Y
        return weights

    def _pairwise_tps_kernel(self, X, Y):
        dist_sq = torch.cdist(X, Y, p=2) ** 2
        return dist_sq * torch.log(dist_sq + 1e-8)

    def transform(self, points):
        points = points.to(self.device)
        K = self._pairwise_tps_kernel(points, self.source_points)
        P = torch.cat(
            [points, torch.ones(points.shape[0], points.shape[1], 1, device=self.device)], dim=2
        )
        L = torch.cat([K, P], dim=2)
        return torch.matmul(L, self.weights)

