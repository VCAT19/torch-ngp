import mcubes
import trimesh
import numpy as np
import torch

N = 512
t = np.linspace(-1.2, 1.2, N+1)
chunk = 10000

with torch.no_grad():
	query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
	print(query_pts.shape)
	sh = query_pts.shape
	raw = self.model.density(torch.tensor(query_pts).to(device)).cpu().numpy()

sigma = np.maximum(raw[...,-1], 0.)

sigma = sigma.reshape(sh[:-1])

threshold = 5.
print('fraction occupied', np.mean(sigma > threshold))
vertices, triangles = mcubes.marching_cubes(sigma, threshold)
print('done', vertices.shape, triangles.shape)

mesh = trimesh.Trimesh(vertices / N - .5, triangles)
mesh.show()
