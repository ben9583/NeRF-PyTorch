import torch
import viser, time
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
  EPOCHS = 2000
  IMAGE_HEIGHT = 200
  IMAGE_WIDTH = 200
  SAMPLES = 10

  def sinusoidal_positional_encoding(L=10):
    seq = torch.arange(L).to(device) * torch.pi

    def helper(x):
      if len(x.shape) == 2:
        pos_enc = torch.zeros((x.shape[0], 1 + 2 * L, x.shape[1])).to(device)
        pos_enc[:, 0, :] = x
        intermediate = torch.einsum('i,bj->bij', seq, x)
        pos_enc[:, 1::2, :] = torch.sin(intermediate)
        pos_enc[:, 2::2, :] = torch.cos(intermediate)
        return pos_enc
      else:
        # batched on first dim
        pos_enc = torch.zeros((x.shape[0], x.shape[1], 1 + 2 * L, x.shape[2])).to(device)
        pos_enc[:, :, 0, :] = x
        intermediate = torch.einsum('i,bcj->bcij', seq, x)
        pos_enc[:, :, 1::2, :] = torch.sin(intermediate)
        pos_enc[:, :, 2::2, :] = torch.cos(intermediate)
        return pos_enc

    return helper


  class SimpleMLPModel(nn.Module):
    def __init__(self, num_sinusoidal=10):
      super().__init__()
      self.num_sinusoidal = num_sinusoidal

      self.encoder = sinusoidal_positional_encoding(num_sinusoidal)
      self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear((1 + 2 * num_sinusoidal) * 2, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 3),
        nn.Sigmoid()
      )

    def forward(self, x):
      x = self.encoder(x)
      x = self.layers(x)
      return x


  class NeRFImageDataset(Dataset):
    def __init__(self, image, sample_size=10000):
      assert len(image.shape) == 3
      assert image.shape[2] == 3

      if type(image) == np.ndarray:
        image = torch.from_numpy(image).to(device)

      if torch.max(image) > 1.0:
        image = image / 255.0

      self.image = image
      self.num_pixels = image.shape[0] * image.shape[1]

      self.sample_size = sample_size

    def __len__(self):
      return 1

    def __getitem__(self, _idx):
      rnd = np.random.randint(0, self.num_pixels, (self.sample_size,))
      x = rnd % self.image.shape[1]
      y = rnd // self.image.shape[1]
      pixels = self.image[y, x, :]
      return torch.stack([torch.Tensor(y) / self.image.shape[0], torch.Tensor(x) / self.image.shape[1]], dim=-1), pixels


  def train(model, train_loader, loss_fn, optimizer, img, epochs=10):
    inputs = torch.stack(torch.meshgrid(torch.arange(IMAGE_HEIGHT).to(device) / IMAGE_HEIGHT, torch.arange(IMAGE_WIDTH).to(device) / IMAGE_WIDTH), dim=-1).reshape(-1, 2)
    losses = np.zeros((epochs,))
    outputs = np.zeros((SAMPLES + 1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    psnrs = np.zeros((SAMPLES + 1,))

    with torch.no_grad():
      y_pred = model(inputs)
      print("PSNR:", (10 * torch.log10(1.0 / loss_fn(y_pred, img.reshape(-1, 3)))).item())
      y_pred = y_pred.detach().numpy().reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
      y_pred = np.clip(y_pred, 0.0, 1.0)
      y_pred = y_pred * 255.0
      y_pred = y_pred.astype(np.uint8)
      outputs[0] = y_pred

    for epoch in range(epochs):
      for X, y in train_loader:
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        losses[epoch] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      print(f'Iter: {epoch}, Loss: {loss},')

      if np.floor((epoch + 1) / epochs * SAMPLES) > np.floor(epoch / epochs * SAMPLES):
        with torch.no_grad():
          y_pred = model(inputs)
          psnr = (10 * torch.log10(1.0 / loss_fn(y_pred, img.reshape(-1, 3)))).item()
          print("PSNR:", psnr)
          y_pred = y_pred.detach().numpy().reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
          y_pred = np.clip(y_pred, 0.0, 1.0)
          y_pred = y_pred * 255.0
          y_pred = y_pred.astype(np.uint8)
          outputs[np.floor(epoch / epochs * SAMPLES).astype(np.int32) + 1] = y_pred
          psnrs[np.floor(epoch / epochs * SAMPLES).astype(np.int32) + 1] = psnr

    return losses, outputs, psnrs

  # sample_image = plt.imread('cross_fox.jpg').astype(np.float32) / 255.0
  # dataset = NeRFImageDataset(sample_image)

  # train_loader = DataLoader(dataset, batch_size=1)
  # model = SimpleMLPModel(256).to(device)
  # loss_fn = nn.MSELoss()
  # optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)

  # losses, outputs, psnrs = train(model, train_loader, loss_fn, optimizer, torch.Tensor(sample_image), epochs=EPOCHS)
  # for i in range(SAMPLES + 1):
  #   plt.imsave('outputs/simple_mlp/{}.png'.format(i), outputs[i], format='png')

  # plt.title('Loss over time')
  # plt.xlabel('Iterations')
  # plt.ylabel('Loss')
  # plt.plot(losses)
  # plt.savefig('outputs/simple_mlp/loss.png')
  # plt.clf()
  # plt.title('PSNR over time')
  # plt.xlabel('Iterations')
  # plt.ylabel('PSNR')
  # plt.plot(psnrs)
  # plt.savefig('outputs/simple_mlp/psnr.png')

  def transform(c2w, x_c):
    return torch.bmm(c2w, x_c)

  def pixel_to_camera(K, uv, s):
    return torch.matmul(K.inverse(), (s * uv).T).T

  def pixel_to_ray(K, c2w, uv):
    w2c = c2w.inverse()
    r_o = torch.bmm(-w2c[:, :3, :3].inverse(), w2c[:, :3, 3].unsqueeze(2)).squeeze(2)
    x_w = transform(c2w, torch.cat((pixel_to_camera(K, uv, 1.0), torch.ones_like(uv[:, 0]).unsqueeze(1)), dim=1).unsqueeze(2)).to(device).squeeze(2)[:, :3]
    r_d = (x_w - r_o) / torch.norm(x_w - r_o, dim=-1, keepdim=True)
    return r_o, r_d


  class NeRF3DImageDataset(Dataset):
    def __init__(self, images, c2ws, K, focal, sample_size=10000):
      assert len(images.shape) == 4
      assert images.shape[3] == 3

      if type(images) == np.ndarray:
        images = torch.from_numpy(images).to(device)

      if torch.max(images) > 1.0:
        images = images / 255.0

      self.image = images
      self.num_pixels = images.shape[0] * images.shape[1] * images.shape[2]

      self.sample_size = sample_size
      self.c2ws = c2ws
      self.K = K
      self.focal = focal

    def __len__(self):
      return 1

    def get_vals(self, vals):
      image_idx = vals // (self.image.shape[1] * self.image.shape[2])
      x = vals % self.image.shape[2]
      y = (vals // self.image.shape[2]) % self.image.shape[1]
      pixels = self.image[image_idx, y, x, :]
      rays_o, rays_d = pixel_to_ray(
        self.K,
        self.c2ws[image_idx],
        torch.stack([
          torch.Tensor(x).to(device) + 0.5,
          torch.Tensor(y).to(device) + 0.5,
          torch.ones_like(torch.Tensor(x)).to(device)
        ], dim=-1)
      )
      return (rays_o, rays_d), pixels

    def __getitem__(self, _idx):
      rnd = np.random.randint(0, self.num_pixels, (self.sample_size,))
      return self.get_vals(rnd)

    def sample_rays(self, sample_size):
      old = self.sample_size
      self.sample_size = sample_size
      rays, pixels = self.__getitem__(0)
      self.sample_size = old
      return rays, pixels

  loaded = np.load('drive/MyDrive/lego_200x200.npz')
  c2ws_train = torch.Tensor(loaded['c2ws_train']).to(device)

  def sample_points_from_ray(r_o, r_d, near=2.0, far=6.0, num_samples=64, perturb=True):
    intervals = torch.linspace(0.0, 1.0, num_samples + 1).repeat(r_o.shape[0], 1).to(device)
    if perturb:
      intervals = torch.clamp(intervals + torch.rand_like(intervals).to(device) / num_samples, torch.zeros_like(intervals).to(device), torch.ones_like(intervals).to(device))
    deltas = intervals[:, 1:] - intervals[:, :-1]
    t_vals = (intervals[:, :-1] + intervals[:, 1:]) / 2.0
    z_vals = near * (1.0 - t_vals) + far * t_vals
    pts = r_o.unsqueeze(1).repeat(1, num_samples, 1) + torch.einsum('bi,bj->bij', r_d, z_vals).mT
    return pts, deltas

  K = torch.Tensor(np.array([[loaded['focal'], 0, IMAGE_WIDTH // 2], [0, loaded['focal'], IMAGE_HEIGHT // 2], [0, 0, 1]])).to(device)

  def show_some_rays():
    dataset = NeRF3DImageDataset(loaded['images_train'], c2ws_train, K, loaded['focal'])
    rays, _ = dataset.sample_rays(100)
    rays_o, rays_d = rays
    points, _ = sample_points_from_ray(rays_o, rays_d, perturb=True)
    rays_o, rays_d, points = rays_o.detach().numpy(), rays_d.detach().numpy(), points.detach().numpy()

    server = viser.ViserServer(share=True)
    for i, (image, c2w) in enumerate(zip(loaded['images_train'], c2ws_train)):
      server.add_camera_frustum(
        f"/cameras/{i}",
        fov=2 * np.arctan2(IMAGE_HEIGHT / 2, loaded['focal']),
        aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
        scale=0.15,
        wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3].detach().numpy(),
        image=image
      )

    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
      server.add_spline_catmull_rom(
        f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
      )
    server.add_point_cloud(
      f"/samples",
      colors=np.zeros_like(points).reshape(-1, 3),
      points=points.reshape(-1, 3),
      point_size=0.02,
    )
    time.sleep(1000)

  def show_rays_from_one():
    dataset = NeRF3DImageDataset(loaded['images_train'], c2ws_train, K, loaded['focal'])

    rays, _ = dataset.get_vals(np.arange(0, 200 * 200, 843))
    rays_o, rays_d = rays
    points, _ = sample_points_from_ray(rays_o, rays_d, perturb=True)
    rays_o, rays_d, points = rays_o.detach().numpy(), rays_d.detach().numpy(), points.detach().numpy()

    # ---------------------------------------

    server = viser.ViserServer(share=True)
    for i, (image, c2w) in enumerate(zip(loaded['images_train'], c2ws_train)):
      server.add_camera_frustum(
        f"/cameras/{i}",
        fov=2 * np.arctan2(IMAGE_HEIGHT / 2, loaded['focal']),
        aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
        scale=0.15,
        wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3].detach().numpy(),
        image=image
      )
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
      positions = np.stack((o, o + d * 6.0))
      server.add_spline_catmull_rom(
          f"/rays/{i}", positions=positions,
      )
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.03,
    )
    time.sleep(1000)

  # show_some_rays()
  # show_rays_from_one()

  def volumetric_rendering(sigmas, rgbs, deltas):
    deltas = deltas.unsqueeze(-1)
    probs = 1.0 - torch.exp(-sigmas * deltas)
    T = torch.zeros((probs.shape[0], probs.shape[1] + 1, probs.shape[2])).to(device)
    T[:, 0, :] = 1.0
    T[:, 1:, :] = torch.exp(-torch.cumsum(sigmas * deltas, dim=1))
    colors = torch.sum(T[:, :-1, :] * probs * rgbs, dim=-2)
    return colors

  class NeRFMLPModel(nn.Module):
    def __init__(self, num_sinusoidal=4):
      super().__init__()
      self.num_sinusoidal = num_sinusoidal

      self.encoder = sinusoidal_positional_encoding(num_sinusoidal)
      self.fc1 = nn.Linear((1 + 2 * num_sinusoidal) * 3, 256)
      self.fc2 = nn.Linear(256, 256)
      self.fc3 = nn.Linear(256, 256)
      self.fc4 = nn.Linear(256, 256)
      self.fc5 = nn.Linear(256 + (1 + 2 * num_sinusoidal) * 3, 256)
      self.fc6 = nn.Linear(256, 256)
      self.fc7 = nn.Linear(256, 256)
      self.fc8 = nn.Linear(256, 256)
      self.fca9 = nn.Linear(256, 1)
      self.fcb9 = nn.Linear(256, 256)
      self.fcb10 = nn.Linear(256 + (1 + 2 * num_sinusoidal) * 3, 128)
      self.fcb11 = nn.Linear(128, 3)

    def forward(self, inputs):
      r_o, r_d = inputs
      r_o, r_d = r_o.squeeze(0), r_d.squeeze(0)
      x, deltas = sample_points_from_ray(r_o, r_d, perturb=True)
      x = self.encoder(x)
      x_orig = torch.flatten(x, start_dim=2)
      r_d = self.encoder(r_d)
      r_d = r_d.unsqueeze(1).repeat(1, 64, 1, 1)
      r_d = torch.flatten(r_d, start_dim=2)
      x = self.fc1(x_orig)
      x = torch.relu(x)
      x = self.fc2(x)
      x = torch.relu(x)
      x = self.fc3(x)
      x = torch.relu(x)
      x = self.fc4(x)
      x = torch.relu(x)
      x = torch.cat((x, x_orig), dim=-1)
      x = self.fc5(x)
      x = torch.relu(x)
      x = self.fc6(x)
      x = torch.relu(x)
      x = self.fc7(x)
      x = torch.relu(x)
      x = self.fc8(x)
      xa = self.fca9(x)
      xa = torch.relu(xa)
      xb = self.fcb9(x)
      xb = torch.cat((xb, r_d), dim=-1)
      xb = self.fcb10(xb)
      xb = torch.relu(xb)
      xb = self.fcb11(xb)
      xb = torch.sigmoid(xb)

      return volumetric_rendering(xa, xb, deltas)


  def train(model, train_dataset, validation_dataset, loss_fn, optimizer, epochs=10):
    train_loader = DataLoader(train_dataset, batch_size=1)
    validation_loader = DataLoader(validation_dataset, batch_size=1)

    losses = np.zeros((epochs,))
    psnrs = np.zeros((epochs,))
    for epoch in range(epochs):
      for X, y in train_loader:
        y_pred = model(X)
        loss = loss_fn(y_pred, y.squeeze(0))
        losses[epoch] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      with torch.no_grad():
        val_losses = []
        for X, y in validation_loader:
          y_pred = model(X)
          val_losses.append(loss_fn(y_pred, y.squeeze(0)).item())
        psnrs[epoch] = 10 * np.log10(1.0 / np.mean(val_losses))
        print(f'Iter: {epoch}, Loss: {loss}, Validation loss: {np.mean(val_losses)}, PSNR: {psnrs[epoch]}')

        if (epoch + 1) % 100 == 0:
          for i in range(10):
            X, _ = validation_dataset.get_vals(np.arange(200 * 200 * i, 200 * 200 * (i + 1)))
            y_pred = model(X)
            y_pred = y_pred.detach().cpu().numpy().reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            y_pred = np.clip(y_pred, 0.0, 1.0)
            y_pred = y_pred * 255.0
            y_pred = y_pred.astype(np.uint8)
            plt.imsave('drive/MyDrive/nerf_outputs/nerf_mlp/{}_{}.png'.format(epoch + 1, i), y_pred, format='png')
    return losses, psnrs

  torch.manual_seed(42)
  sigmas = torch.rand((10, 64, 1)).to(device)
  rgbs = torch.rand((10, 64, 3)).to(device)
  step_size = torch.ones((10, 64)).to(device) * (6.0 - 2.0) / 64
  rendered_colors = volumetric_rendering(sigmas, rgbs, step_size)

  correct = torch.Tensor([
    [0.5006, 0.3728, 0.4728],
    [0.4322, 0.3559, 0.4134],
    [0.4027, 0.4394, 0.4610],
    [0.4514, 0.3829, 0.4196],
    [0.4002, 0.4599, 0.4103],
    [0.4471, 0.4044, 0.4069],
    [0.4285, 0.4072, 0.3777],
    [0.4152, 0.4190, 0.4361],
    [0.4051, 0.3651, 0.3969],
    [0.3253, 0.3587, 0.4215]
  ]).to(device)
  assert torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4)

  dataset = NeRF3DImageDataset(loaded['images_train'], c2ws_train, K, loaded['focal'])
  validation_dataset = NeRF3DImageDataset(loaded['images_val'], torch.Tensor(loaded['c2ws_val']).to(device), K, loaded['focal'])
  model = NeRFMLPModel(10).to(device)
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
  losses, psnrs = train(model, dataset, validation_dataset, loss_fn, optimizer, epochs=EPOCHS)

  plt.title('Loss over time')
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.plot(losses)
  plt.savefig('drive/MyDrive/nerf_outputs/nerf_mlp/loss.png')
  plt.clf()
  plt.title('PSNR over time')
  plt.xlabel('Iterations')
  plt.ylabel('PSNR')
  plt.plot(psnrs)
  plt.savefig('drive/MyDrive/nerf_outputs/nerf_mlp/psnr.png')

  print("Evaluating test set...")
  test_dataset = NeRF3DImageDataset(torch.zeros((100, 200, 200, 3)).to(device), torch.Tensor(loaded['c2ws_test']).to(device), K, loaded['focal'])
  with torch.no_grad():
    for i in range(60):
      X, _ = test_dataset.get_vals(np.arange(200 * 200 * i, 200 * 200 * (i + 1)))
      y_pred = model(X)
      y_pred = y_pred.detach().cpu().numpy().reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
      y_pred = np.clip(y_pred, 0.0, 1.0)
      y_pred = y_pred * 255.0
      y_pred = y_pred.astype(np.uint8)
      plt.imsave('drive/MyDrive/nerf_outputs/nerf_mlp/test_{}.png'.format(i), y_pred, format='png')

if __name__ == '__main__':
  main()
