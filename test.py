import click
import tqdm
import torch
import trimesh

from pl_model import LitModel
from data.st_data import TeethLandDataset


@click.command()
@click.option('--checkpoint', type=str, required=True,
              help='path to a Lightning .ckpt (required; there is no baked-in default)')
@click.option('--split-file', type=str, default='test.txt')
@click.option('--limit', type=int, default=None, help='only visualise the first N patches')
@click.option('--show/--no-show', default=True, help='open the trimesh viewer')
def run(checkpoint, split_file, limit, show):
    # device is derived from the checkpoint/model, never hardcoded
    model = LitModel.load_from_checkpoint(checkpoint, map_location='cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()

    args = model.hparams.args
    dataset = TeethLandDataset(args, split_file, False)

    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in tqdm.tqdm(range(n)):
        feats, t_idx, _, _, _ = dataset[i]
        off_file = dataset.files[i][0]

        mesh = trimesh.load(off_file)
        vs_offset = mesh.vertices.mean(0)
        x = feats.unsqueeze(0).to(device)

        with torch.no_grad():
            # inference() returns one (points, labels) pair per batch element
            pts, p_labels = model.infer(x)[0]
        pts = pts + vs_offset

        print(f"{off_file}  t_idx={t_idx}  -> {len(pts)} landmarks")
        for p, lab in zip(pts, p_labels):
            print(f"   label {int(lab)}: {p.round(3).tolist()}")
        if not show:
            continue

        spheres = [trimesh.primitives.Sphere(radius=0.2, center=p).to_mesh() for p in pts]
        for s in spheres:
            s.visual.vertex_colors = (0, 255, 0, 255)
        trimesh.Scene([mesh] + spheres).show()


if __name__ == "__main__":
    run()
