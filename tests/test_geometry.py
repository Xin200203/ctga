import pytest

torch = pytest.importorskip("torch")

from ctga.common.geometry import bbox_iou_3d, voxelize


def test_voxelize_preserves_shape():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=torch.float32)
    voxels = voxelize(points, 0.1)
    assert voxels.shape == (2, 3)


def test_bbox_iou_3d_identity():
    box = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)
    assert torch.isclose(bbox_iou_3d(box, box), torch.tensor(1.0))
