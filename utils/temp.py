def matrix_function(quat): # quat is a 7D tensor
    matrix = np.zeros((4, 4))
    t1, t2, t3 = quat[0], quat[1], quat[2]
    x = quat[3]
    y = quat[4]
    z = quat[5]
    w = quat[6]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix[0, 0] = x2 - y2 - z2 + w2
    matrix[1, 0] = 2 * (xy + zw)
    matrix[2, 0] = 2 * (xz - yw)

    matrix[0, 1] = 2 * (xy - zw)
    matrix[1, 1] = - x2 + y2 - z2 + w2
    matrix[2, 1] = 2 * (yz + xw)

    matrix[0, 2] = 2 * (xz + yw)
    matrix[1, 2] = 2 * (yz - xw)
    matrix[2, 2] = - x2 - y2 + z2 + w2

    matrix[0, 3] = t1
    matrix[1, 3] = t2
    matrix[2, 3] = t3

    matrix[3, 3] = 1

    return matrix


def batch_matrix_function(quats): 
    # quats is a [B, 7] tensor
    B = quats.shape[0]
    matrices = torch.zeros(B, 4, 4, device=quats.device, dtype=quats.dtype)
    
    t1, t2, t3 = quats[:, 0], quats[:, 1], quats[:, 2]
    x, y, z, w = quats[:, 3], quats[:, 4], quats[:, 5], quats[:, 6]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrices[:, 0, 0] = x2 - y2 - z2 + w2
    matrices[:, 1, 0] = 2 * (xy + zw)
    matrices[:, 2, 0] = 2 * (xz - yw)

    matrices[:, 0, 1] = 2 * (xy - zw)
    matrices[:, 1, 1] = - x2 + y2 - z2 + w2
    matrices[:, 2, 1] = 2 * (yz + xw)

    matrices[:, 0, 2] = 2 * (xz + yw)
    matrices[:, 1, 2] = 2 * (yz - xw)
    matrices[:, 2, 2] = - x2 - y2 + z2 + w2

    matrices[:, 0, 3] = t1
    matrices[:, 1, 3] = t2
    matrices[:, 2, 3] = t3

    matrices[:, 3, 3] = 1

    return matrices

# Create dummy tensors for illustration
matrix_tensor = torch.rand(32, 7)
point_cloud_tensor = torch.rand(32, 256, 3)

# Augment the point cloud tensor with ones to make it [32, 256, 4]
ones = torch.ones(32, 256, 1)
augmented_point_cloud = torch.cat([point_cloud_tensor, ones], dim=2)

# Process the matrix tensor to get the new tensor of size [32, 4, 4]
transform_matrices = torch.stack([matrix_function(tensor) for tensor in matrix_tensor])

# Perform batch matrix multiplication
transformed_augmented_point_cloud = torch.bmm(augmented_point_cloud, transform_matrices.transpose(1, 2))

# Extract the transformed point cloud tensor of size [32, 256, 3]
transformed_point_cloud = transformed_augmented_point_cloud[:, :, :3]
