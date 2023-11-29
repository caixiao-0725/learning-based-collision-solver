def check_collision(triangle1, triangle2):
    def get_axes(triangle):
        axes = []
        for i in range(len(triangle)):
            p1 = triangle[i]
            p2 = triangle[(i + 1) % len(triangle)]
            edge = (p1[0] - p2[0], p1[1] - p2[1])
            axis = (-edge[1], edge[0])
            axes.append(axis)
        return axes

    def project(triangle, axis):
        min_proj = max_proj = None
        for point in triangle:
            projection = point[0] * axis[0] + point[1] * axis[1]
            if min_proj is None or projection < min_proj:
                min_proj = projection
            if max_proj is None or projection > max_proj:
                max_proj = projection
        return (min_proj, max_proj)

    def overlap(projection1, projection2):
        return not (projection1[1] < projection2[0] or projection2[1] < projection1[0])

    axes1 = get_axes(triangle1)
    axes2 = get_axes(triangle2)

    for axis in axes1 + axes2:
        projection1 = project(triangle1, axis)
        projection2 = project(triangle2, axis)
        if not overlap(projection1, projection2):
            return False

    return True