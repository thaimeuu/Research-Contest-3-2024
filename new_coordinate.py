def new_coordinate(
    old_coordinates: tuple[int, int],
    old_shape: tuple[int, int],
    new_shape: tuple[int, int] = [256, 256],
) -> tuple[int, int]:
    """
    This script is an utility
    
    Returns new coordinate for pixels when changing size

    Args:
        old_coordinates (tuple[int, int]): _description_
        old_shape (tuple[int, int]): _description_
        new_shape (tuple[int, int], optional): _description_. Defaults to [256, 256].

    Returns:
        tuple[int, int]: _description_
    """
    old_x, old_y = old_coordinates
    old_width, old_height = old_shape
    new_width, new_height = new_shape

    new_x = (old_x / old_width) * new_width
    new_y = (old_y / old_height) * new_height

    return (new_x, new_y)

if __name__ == "__main__":
    new_x, new_y = new_coordinate((100, 100), (256, 256), (2658, 1773))
    new_x, new_y = round(new_x), round(new_y)
    print(new_x, new_y)
    