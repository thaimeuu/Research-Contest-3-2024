import re


def find_vertices(type, xml_file_path):
    """
    Find vertices coordinate based on type

    Args:
        type (str): Generating, Primary, Secondary, Tertiary, Quaternary or End
        xml_file_path: path to result file

    Returns:
        list of coordinates
    """
    lis = []

    with open(xml_file_path, "r") as file:
        xml_file = file.read()

    pattern = re.compile(rf'" x="(.+)" y="(.+)" type="{type}"')
    matches = pattern.finditer(xml_file)

    for match in matches:
        a, b = int(match.group(1)), int(match.group(2))
        lis.append([a, b])

    return lis
