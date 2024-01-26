from vertices_coordinates import find_vertices
import cv2

"""
Inputs:
    Grayscale image path of Panicle
    XML Result path by Jouannic Paper 
    
Outputs:
    Original Image with Vertices added with color based on their types
    Save this new image to folder labelled_dataset/ 
    
Instructions:
    Run rgb_2_gray.py first then take its output as this file's input

"""

def main():
    path = r"grayscale_dataset/1C8-10-2.JPG"
    img = cv2.imread(path)

    xml_file_path = f"Jouannic_xml/{path[18:-4]}.ricepr"

    generating = find_vertices("Generating", xml_file_path)
    primary = find_vertices("Primary", xml_file_path)
    secondary = find_vertices("Seconday", xml_file_path)
    tertiary = find_vertices("Tertiary", xml_file_path)
    quaternary = find_vertices("Quaternary", xml_file_path)
    terminal = find_vertices("End", xml_file_path)

    for a, b in generating:
        cv2.circle(img, (a, b), 10, (0, 255, 255), cv2.FILLED)
    for a, b in primary:
        cv2.circle(img, (a, b), 10, (255, 255, 255), cv2.FILLED)
    for a, b in secondary:
        cv2.circle(img, (a, b), 10, (255, 0, 0), cv2.FILLED)
    for a, b in tertiary:
        cv2.circle(img, (a, b), 10, (0, 255, 0), cv2.FILLED)
    for a, b in quaternary:
        cv2.circle(img, (a, b), 10, (255, 255, 0), cv2.FILLED)
    for a, b in terminal:
        cv2.circle(img, (a, b), 10, (0, 0, 255), cv2.FILLED)

    cv2.imwrite(f"labelled_dataset/{path[18:]}", img)


if __name__ == "__main__":
    main()
