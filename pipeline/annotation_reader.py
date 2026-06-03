import os
import cv2

class SoyAnnotationReader:
    @staticmethod
    def yolo_to_bbox(yolo_line, img_width, img_height):
        parts = yolo_line.strip().split()
        if len(parts) < 5: return None
        
        class_id = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:5])
        
        xmin = int((x_c - w / 2) * img_width)
        ymin = int((y_c - h / 2) * img_height)
        xmax = int((x_c + w / 2) * img_width)
        ymax = int((x_c + h / 2) * img_height)
        
        return {"class_id": class_id, "box": (xmin, ymin, xmax, ymax)}

    def parse_annotation_file(self, label_path, img_path):
        img = cv2.imread(img_path)
        if img is None: return []
        h, w, _ = img.shape
        
        parsed_objects = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    obj_data = self.yolo_to_bbox(line, w, h)
                    if obj_data:
                        parsed_objects.append(obj_data)
        return parsed_objects
