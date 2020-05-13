import os 
import cv2 as cv 
import json 

train_folder = 'data/train'

img_list = list(filter(lambda x: ".jpg" in x, os.listdir(train_folder)))

start_x, start_y, end_x, end_y = 0, 0, 0, 0
cv_img, draw_img, temp_img = None, None, None  
start_draw = False 
cv.namedWindow("image")

anns = []
cls_id = 0

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, start_draw, draw_img, cv_img, temp_img
    
    if event == cv.EVENT_LBUTTONDOWN:
        start_x = x if x>0 else 0
        start_y = y if y>0 else 0
        start_draw = True

    if event == cv.EVENT_MOUSEMOVE:
        if start_draw:
            draw_img = temp_img.copy()
            cv.rectangle(draw_img, (start_x, start_y), (x, y), (0,0,255), thickness=2)
            cv.imshow("image", draw_img)
        

    if event == cv.EVENT_LBUTTONUP:
        end_x = x 
        end_y = y 
        h = end_y-start_y
        w = end_x-start_x
        img_h, img_w = temp_img.shape[0], temp_img.shape[1]
        if start_y+h > img_h:
            h = img_h - start_y
        if start_x+w > img_w:
            w = img_w - start_x

        start_draw = False
        temp_img = draw_img.copy()

        box = [start_x, start_y, h, w]
        bbox_json = {"bbox": box, "cls_id": cls_id}
        anns.append(bbox_json)
        


if __name__ == "__main__":
    max_len = len(img_list)
    cv.setMouseCallback("image", draw_rectangle)

    for i in range(max_len):
        img_path = os.path.join(train_folder, img_list[i])
        img_name = img_list[i].split('.')[0]
        json_path = os.path.join(train_folder, f"{img_name}.json")
        print(f"img_path: {img_path}, [{i+1}/{max_len}]")
        cv_img = cv.imread(img_path)
        img_h, img_w = cv_img.shape[0], cv_img.shape[1]
        draw_img = cv_img.copy()
        temp_img = cv_img.copy()

        while(1):
            cv.imshow("image", draw_img)
            key = cv.waitKey(1)
            
            if key == 27:
                exit()
            if key == 32:
                annotation_json = {}
                annotation_json["img_path"] = img_list[i]
                annotation_json["width"] = img_w 
                annotation_json["height"] = img_h
                annotation_json["anns"] = anns

                # print(f"annotation_json : {annotation_json}")
                with open(json_path, 'w') as f:
                    json.dump(annotation_json, f, indent=4)

                anns = []
                break
            
            if key == 81:
                print("clear!")
                draw_img = cv_img.copy()
                temp_img = cv_img.copy()
                anns = []


    