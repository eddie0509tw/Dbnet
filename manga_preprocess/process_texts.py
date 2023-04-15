import manga109api
import os

def xmlToTxt(annotation, page_index: int, store_root_path: str, book_name: str, img_name: str) -> None:
    # "text" as +ve sample, "face" as -ve sample
    ANNOTATION_TYPES = ["face", "text"]
    for annotation_type in ANNOTATION_TYPES:
        rois = annotation["page"][page_idx][annotation_type]

        with open(f"{store_root_path}/{book_name}_{img_name}.txt", "a+") as f:
            for roi in rois:
                if annotation_type == "text":
                    # \n cause txt file format error, ',' will be used to separate coordinates
                    text = roi["#text"].replace('\n', ' ').replace(',', ' ')
                else:
                    # not text, use ### for annotation
                    text = "###"
                
                s = f'{roi["@xmin"]},{roi["@ymin"]},{roi["@xmax"]},{roi["@ymin"]},{roi["@xmax"]},{roi["@ymax"]},{roi["@xmin"]},{roi["@ymax"]},{text}\n'

                f.write(s)


if __name__ == "__main__":
    manga109_root_dir = "../Manga109s_released_2021_12_30"
    
    os.makedirs('train/mangatxt', exist_ok=True)
    
    p = manga109api.Parser(root_dir=manga109_root_dir)
    book_list = p.books
    
    for book in book_list:
        store_root_path = 'train/mangatxt'

        # get all annotations on this book
        annotation = p.get_annotation(book=book)
        # get number of pages in this book
        num_pages = len(annotation["page"])
        
        for page_idx in range(num_pages):
            image_full_path = p.img_path(book=book, index=page_idx)
            # image name eg: ARMS_001
            image_name = image_full_path.split('/')[-1].split('.')[0]
            
            xmlToTxt(annotation, page_idx, store_root_path, book, image_name)
            
            
    
    