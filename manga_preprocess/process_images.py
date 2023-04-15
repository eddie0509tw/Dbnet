import manga109api
import os

if __name__ == "__main__":
    manga109_root_dir = "../Manga109s_released_2021_12_30"
    
    os.makedirs('train/images', exist_ok=True)
        
    p = manga109api.Parser(root_dir=manga109_root_dir)
    book_list = p.books  # a list of all books
    
    for book in book_list:
        store_root_path = 'train/images/'

        # get all annotations on this book
        annotation = p.get_annotation(book=book)
        # get number of pages in this book
        num_pages = len(annotation["page"])
        
        for page_idx in range(num_pages):
            # get image name pf this page
            image_full_path = p.img_path(book=book, index=page_idx)
            image_name = image_full_path.split('/')[-1]
            
            os.system(f'cp "{image_full_path}" "{store_root_path}/{book}_{image_name}"')
    