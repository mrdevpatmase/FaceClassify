from icrawler.builtin import GoogleImageCrawler

def download_images(query, folder_name, max_num=50):
    google_crawler = GoogleImageCrawler(storage={'root_dir': f'dataset/{folder_name}'})
    google_crawler.crawl(keyword=query, max_num=max_num)

# Cricketer face image downloads
download_images("Virat Kohli face", "virat_kohli", 200)
download_images("MS Dhoni face", "ms_dhoni", 200)
download_images("AB de Villiers face", "ab_de_villiers", 200)
download_images("Jasprit Bumrah face", "jasprit_bumrah", 200)
download_images("Ruturaj Gaikwad face", "ruturaj_gaikwad", 200)
