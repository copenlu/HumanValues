import zstandard
import json
import pickle
from tqdm.auto import tqdm
from glob import glob
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import argparse




def arg_parser():
   parser = argparse.ArgumentParser(description="Collect Reddit data to label")
   parser.add_argument("path_to_pushshift", type=str, help="Path to the pushshift data", )
   parser.add_argument("save_path", type=str, help="Path to save the data")
   return parser.parse_args()


def read_compressed_zst_line_by_line(path: str):
   """
   This function reads a compressed .zst file line by line.
   :param path: Path to the .zst file
   :return: Yields one line at a time from the file
   """
   with open(path, 'rb') as fh:
      dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
      reader = dctx.stream_reader(fh)
      chunk = ''
      while True:
         datastream = b''
         while True:
               try:
                  datastream += reader.read(16384)
               except zstandard.ZstdError:
                  return
               if len(datastream) == 0:
                  return
               try:
                  chunk += datastream.decode('utf-8')
                  break
               except UnicodeDecodeError:
                  continue
         newline_id = chunk.rfind("\n")
         if newline_id > 0:
               to_process = chunk[:newline_id]
               chunk = chunk[newline_id + 1:]
               lines = to_process.split("\n")
               for line in lines:
                  yield line

def read_a_collection_of_zsts_line_by_line(paths: List[str]):
   """
   This function reads a collection of .zst files line by line.
   :param paths: List of paths to the .zst files
   :return: Yields a tuple containing the file id and one line at a time from the file
   """
   for i, path in enumerate(paths):
      print(path)
      for line in read_compressed_zst_line_by_line(path):
         yield i, line
            
         
def collect_posts_index_from_zst_files(pushift_path: str,
                                       subreddits: Set[str],
                                       min_score=10,
                                       max_dump_size_per_path=10000) -> Dict[str, List[int]]:
   """
   This function collects the index of posts from .zst files.
   :param subreddits: Set of subreddit names
   :param min_score: Minimum score for a post to be considered
   :param max_dump_size_per_path: Maximum dump size per path
   :return: Dictionary with subreddit names as keys and list of post indexes as values
   """
   paths = glob(f"{pushift_path}/RS*.zst")
   paths.sort()
   posts_index = {subreddit: [[]] * len(paths) for subreddit in subreddits}
   for i, (file_id, line) in enumerate(tqdm(read_a_collection_of_zsts_line_by_line(paths))):
      if i % 1000000 == 0:
         print(i, sum([len(f) for v in posts_index.values() for f in v]))
      j = json.loads(line)
      sub_name = j["subreddit"]
      if sub_name not in subreddits or len(posts_index[sub_name][file_id]) > max_dump_size_per_path:
         continue
      title, body, score = j["title"], j["selftext"], j["score"]
      if len(body.split()) < 10 and len(title.split()) < 10 or title == "[deleted by user]" or body == "[deleted]" or int(score) < min_score:
         continue
      posts_index[sub_name][file_id].append(i)
   return posts_index


def collect_posts_from_zst_files(pushift_path: str, posts_index: Set[int]) -> Dict[str, List[Tuple[str, str, str]]]:
   """
   This function collects posts from .zst files and saves them in a pickle file.
   :param posts_index: Set of post indexes
   :param out_path: Output path for the pickle file
   """
   paths = glob(f"{pushift_path}/RS*.zst")
   paths.sort()
   posts = defaultdict(list)
   for i, (file_id, line) in enumerate(tqdm(read_a_collection_of_zsts_line_by_line(paths))):
      if i % 1000000 == 0:
         print(i, len(posts))
      if i not in posts_index:
         continue
      j = json.loads(line)
      sub_name, idx, title, body = j["subreddit"], j["id"], j["title"], j["selftext"]
      posts[sub_name].append((idx, title, body))
   return posts
   


def main(args):
   # loading the subreddits we are interested in
   subreddits = open("outputs/subreddits.txt").read().split("\n")
   subreddits = set([s.strip() for s in subreddits])
   
   posts_index = collect_posts_index_from_zst_files(args.path_to_pushshift ,subreddits, min_score=10)
   posts_index = {subreddit: sum(files, start=[]) for subreddit, files in posts_index.items()}
   pickle.dump(posts_index, open(f"{args.save_path}/posts_index_refined.pkl", "wb"))
   
   # subsampling 1000 posts per subreddit
   sampled_post_indexes = set()
   for subreddit in posts_index:
      if len(posts_index[subreddit]) > 1000:
         sampled_post_indexes.update(set(np.random.choice(posts_index[subreddit], 1000, replace=False)))
      else:
         sampled_post_indexes.update(set(posts_index[subreddit]))

   posts = collect_posts_from_zst_files(args.path_to_pushshift, sampled_post_indexes)
   pickle.dump(posts, open(f"{args.save_path}/1000_posts_per_subreddit.pkl", "wb"))


if __name__ == "__main__":
   args = arg_parser()
   main(args)
    

