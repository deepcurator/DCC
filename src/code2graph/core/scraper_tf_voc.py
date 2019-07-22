import requests
import json
import networkx as nx
from bs4 import BeautifulSoup

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


class TFVocScraper:

    def __init__(self, version):
        self.version = version
        self.tf_types_root_url = "https://www.tensorflow.org/versions/%s/api_docs/python/tf" % self.version

        # print(self.tf_types_root_url)
        self.root = {}

        self.cached_json_path = Path(
            '..')/"tmp"/("tf_types_%s.json" % self.version)
        # '../tmp/graphs/test.graphml'
        self.cached_graph_path = Path(
            '..')/"tmp"/("tf_types_%s.graphml" % self.version)

        if self.cached_json_path.exists():
            with open(str(self.cached_json_path), 'r') as f:
                self.root = json.load(f)
        else:
            self.scrape_tf_website()

    def scrape_tf_website(self):

        print("scraping from the website")

        self.root = {"name": "root", "children": [],
                     "url": self.tf_types_root_url}

        tf_type_html = BeautifulSoup(requests.get(
            self.tf_types_root_url, timeout=5).content, 'html.parser')

        table_data = tf_type_html.find("ul", "devsite-nav-list", menu="_book")

        list_data = table_data.find_all(
            "li", {"class": "devsite-nav-item devsite-nav-expandable"}, recursive=False)

        for data in list_data:

            name = data.find("span").text.strip()

            if '.' in name:
                for child in self.root['children']:
                    if child['name'] == 'tf':
                        child['children'].append(
                            self.recur_scrape_tf_itemlist(data, name))
                        break

            else:
                print(name)
                assert (name == 'tf' or name == 'tfdbg')

                self.root['children'].append(
                    self.recur_scrape_tf_itemlist(data, name))

        print("saving to the cached file: %s" %
              str(self.cached_json_path.absolute()))

        with open(str(self.cached_json_path), 'w') as f:
            json.dump(self.root, f)

    def recur_scrape_tf_itemlist(self, cur_data, base_name):

        table = cur_data.find("ul", "devsite-nav-section")

        list_data = table.find_all(
            "li", {"class": "devsite-nav-item"}, recursive=False)

        node = {"name": base_name}

        children = []

        for item_data in list_data:
            classes = item_data['class']

            item_name = item_data.find(
                "span", {"class": "devsite-nav-text"}).text.strip()
            full_name = base_name + '.' + item_name

            if "devsite-nav-expandable" in classes:
                children.append(
                    self.recur_scrape_tf_itemlist(item_data, full_name))

            else:  # devsite-nav-item only case

                url = item_data.find("a").get('href')

                if item_name == "Overview":  # pseudo leaf
                    node["url"] = url

                else:
                    leaf_node = {"name": full_name, "url": url}
                    children.append(leaf_node)

        if len(children) > 0:
            node["children"] = children

        return node

    def dump_tree(self):
        self.recur_dump_tree(self.root, 0)

    def recur_dump_tree(self, node, num_of_tabs):
        print("--"*num_of_tabs + node['name'])
        if 'children' in node:
            [self.recur_dump_tree(c, num_of_tabs+1) for c in node['children']]

    def gen_graphml(self):  # use gephi to open the graphml
        self.G = nx.Graph()
        self.recur_gen_graphml(self.root)

        nx.write_graphml(self.G, str(self.cached_graph_path))

    def recur_gen_graphml(self, node):
        if 'children' in node:
            for child in node['children']:
                self.G.add_edge(node['name'], child['name'])
                self.recur_gen_graphml(child)


if __name__ == "__main__":

    scraper = TFVocScraper("r1.14")
    scraper.dump_tree()
    scraper.gen_graphml()
