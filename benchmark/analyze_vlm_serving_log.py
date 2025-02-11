import re

import fire
import numpy as np


def analyze_preprocess(content):
    pattern = r'preprocessing cost (\d+\.\d+) s, image_size \((\d+, \d+)\), image_token (\d+)'

    matches = re.findall(pattern, content)
    results = []
    for match in matches:
        cost = float(match[0])
        image_size = tuple(map(int, match[1].split(', ')))
        image_token = int(match[2])
        results.append((cost, image_size, image_token))
    ave_cost = np.average([x for x, _, _ in results])
    ave_image_token = np.average([x for _, _, x in results])
    print(f'average image token: {int(ave_image_token)}')
    print(f'preprocessing average cost: {ave_cost:.3f} s')


def analyze_vision_forward(content):
    pattern = r'forward cost (\d+\.\d+) s'
    matches = re.findall(pattern, content)
    costs = [float(match) for match in matches]
    ave_cost = np.average(costs)
    print(f'vision forward average cost: {ave_cost:.3f} s')


def main(log_file):
    with open(log_file, 'r') as f:
        content = f.read()

        analyze_preprocess(content)
        analyze_vision_forward(content)


if __name__ == '__main__':
    fire.Fire(main)
