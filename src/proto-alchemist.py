import sys
import os
import glob
import cv2
import numpy as np
import argparse

ERASE_LINE = '\x1b[2K'

def find_base(neg, print_progress=False):
    white_sample = [0, 0]
    previous_max = 0

    for y in range(neg.shape[1]):
        if print_progress:
            progress = y/neg.shape[1] * 100
            if progress.is_integer():
                print(
                    ERASE_LINE + f'Searching for base... {progress} %',
                    end='\r'),
        for x in range(neg.shape[0]):
            local_max = 0
            for chan in range(neg.shape[2]):
                local_max += neg.item(x, y, chan)
            if local_max > previous_max:
                previous_max = local_max
                white_sample = [x, y]

    return [ 
        neg.item(white_sample[0], white_sample[1], 0),
        neg.item(white_sample[0], white_sample[1], 1),
        neg.item(white_sample[0], white_sample[1], 2)
    ]

def invert(neg, base, print_progress=False):
    if print_progress:
            print(
                ERASE_LINE + 'Removing orange mask...', end='\r'),

    # remove orange mask
    b,g,r = cv2.split(neg)
    b = b * (1 / base[0])
    g = g * (1 / base[1])
    r = r * (1 / base[2])
    res = cv2.merge((b,g,r))

    if print_progress:
            print(
                ERASE_LINE + 'Inverting...', end='\r'),

    # invert
    res = 1 - res

    if print_progress:
            print(
                ERASE_LINE + 'Normalizing...', end='\r'),
    res = cv2.normalize(res, None, 0.0, 1.0, cv2.NORM_MINMAX)

    return res

if __name__ == '__main__':

    # Build the argument parser

    parser = argparse.ArgumentParser(
        description='Convert a color negative image to a positive.')

    parser.add_argument('sources',
        nargs='+',
        help='The linear scan of the film or a glob pattern describing '
        'multiple files to treat')
    
    parser.add_argument('--base', '-b',
        dest='base',
        default=None,
        help='An image containing colours to be averaged to describe the '
        'neutral base. Omitting this will cause Spice/Alchemist to search for '
        'the lightest pixel in the negative and assume that it is the base '
        'colour.')

    parser.add_argument('--show', '-s',
        action='store_true',
        help='Show the converted image.')
    
    parser.add_argument('--output', '-o',
        help='Save the converted image to this directory using the name of the '
        'input file. Either this or --show has to be specified.')
    
    parser.add_argument('--bit-depth', '-d',
        type=int,
        default=16,
        choices=[ 8, 16, 32 ],
        help='The bit depth to save the result with. '
        'Defaults to 16 bits per channel.')

    args = parser.parse_args()

    # Handle invalid input

    if not (args.show or args.output):
        print('Either --show or --output has to be specified.')
        sys.exit(-1)

    # Do the actual processing

    # Find the mean of the provided orange mask
    if args.base:
        base = cv2.imread(args.base, cv2.IMREAD_UNCHANGED)
        base = np.float32(base) / np.iinfo(base.dtype).max
        base = cv2.mean(base)
        print(f"Found base: {base}")

    if len(args.sources) == 1:
        files = glob.glob(args.sources[0])
    else:
        files = args.sources
    counter = 0
    for file in files:
        counter += 1
        print(f'Processing file {counter} of {len(files)}...')
        neg = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype('float')

        # fall back on searching for maximum pixel value in case of unspecified
        # base colour
        if not args.base:
            base = find_base(neg, True)

        positive = invert(neg, base, True)

        if args.output:
            filename = os.path.join(args.output, os.path.basename(file))
            os.makedirs(os.path.dirname(os.path.abspath(filename)),
                exist_ok=True)

            if args.bit_depth == 8:
                d_type = np.uint8
                maxval = np.iinfo(d_type).max
            elif args.bit_depth == 16:
                d_type = np.uint16
                maxval = np.iinfo(d_type).max
            elif args.bit_depth == 32:
                d_type = np.float32
                maxval = 1
            cv2.imwrite(
                filename,
                (np.clip(positive, 0, 1) * maxval).astype(d_type))

        if args.show:
            cv2.imshow(
                f'Proto Alchemist | {file} converted to positive',
                positive)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(ERASE_LINE + 'Done.')