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

def average_base(base, print_progress=False):
    avg = [0 for chan in range(base.shape[2])]
    samples = 0

    for y in range(base.shape[1]):
        if print_progress:
            progress = y/base.shape[1] * 100
            if progress.is_integer():
                print(
                    ERASE_LINE + f'Averaging base... {progress} %',
                    end='\r'),
        for x in range(base.shape[0]):
            samples += 1
            for chan in range(neg.shape[2]):
                avg[chan] += base.item(x, y, chan)

    for chan_avg in avg:
        chan_avg = chan_avg / samples

    return avg

def invert(neg, base, print_progress=False):
    scale_b = 1 / base[0]
    scale_g = 1 / base[1]
    scale_r = 1 / base[2]

    # remove orange mask
    b,g,r = cv2.split(neg)
    b = b * scale_b
    g = g * scale_g
    r = r * scale_r
    res = cv2.merge((b,g,r))

    # invert
    # K = 0.00005
    # gamP = 2.2
    # gamC = 0.45
    # res = ((K / (res**gamP))**gamC)

    if print_progress:
            print(
                ERASE_LINE + 'Inverting...', end='\r'),
    res = 1 - res

    brightest = [0, 0, 0]

    for y in range(res.shape[1]):
        if print_progress:
            progress = y/res.shape[1] * 100
            if progress.is_integer():
                print(
                    ERASE_LINE + f'Scaling... {progress} %', end='\r'),
        for x in range(res.shape[0]):
            for chan in range(res.shape[2]):
                if res.item(x, y, chan) > res.item(
                    brightest[0],
                    brightest[1],
                    brightest[2]
                ):
                    brightest = [x, y, chan]

    res = res * (1/res.item(
                    brightest[0],
                    brightest[1],
                    brightest[2]
                ))

    return res

if __name__ == '__main__':

    # Build the argument parser

    parser = argparse.ArgumentParser(
        description='Convert a color negative image to a positive.')

    parser.add_argument('negative',
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
    
    # parser.add_argument('--color-space', '-c',
    #     default='ProphotoRGB',
    #     choices=[
    #         'ProphotoRGB',
    #         'AdobeRGB',
    #         'sRGB'
    #         ],
    #     help='[ UNIMPLEMENTED ] The colour space to save the result with. '
    #     'Defaults to "ProphotoRGB".')
    
    parser.add_argument('--bit-depth', '-d',
        type=int,
        default=16,
        choices=[ 8, 16, 32 ],
        help='[ UNIMPLEMENTED ] The bit depth to save the result with. '
        'Defaults to 16 bits per channel.')

    args = parser.parse_args()

    # Handle invalid input

    if not (args.show or args.output):
        print('Either --show or --output has to be specified.')
        sys.exit(-1)

    # if not os.path.isfile(args.negative):
    #     print(f'"{args.negative}" is not a file.')
    #     sys.exit(-1)

    # Do the actual processing

    files = glob.glob(args.negative)
    counter = 0
    for file in files:
        counter += 1
        print(f'Processing file {counter} of {len(files)}...')
        neg = cv2.imread(file)
        if args.base:
            base = average_base(cv2.imread(args.base), True)
        else:
            base = find_base(neg, True)

        positive = invert(neg, base, True)

        if args.output:
            filename = os.path.join(args.output, os.path.basename(file))
            os.makedirs(os.path.dirname(os.path.abspath(filename)),
                exist_ok=True)
            cv2.imwrite(
                filename,
                (np.clip(positive, 0, 1) * 65535).astype(np.uint16))

        if args.show:
            cv2.imshow(
                f'Proto Alchemist | {file} converted to positive',
                positive)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(ERASE_LINE + 'Done.')