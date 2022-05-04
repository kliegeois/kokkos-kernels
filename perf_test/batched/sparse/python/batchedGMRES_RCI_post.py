import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Postprocess the results.')
    parser.add_argument('--timer_filename', metavar='timer_filename', default='',
                        help='used timer_filename')
    parser.add_argument('--implementation', metavar='implementation', default='',
                        help='used implementation')
    parser.add_argument('--layout', metavar='layout', default='',
                        help='used layout')
    args = parser.parse_args()

    if args.layout == 'Left':
        time_name = '_left.txt'
    if args.layout == 'Right':
        time_name = '_right.txt'

    tmp = np.loadtxt(args.timer_filename+'_'+str(args.implementation)+time_name)
    print(np.mean(tmp))


if __name__ == "__main__":
 
	main()