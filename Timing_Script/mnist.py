import dbscan
import load_mnist
import argparse
import time


#######################
#Side Note: I had to specify the data_dir in the data loding script instead of parameterize it, because
#the it seems that the parser can only parse for the main() function in this script



parser = argparse.ArgumentParser()
parser.add_argument("epsilon", type=float, help="DBSCAN Parameter epsilon")
parser.add_argument("min_points", type=int, help="DBSCAN Parameter min_points")
#parser.add_argument("distance_func", type=dbscan.Callable, help="Default is l2_distance func")
args = parser.parse_args()

eps = args.epsilon
min_pts = args.min_points



def main() -> None:
    X = load_mnist.main()
    start_time = time.time()
    m = dbscan.DBScan(epsilon=eps, min_points = min_pts)
    m.train(X)
    print(X)
    print(m.assignments, m.num_clusters)
    print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == "__main__":
    main()
