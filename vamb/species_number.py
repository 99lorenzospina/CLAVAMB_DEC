"""!

@brief The module contains G-Means algorithm and other related services. https://pyclustering.github.io/docs/0.9.2/html/d8/dd0/gmeans_8py_source.html
@details Implementation based on paper @cite inproceedings::cluster::gmeans::1.

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2019
@copyright GNU Public License

@cond GNU_PUBLIC_LICENSE
    PyClustering is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyClustering is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@endcond

"""


import numpy
import scipy.stats

from pyclustering.core.gmeans_wrapper import gmeans as gmeans_wrapper
from pyclustering.core.wrapper import ccore_library

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils import distance_metric, type_metric


class SpeciesNumber:
    """!
    @brief Class implements G-Means clustering algorithm.
    @details The G-means algorithm starts with a small number of centers, and grows the number of centers.
            Each iteration of the G-Means algorithm splits into two those centers whose data appear not to come from a
            Gaussian distribution. G-means repeatedly makes decisions based on a statistical test for the data
            assigned to each center.

    Implementation based on the paper @cite inproceedings::cluster::gmeans::1.

    @image html gmeans_example_clustering.png "G-Means clustering results on most common data-sets."

    Example #1. In this example, G-Means starts analysis from single cluster.
    @code
        from pyclustering.cluster import cluster_visualizer
        from pyclustering.cluster.gmeans import gmeans
        from pyclustering.utils import read_sample
        from pyclustering.samples.definitions import FCPS_SAMPLES

        # Read sample 'Lsun' from file.
        sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)

        # Create instance of G-Means algorithm. By default algorithm start search from single cluster.
        gmeans_instance = gmeans(sample).process()

        # Extract clustering results: clusters and their centers
        clusters = gmeans_instance.get_clusters()
        centers = gmeans_instance.get_centers()

        # Print total sum of metric errors
        print("Total WCE:", gmeans_instance.get_total_wce())

        # Visualize clustering results
        visualizer = cluster_visualizer()
        visualizer.append_clusters(clusters, sample)
        visualizer.show()
    @endcode

    Example #2. Sometimes G-Means may found local optimum. 'repeat' value can be used to increase probability to
    find global optimum. Argument 'repeat' defines how many times K-Means clustering with K-Means++
    initialization should be run to find optimal clusters.
    @code
        # Read sample 'Tetra' from file.
        sample = read_sample(FCPS_SAMPLES.SAMPLE_TETRA)

        # Create instance of G-Means algorithm. By default algorithm start search from single cluster.
        gmeans_instance = gmeans(sample, repeat=10).process()

        # Extract clustering results: clusters and their centers
        clusters = gmeans_instance.get_clusters()

        # Visualize clustering results
        visualizer = cluster_visualizer()
        visualizer.append_clusters(clusters, sample)
        visualizer.show()
    @endcode

    """

    def __init__(self, data, k_init=1, ccore=True, **kwargs):
        """!
        @brief Initializes G-Means algorithm.

        @param[in] data (array_like): Input data that is presented as array of points (objects), each point should be
                    represented by array_like data structure.
                    in my case, data is a np array, each row the tnf+abudance for a contig
        @param[in] k_init (uint): Initial amount of centers (by default started search from 1).
        @param[in] ccore (bool): Defines whether CCORE library (C/C++ part of the library) should be used instead of
                    Python code.
        @param[in] **kwargs: Arbitrary keyword arguments (available arguments: 'tolerance', 'repeat').

        <b>Keyword Args:</b><br>
            - tolerance (double): tolerance (double): Stop condition for each K-Means iteration: if maximum value of
            change of centers of clusters is less than tolerance than algorithm will stop processing.
            - repeat (unit): How many times K-Means should be run to improve parameters (by default is 3).
            With larger 'repeat' values suggesting higher probability of finding global optimum.

        """
        self.__data = data
        self.__k_init = k_init

        self.__clusters = []
        self.__centers = []
        self.definitive_centers = []
        self.__total_wce = 0.0
        self.__ccore = ccore
        self.nclusters = 1

        self.__tolerance = kwargs.get('tolerance', 0.025)
        self.__repeat = kwargs.get('repeat', 3)

        if self.__ccore is True:
            self.__ccore = ccore_library.workable()

        self._verify_arguments()


    def process(self):
        """!
        @brief Performs cluster analysis in line with rules of G-Means algorithm.

        @return (gmeans) Returns itself (G-Means instance).

        @see get_clusters()
        @see get_centers()

        """
        if self.__ccore is True:
            return self._process_by_ccore()

        return self._process_by_python()


    def _process_by_ccore(self):
        """!
        @brief Performs cluster analysis using CCORE (C/C++ part of pyclustering library).

        """
        self.__clusters, self.__centers, self.__total_wce = gmeans_wrapper(self.__data, self.__k_init, self.__tolerance, self.__repeat)
        return self


    def _process_by_python(self):
        """!
        @brief Performs cluster analysis using Python.

        """
        self.__clusters, self.__centers, _ = self._search_optimal_parameters(self.__data, self.__k_init)
        while True:
            added = self._statistical_optimization()

            if not added:
                break

            self._perform_clustering()

        self.nclusters = len(self.definitive_centers)
        return self

    def estimate_k(self):
        """!
            @brief Return the value of k.
            
            @return (int) Number of clusters

        """
        return self.nclusters

    def predict(self, points):
        """!
        @brief Calculates the closest cluster to each point.

        @param[in] points (array_like): Points for which closest clusters are calculated.

        @return (list) List of closest clusters for each point. Each cluster is denoted by index. Return empty
                collection if 'process()' method was not called.

        """
        nppoints = numpy.array(points)
        if len(self.__clusters) == 0:
            return []

        metric = distance_metric(type_metric.EUCLIDEAN_SQUARE, numpy_usage=True)

        differences = numpy.zeros((len(nppoints), len(self.__centers)))
        for index_point in range(len(nppoints)):
            differences[index_point] = metric(nppoints[index_point], self.__centers)

        return numpy.argmin(differences, axis=1)


    def get_clusters(self):
        """!
        @brief Returns list of allocated clusters, each cluster contains indexes of objects in list of data.

        @return (array_like) Allocated clusters.

        @see process()
        @see get_centers()

        """
        return self.__clusters


    def get_centers(self):
        """!
        @brief Returns list of centers of currently elaborating clusters.

        @return (array_like) Allocated centers.

        @see process()
        @see get_clusters()

        """
        return self.__centers

    def get_definitive_centers(self):
        """!
        @brief Returns list of centers of definitvely allocated clusters.
        
        @return (array_like) Allocated centers.
        
        """
        return self.definitive_centers


    def get_total_wce(self):
        """!
        @brief Returns sum of metric errors that depends on metric that was used for clustering (by default SSE - Sum of Squared Errors).
        @details Sum of metric errors is calculated using distance between point and its center:
                \f[error=\sum_{i=0}^{N}distance(x_{i}-center(x_{i}))\f]

        @see process()
        @see get_clusters()

        """

        return self.__total_wce


    def _statistical_optimization(self):
        """!
        @brief Try to split cluster into two to find optimal amount of clusters.

        """
        centers = []
        added = False
        for index in range(len(self.__clusters)):
            #either new_centers contains the new centers or the points of the cluster
            new_centers, split = self._split_and_search_optimal(self.__clusters[index])
            if not split:  #the cluster is not split!
                self.definitive_centers.append(self.__centers[index])
                points_to_remove = new_centers
                indices_to_keep = numpy.setdiff1d(numpy.arange(len(self.__data)), numpy.where(numpy.isin(self.__data, points_to_remove).all(axis=1)))
                self.__data = self.__data[indices_to_keep]
            else:
                centers += new_centers
                added = True

        self.__centers = centers
        return added


    def _split_and_search_optimal(self, cluster):
        """!
        @brief Split specified cluster into two by performing K-Means clustering and check correctness by
                Kolmogorov-Smirnov test.

        @param[in] cluster (array_like) Cluster that should be analyzed and optimized by splitting if it is required.

        @return (array_like) Two new centers if two new clusters are considered as more suitable.
                (None) If current cluster is more suitable.
        """

        points = [self.__data[index_point] for index_point in cluster]
        if len(cluster) == 1:
            return points, False
        _, new_centers, _ = self._search_optimal_parameters(points, 2)


        if len(new_centers) > 1:
            v = numpy.subtract(new_centers[0], new_centers[1])
            square_norm = numpy.sum(numpy.multiply(v, v))
            points = numpy.divide(numpy.sum(numpy.multiply(new_centers, v), axis=1), square_norm)
            normalized_cluster_data = (points - numpy.mean(points, axis=0)) / numpy.std(points, axis=0)
            sorted_data = numpy.sort(normalized_cluster_data, axis=0)
            cdf = scipy.stats.norm.cdf(sorted_data)
            ecdf = numpy.arange(1, len(sorted_data) + 1) / len(sorted_data)
            KS = self.ks(ecdf, cdf)
            accept_null_hypothesis = KS.pvalue > 0.05
            if not accept_null_hypothesis:
                return new_centers, True   # If null hypothesis is rejected then use two new clusters

        return points, False

    def ks(self, ecdf, cdf):
        
        return scipy.stats.ks_2samp(ecdf, cdf)

    def _search_optimal_parameters(self, data, amount):
        """!
        @brief Performs cluster analysis for specified data several times to find optimal clustering result in line
                with WCE.

        @param[in] data (array_like): Input data that should be clustered.
        @param[in] amount (unit): Amount of clusters that should be allocated.

        @return (tuple) Optimal clustering result: (clusters, centers, wce).

        """
        best_wce, best_clusters, best_centers = float('+inf'), [], []
        for _ in range(self.__repeat):
            initial_centers = kmeans_plusplus_initializer(data, amount).initialize()
            solver = kmeans(data, initial_centers, tolerance=self.__tolerance, ccore=False).process()

            candidate_wce = solver.get_total_wce()
            if candidate_wce < best_wce:
                best_wce = candidate_wce
                best_clusters = solver.get_clusters()
                best_centers = solver.get_centers()

            if len(initial_centers) == 1:
                break   # No need to rerun clustering for one initial center.

        return best_clusters, best_centers, best_wce


    def _perform_clustering(self):
        """!
        @brief Performs cluster analysis using K-Means algorithm using current centers are initial.

        @param[in] data (array_like): Input data for cluster analysis.

        """
        solver = kmeans(self.__data, self.__centers, tolerance=self.__tolerance, ccore=False).process()
        self.__clusters = solver.get_clusters()
        self.__centers = solver.get_centers()
        self.__total_wce = solver.get_total_wce()


    def _verify_arguments(self):
        """!
        @brief Verify input parameters for the algorithm and throw exception in case of incorrectness.

        """
        if len(self.__data) == 0:
            raise ValueError("Input data is empty (size: '%d')." % len(self.__data))

        if self.__k_init <= 0:
            raise ValueError("Initial amount of centers should be greater than 0 "
                            "(current value: '%d')." % self.__k_init)

        if self.__tolerance <= 0.0:
            raise ValueError("Tolerance should be greater than 0 (current value: '%f')." % self.__tolerance)

        if self.__repeat <= 0:
            raise ValueError("Amount of attempt to find optimal parameters should be greater than 0 "
                            "(current value: '%d')." % self.__repeat)