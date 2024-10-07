import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwizeDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:

        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """

        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dict_func: function reference
        """

        dist_func = None

        def euclidiean(N, copy_data):
            distance_matrix = np.zeros(shape=(N, N))
            for i in range(N):
                for j in range(i, N):
                    if self.is_normalize:
                        distance_matrix[i,j] = norm_ED_distance(copy_data[i], copy_data[j])
                    else:
                        distance_matrix[i,j] = ED_distance(copy_data[i],  copy_data[j])
            for i in range(N):
                for j in range(i, N):
                    distance_matrix[j,i] = distance_matrix[i,j]
            return distance_matrix

        def dtw(N, copy_data):
            distance_matrix = np.zeros(shape=(N, N))
            for i in range(N):
                for j in range(i, N):
                    if self.is_normalize:
                        distance_matrix[i,j] = DTW_distance(z_normalize(copy_data[i]),  z_normalize(copy_data[j]))
                    else:
                        distance_matrix[i,j] = DTW_distance(copy_data[i],  copy_data[j])
            for i in range(N):
                for j in range(i, N):
                    distance_matrix[j,i] = distance_matrix[i,j]
            return distance_matrix

        if self.metric == 'euclidean':
            dist_func = euclidiean
        elif self.metric == 'dtw':
            dist_func = dtw
        return dist_func


    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        
        Returns
        -------
        matrix_values: distance matrix
        """
        
        matrix_shape = (input_data.shape[0], input_data.shape[0])
        matrix_values = np.zeros(shape=matrix_shape)
        
        N = input_data.shape[0] # number of time series
  
        copy_data = input_data.copy()

        dist_func = self._choose_distance()
        
        matrix_values = dist_func(N, copy_data)

        return matrix_values