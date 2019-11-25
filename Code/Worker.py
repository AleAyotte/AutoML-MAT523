"""
    @file:              Worker.py
    @Author:            Nicolas Raymond
    @Creation Date:     25/11/2019
    @Last modification: 25/11/2019
    @Description:       This file the worker object needed for hyper-parameter optimization algorithm from
                        HpBandster library.
"""

import time

from hpbandster.core.worker import Worker


class MyWorker(Worker):

    def __init__(self, *args, loss_function, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.loss_function = loss_function

    def compute(self, hyperparams, **kwargs):

        """

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            hyperparams: dictionary containing the sampled configurations by the optimizer

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        res = self.loss_function(hyperparams)
        time.sleep(self.sleep_interval)

        return ({
            'loss': res,  # this is the a mandatory field to run hyperband
            'info': [res]  # can be used for any user-defined information - also mandatory
        })
