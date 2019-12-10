"""
    @file:              Worker.py
    @Author:            Nicolas Raymond
    @Creation Date:     25/11/2019
    @Last modification: 25/11/2019
    @Description:       This file provides the worker object needed for hyper-parameter optimization algorithm from
                        HpBandster library. It also provides functions to setup the environment before using
                        HpBandster.
"""

import time
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.optimizers import HyperBand
from hpbandster.core.worker import Worker
from numpy import log

class MyWorker(Worker):

    def __init__(self, *args, loss_function, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.loss_function = loss_function

    def compute(self, config, budget, **kwargs):

        """

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: number of epoch allowed the iteration

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        res = self.loss_function(config, budget)
        time.sleep(self.sleep_interval)

        return ({
            'loss': res,  # this is a mandatory field to run hyperband
            'info': config  # can be used for any user-defined information - also mandatory
        })


def start_hpbandster_process(method, configspace, loss, total_budget, max_budget_per_config, eta=3):

    """
    Starts a server and a worker object needed for the HpBandSter optimization process

    :param method: (str) Specifies if we use BOHB or hyperband
    :param configspace: Hyper-parameter search space
    :param loss: Loss function to minimize
    :param total_budget: Total budget (in number of epochs) allowed for optimization
    :param max_budget_per_config: Maximal number of epochs allowed for one config
    :param eta: split size between every steps of successful halving
    :return: NameServer and HpBandSter optimizer
    """

    # Start a nameserver:
    NS = hpns.NameServer(run_id=method)
    ns_host, ns_port = NS.start()

    # Start local worker
    w = MyWorker(run_id=method, nameserver=ns_host, nameserver_port=ns_port, timeout=120, loss_function=loss)
    w.run(background=True)

    if method == 'BOHB':

        optimizer = BOHB(configspace=configspace,
                         run_id=method,
                         nameserver=ns_host,
                         nameserver_port=ns_port,
                         min_budget=1, max_budget=max_budget_per_config,
                         )
    else:

        optimizer = HyperBand(configspace=configspace,
                              run_id=method,
                              nameserver=ns_host,
                              nameserver_port=ns_port,
                              min_budget=1, max_budget=max_budget_per_config,
                              )

    # We compute the maximal number of iteration to be exact with the original paper
    # (We divide the total budget by the fixed budget per successful halving iteration : (Smax+1)*bmax)
    max_iter = total_budget/(int(-1*(log(1/max_budget_per_config))/log(eta) + 1)*max_budget_per_config)

    return NS, max_iter, optimizer
