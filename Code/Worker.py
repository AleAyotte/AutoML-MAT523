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
import os
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import socket
from hpbandster.optimizers import BOHB
from hpbandster.optimizers import HyperBand
from hpbandster.core.worker import Worker


class MyWorker(Worker):

    def __init__(self, *args, loss_function, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.loss_function = loss_function

    def compute(self, config, **kwargs):

        """

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        res = self.loss_function(config)
        time.sleep(self.sleep_interval)

        return ({
            'loss': res,  # this is the a mandatory field to run hyperband
            'info': {'res':res}  # can be used for any user-defined information - also mandatory
        })


def start_hpbandster_process(method, configspace, loss):

    """
    Starts a server and a worker object needed for the HpBandSter optimization process

    :param method: (str) Specifies if we use BOHB or hyperband
    :param configspace: Hyper-parameter search space
    :param loss: Loss function to minimize
    :return: NameServer and HpBandSter optimizer
    """

    # Set the host for the process
    host = socket.gethostbyname('localhost')

    # This line is most useful for really long runs, where intermediate results could already be
    # interesting. The core.result submodule contains the functionality to read the two generated
    # files (results.json and configs.json) and create a Result object.
    #result_logger = hpres.json_result_logger(directory=os.getcwd(), overwrite=True)

    # Start a nameserver:
    NS = hpns.NameServer(run_id=method, host=host)
    ns_host, ns_port = NS.start()

    # Start local worker
    w = MyWorker(run_id=method, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120, loss_function=loss)
    w.run(background=True)

    if method == 'BOHB':

        optimizer = BOHB(configspace=configspace,
                         run_id=method,
                         host=host,
                         nameserver=ns_host,
                         nameserver_port=ns_port,
                         min_budget=1, max_budget=9,
                         )
    else:

        optimizer = HyperBand(configspace=configspace,
                              run_id=method,
                              host=host,
                              nameserver=ns_host,
                              nameserver_port=ns_port,
                              min_budget=1, max_budget=9,
                              )

    return NS, optimizer
