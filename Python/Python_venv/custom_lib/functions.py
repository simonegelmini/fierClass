#   Authors: Jessica Leoni (jessica.leoni@polimi.it)
#            Francesco Zinnari (francesco.zinnari@polimi.it)
#            Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
#   Date: 2019/05/03.
#
#   If you are going to use fierClass in your research project, please cite its reference article
#   S. Gelmini, S. Formentin, et al. "fierClass: A multi-signal, cepstrum-based, time series classifier,"
#   Engineering Applications of Artificial Intelligence, Volume 87, 2020, https://doi.org/10.1016/j.engappai.2019.103262.
#
#   Copyright and license: © Jessica Leoni, Francesco Zinnari, Simone Gelmini, Politecnico di Milano
#   Licensed under the [MIT License](LICENSE).
#
#  In case of need, feel free to ask the author.

################################################################
#                           Libraries                          #
################################################################
import numpy as np


################################################################
#                    Low Pass Filtering Function               #
################################################################
def lowpass(a, b, c, d, fs, values):
    # Initial conditions setting
    T = 1/fs
    y_0 = 0
    u_0 = 0
    
    # Tustin discretization method: s = (2/T)*((z-1)/(z+1))
    output = []
    for i in range(len(values)):
        # Filter application
        y = (1/((2*c)+(d*T)))*(2*a*(values[i]
            +(b-2*a)*u_0)+y_0*(2*c-d*T))
        output.append(y)
        y_0 = y
        u_0 = values[i]
    output = np.squeeze(np.asarray(output))
    return output