from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
utils = importr('utils')

package_names = ("doMC", "Matrix", "data.table", "FNN", "foreach")
utils.install_packages(StrVector(package_names))