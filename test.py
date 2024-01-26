import pandas_ta as ta
import pandas as pd
from pandas_ta import core
e = pd.DataFrame()

core_indicators = e.ta.indicators(as_list=True)
fun = getattr(core, core_indicators[0])

e.ta.PVOL()