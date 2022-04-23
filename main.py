from preprocessor import Preprocessor
from linear_regression import linearRegression
from knn_regression import knnRegression

MARGIN = 365*5
TEST_SIZE = 0.25
def runPreprocessor(path, table_path):
    preprocessor = Preprocessor(path, table_path)
    data = preprocessor.Encode()
    return data

def runLinearRegression(data, x_col, y_col):
    print('Running LinearRegression...')
    regressor = linearRegression(data, x_col, y_col, MARGIN, TEST_SIZE)
    regressor.runTest()

def runKNNRegression(data, x_col, y_col):
    print('Running KNNRegression...')
    regressor = knnRegression(data, x_col, y_col, MARGIN, TEST_SIZE)
    regressor.runTest()
    
def runAliveTest(data, x_col, y_col):
    y_col_bin = 'binary'+y_col
    _data = data[data[y_col_bin] == 0]
    print('-------- Test for alive data ---------------')
    runLinearRegression(_data, x_col, y_col)
    runKNNRegression(_data, x_col, y_col)

def runDeadTest(data, x_col, y_col):
    y_col_bin = 'binary'+y_col
    _data = data[data[y_col_bin] == 1]
    print('-------- Test for dead data ---------------')
    runLinearRegression(_data, x_col, y_col)
    runKNNRegression(_data, x_col, y_col)

def runAggregatedTest(data, x_col, y_col):
    print('-------- Test for aggregated data ---------------')
    runLinearRegression(data, x_col, y_col)
    runKNNRegression(data, x_col, y_col)
                 
if __name__ == '__main__':
    data = runPreprocessor('OppScrData.csv', 'ssa_life_expectancy.csv')
    x_col = ['L1_HU_BMD', 'Total Body                Area EA (cm2)', 'SAT Area (cm2)', 'VAT/SAT     Ratio', 'Muscle HU', 'L3 SMI (cm2/m2)', 'AoCa        Agatston', 'Liver HU    (Median)']
    y_col = '_DEATH [d from CT]'

    runAliveTest(data, x_col, y_col)
    runDeadTest(data, x_col, y_col)
    runAggregatedTest(data, x_col, y_col)
