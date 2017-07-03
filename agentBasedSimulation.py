import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys

class AgentBasedSimulation:

    def processInputData(self,dataFilePath):
        """Returns processed source file provided by Simudyne.
        param: dataFilePath
        This method processes the data and does precalculations. Computes a_1,a_2,sc_gb
        and appends to the dataframe that it then returned.
        """
        assert isinstance(dataFilePath, str), "%r must be a string path to the data." \
                                              % dataFilePath
        assert dataFilePath.endswith(".csv"), "%s must be a path to .csv file" % dataFilePath
        # Potential for extension. For example, support of other file formats.
        # Another potential for extension: checks that the file is of a valid format. For example,
        # that the read pandas dataFrame has the correct number of columns, and that the names are correct.
        # That the dataframe is not of zero size, i.e. that there is data.
        # A more expensive check would also be to ensure that the data integrity is correct. i.e.
        # that you do not have any outliers or inconsitencies, and/or that you do not have nulls etc.

        try:
            inputData = pd.read_csv(dataFilePath)
            # Get rid of auto-renew data as the breed for these does not change.
            inputData = inputData.loc[inputData.Auto_Renew != 1]
            inputData.reset_index(inplace=True)

            # Pre-calculations
            inputData.loc[:, "a1"] = inputData.Payment_at_Purchase / inputData.Attribute_Price
            inputData.loc[:, "a2"] = inputData.Attribute_Promotions * inputData.Inertia_for_Switch
            inputData.loc[:, "sg_ab"] = inputData.Social_Grade * inputData.Attribute_Brand

            for i in range(15):
                inputData.loc[:, "Y_" + str(i + 1)] = [0]
            return inputData
        except IOError as e:  # For example there is no such file in the directory.
            print(e)
        except:
            print("Unexpected error")  # Potential for extension: better error catching.

    def generateBrandFactorVector(self, lowerBound, upperBound, size):
        return np.random.uniform(lowerBound, upperBound, size)

    def runModel(self,brand_factor, inputData, verbose=0):
        """Runs the agent-based simulation.
        param: verbose. if 1, then. if 2, then
        """
        # Potential improvement: if the len(brand_factor) is large enough,
        # we can compute expected value: np.mean(brand_factor) and ensure
        # that it is 1/2*(lowerBound+UpperBound) within some epsillon.

        # It is not necessary to increment age as it is not used in any calculations.
        # If however, it is required in the model output, then the calculation is trivial
        # depending on the year that we are on.

        for i in range(15): self.calculateEpochs(i, brand_factor, inputData)

    def calculateEpochs(self,year, brand_factor, data):
        """Calculated annual simulation epochs.
        param: TODO
        param:
        param:
        param:
        param:
        """
        if (year == 0):
            currentColumn = 'Agent_Breed'
            nextColumn = 'Y_1'
        else:
            currentColumn = 'Y_' + str(year)
            nextColumn = 'Y_' + str(year + 1)
        breedColumns = {"currCol": currentColumn, "nextCol": nextColumn}

        breedCMap = {True: "Breed_NC_Changed",
                     False: "Breed_C_Unchanged"}  # Why isn't False: Breed_C? Because we are also looping through Breed NC!
        breedNCMap = {True: "Breed_C_Changed", False: "Breed_NC_Unchanged"}  # Ditto.
        breedMaps = {"breedCMap": breedCMap,
                     "breedNCMap": breedNCMap}  # not using list, because easier to use a wrong index

        # Those that switched and those that didn't
        data.loc[:, breedColumns["nextCol"]] = data.groupby(breedColumns["currCol"]).apply(lambda group: \
                        self.applyMutations(group,brand_factor,breedColumns,breedMaps))

    def applyMutations(groupDf, brand_factor, breedColumns, breedMaps):
        # arguments are a little bit all over the place. brand_factor, for example can be calced here.

        breed = np.unique(groupDf[breedColumns['currCol']])[
            0]  # Probably unnecessary. There should be a way to access group id.

        rand = np.random.uniform(0, 3, len(groupDf))
        affinity = groupDf.a1 + pd.Series(rand) * groupDf.a2

        # applies the maps to both breed_c and breed_nc twice due to the fact that there are
        # (breed_c_changed and breed_c_unchaged) as well as (breed_nc_changed) and (breed_nc_unchanged)
        # to circumvent in the future, use the C++ mock pointer class from the above to keep track
        # whether you have already updated the dataFrame. So you would have two Ref classes, one
        # for breed_c and one for breed_nc.
        if (breed.startswith("Breed_C")):
            columnWithCCheck = (groupDf[breedColumns["currCol"]].str.startswith(
                "Breed_C"))  # instead of making 3 checks: == "Breed_C" or == "Breed_C_Changed" or == "Breed_C_Unchanged"
            maskSwitchToNC = columnWithCCheck & (
            affinity < groupDf.sg_ab)  # Potential for improvement. Perform checks on the sliced of the DataFrame

            return maskSwitchToNC.map(breedMaps["breedCMap"])
        elif (breed.startswith("Breed_NC")):
            columnWithNCCheck = (groupDf[breedColumns["currCol"]].str.startswith("Breed_NC"))
            maskSwitchToC = columnWithNCCheck & (affinity < pd.Series(brand_factor) * groupDf.sg_ab)

            return maskSwitchToC.map(breedMaps["breedNCMap"])
        else:
            raise ValueError('Unknown breed %s. Can start with "Breed_C" or "Breed_NC".' % (breed))

if __name__ == "__main__":
    dataFilePath = "C:/Users/Nazariy/Desktop/Simudyne_Backend_Test.csv"
    saveFilePath = "C:/Users/Nazariy/Desktop/Simudyne_Bacend_Test_Output.csv"
    lb_BrandFactor = 0.1
    ub_BrandFactor = 2.9

    inputData = processInputData2(dataFilePath)
    inputDataSize = len(inputData)

    start = time.time()
    runModel(generateBrandFactorVector(lb_BrandFactor, ub_BrandFactor, inputDataSize), inputData)
    end = time.time()

    print(inputData)