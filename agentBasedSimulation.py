import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

class AgentBasedSimulation(object):

    def __init__(self):
        self.breedCMap  = {True: "Breed_NC_Changed", False: "Breed_C_Unchanged"}       # Maps to change the default pd's
        self.breedNCMap = {True: "Breed_C_Changed",  False: "Breed_NC_Unchanged"}      # mask values of True, False to
        self.breedMaps  = {"breedCMap": self.breedCMap, "breedNCMap": self.breedNCMap} # the indicated values.

    def processInputData(self, dataFilePath):
        """Returns processed source file. This method processes the data and does precalculations. Computes a_1,a_2,sc_gb
        and appends to the dataframe that is class variable.
        :param dataFilePath: the source file.
        :return: 
        """
        assert isinstance(dataFilePath, str), "%r must be a string path to the data." % dataFilePath
        assert dataFilePath.endswith(".csv"), "%s must be a path to .csv file"        % dataFilePath
        # Potential for extension. For example, support of other file formats.
        # Another potential for extension: checks that the file is of a valid format. For example,
        # that the read pandas dataFrame has the correct number of columns, and that the names are correct.
        # That the dataframe is not of zero size, i.e. that there is data.
        # A more expensive check would also be to ensure that the data integrity is correct. i.e.
        # that you do not have any outliers or inconsitencies, and/or that you do not have nulls etc.
        # Also ensure that there are no empty strings before the unique values of breed!

        try:
            self.data = pd.read_csv(dataFilePath)

            self.data = self.data.loc[self.data.Auto_Renew != 1] # Get rid of auto-renew data as the breed for these
                                                                 # does not change.
            self.data.reset_index(inplace=True)

            # --- Pre-calculations ---
            self.data.loc[:, "a1"]    = self.data.Payment_at_Purchase / self.data.Attribute_Price
            self.data.loc[:, "a2"]    = self.data.Attribute_Promotions * self.data.Inertia_for_Switch
            self.data.loc[:, "sg_ab"] = self.data.Social_Grade * self.data.Attribute_Brand

            for i in range(15):
                self.data.loc[:, "Y_" + str(i + 1)] = [0]        # This is the column that will keep track of the
                                                                 # of the mutations.
            return self.data
        except IOError as e:           # For example, there is no such file in the directory.
            print(e)
        except:
            print("Unexpected error")

    def generateBrandFactorVector(self, lowerBound, upperBound, size):
        return np.random.uniform(lowerBound, upperBound, size)

    def runModel(self):
        """Runs the agent-based simulation."""
        for i in range(15): self.calculateEpochs(i)

    def calculateEpochs(self, year):
        """Calculates annual simulation epochs.
        :param year: year for which the epoch is calculated.
        """
        if (year == 0):
            currentColumn = 'Agent_Breed'
            nextColumn    = 'Y_1'
        else:
            currentColumn = 'Y_'+str(year)
            nextColumn    = 'Y_'+str(year+1)

        breedColumns = {"currCol": currentColumn, "nextCol": nextColumn}

        #NB: groupby applies twice to the first group (see warning): http://pandas.pydata.org/pandas-docs/stable/groupby.html#flexible-apply
        #This is "to decide whether it can take a fast or slow code path".

        #Major improvements that can further decrease the processing time to below 320 milliseconds:
        # 1) do not calculate both Breed_C_Changed and then Breed_C_Unchanged. Can be done by implementing
        #a mock class variable with one variable and that variable would tell us whether we have already applied the
        #group by on a given breed.
        # 2) I have noticed that some policies have same inertias. Thus, pre-compute a1,a2.
        self.data = self.data.groupby(breedColumns["currCol"]).apply(lambda group: self.applyMutations(group,breedColumns))

    def applyMutations(self, groupDf, breedColumns):
        """
        :param groupDf: The dataframe group slice that is passed by the groupBy above.
        :param breedColumns: List. First element is the first column on which the mutation is performed. Second element
        is the mutated breed (next year).
        """
        assert isinstance(groupDf, pd.DataFrame), "%r is not a pd.DataFrame!" % (groupDf)

        breed = groupDf[breedColumns['currCol']].iloc[0]  # do the iloc.

        brand_factor = np.random.uniform(0.1,2.9,len(groupDf)) # simplifying assumption. Magic Numbers.
        rand = np.random.uniform(0, 3, len(groupDf))
        affinity = groupDf.a1 + rand * groupDf.a2

        # Applies the maps to both breed_c and breed_nc twice due to the fact that there are
        # (breed_c_changed and breed_c_unchaged) as well as (breed_nc_changed) and (breed_nc_unchanged).
        # To circumvent in the future, use the C++ mock pointer class from the above to keep track of
        # whether you have already updated the dataFrame. So you would have two Ref classes, one
        # for breed_c and one for breed_nc.
        if (breed.startswith("Breed_C")):

            maskSwitchToNC = (affinity < groupDf.sg_ab)
            groupDf.loc[:,breedColumns["nextCol"]] = maskSwitchToNC.map(self.breedMaps["breedCMap"])
            return groupDf

        elif (breed.startswith("Breed_NC")):

            maskSwitchToC = (affinity < brand_factor * groupDf.sg_ab)
            groupDf.loc[:,breedColumns["nextCol"]] = maskSwitchToC.map(self.breedMaps["breedNCMap"])
            return groupDf

        else:
            raise ValueError('Unknown breed: %s. Can start with "Breed_C" or "Breed_NC".' % (breed))

    def modelOutput(self):
        """Simple graphical output."""
        f = plt.figure(figsize=(15, 8))
        ax = f.add_subplot(111)

        cGained   = []
        cLost     = []
        cRegained = []

        regainedMap = {True:'Regained',False:'Not_Regained'}

        for i in range(15):
            cGained.append(len(self.data[self.data['Y_' + str(i + 1)] == 'Breed_C_Changed']))
            cLost.append(len(self.data[self.data['Y_' + str(i + 1)]   == 'Breed_NC_Changed']))

            if(i >= 2):
                if(i==2):
                    breedColumns = ['Agent_Breed','Y_1','Y_2']
                else:
                    breedColumns = ['Y_'+str(i-2),'Y_'+str(i-1),'Y_'+str(i)]

                self.data.loc[:,'Regained_'+str(i)] = self.data.apply(lambda row: \
                            self.regainingFilter(row,breedColumns),axis=1).map(regainedMap)
                cRegained.append(len(self.data[self.data['Regained_'+str(i)] == 'Regained']))

        self.data.loc[:,'Regained_15'] = self.data.apply(lambda row: \
                            self.regainingFilter(row,breedColumns),axis=1).map(regainedMap)
        cRegained.append(len(self.data[self.data['Regained_' + str(i)] == 'Regained']))

        x = np.arange(15) + 1
        regained_x = np.arange(14) + 2

        ax.plot(x, cGained, 'g--')
        ax.plot(x, cLost,   'r-')
        ax.plot(regained_x, cRegained, 'k-.')

        ax.legend(['# Breed_C Gained', '# Breed_C Lost', '# C Regained'], fontsize=20, loc='upper right')
        plt.show()

    def regainingFilter(self,row,breedColumns):
        # C-NC-C filter.
        if(row[breedColumns[0]].startswith("Breed_C") and row[breedColumns[1]].startswith("Breed_NC") \
                   and row[breedColumns[2]].startswith("Breed_C")):
            return True
        else:
            return False

class Ref:
    """
    C++ pointer simulator. One of the uses is the keeping in memory
    of the running updates.
    """
    def __init__(self, obj):        self.obj = obj
    def get(self):           return self.obj
    def set(self, obj):             self.obj = obj

if __name__ == "__main__":
    abs = AgentBasedSimulation()

    dataFilePath = "C:/Users/Nazariy/Desktop/Simudyne_Backend_Test.csv"
    saveFilePath = "C:/Users/Nazariy/Desktop/Simudyne_Bacend_Test_Output.csv"
    lb_BrandFactor = 0.1
    ub_BrandFactor = 2.9

    inputData = abs.processInputData(dataFilePath)
    inputDataSize = len(inputData)

    start = time.time()
    abs.runModel()
    end = time.time()

    print("It took: " + "{0:.4f}".format((end-start)*1000) + " milliseconds.")

    abs.modelOutput()

    try:
        abs.data.to_csv(saveFilePath,index=False)
    except Exception as e:
        print(e)