from pyspark import keyword_only
from pyspark.ml import Transformer

from pyspark.ml.param.shared import (
    Param, Params, TypeConverters, HasOutputCols, HasInputCols
)
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml import Estimator
from pyspark.sql import DataFrame
from pyspark.sql.functions import desc
from pyspark.sql.functions import col, abs
from typing import List
from pyspark.sql.types import TimestampType, LongType
from finance_complaint.logger import logger
from finance_complaint.config.spark_manager import spark_session


class DerivedFeatureGenerator(Transformer, HasInputCols, HasOutputCols,
                              DefaultParamsReadable, DefaultParamsWritable):
    
    @keyword_only
    def __init__(self, inputCols: List[str] = None, outputCols: List[str] = None):
        super(DerivedFeatureGenerator, self).__init__()
        kwargs = self._input_kwargs
        
        self.second_within_day = 60 * 60 * 24
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols: List[str] = None, outputCols: List[str] = None, ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, value: List[str]):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def setOutputCols(self, value: List[str]):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCols=value)
    
    def _fit(self, dataframe: DataFrame):
        return self
    
    def _transform(self, dataframe: DataFrame):
        input_cols = self.getInputCols()
        
        for column in input_cols:
            dataframe = dataframe.withColumn(column,
                                             col(column).cast(TimestampType()))
        
        dataframe = dataframe.withColumn(self.getOutputCols()[0], abs(
            col(input_cols[1]).cast(LongType()) - col(input_cols[0]).cast(LongType())) / (
                                             self.second_within_day))
        return dataframe


class FrequencyEncoder(Estimator, HasInputCols, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable):
    # Define a parameter for storing frequency information.
    frequencyInfo = Param(Params._dummy(), "getfrequencyInfo", "getfrequencyInfo", typeConverter=TypeConverters.toList)

    @keyword_only
    def __init__(self, inputCols: List[str] = None, outputCols: List[str] = None):
        super(FrequencyEncoder, self).__init__()
        kwargs = self._input_kwargs

        # Create a Param instance for frequencyInfo.
        self.frequencyInfo = Param(self, "frequencyInfo", "")
        self._setDefault(frequencyInfo="")  # Set default value.
        # self._set(**kwargs)

        self.setParams(**kwargs)  # Set parameters using the provided keyword arguments.

    # Setter method for frequencyInfo parameter.
    def setfrequencyInfo(self, frequencyInfo: list):
        return self._set(frequencyInfo=frequencyInfo)

    # Getter method for frequencyInfo parameter.
    def getfrequencyInfo(self):
        return self.getOrDefault(self.frequencyInfo)

    @keyword_only
    def setParams(self, inputCols: List[str] = None, outputCols: List[str] = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)  # Set parameters using the provided keyword arguments.

    # Setter method for inputCols parameter.
    def setInputCols(self, value: List[str]):
        return self._set(inputCol=value)

    # Setter method for outputCols parameter.
    def setOutputCols(self, value: List[str]):
        return self._set(outputCols=value)

    # Fit method for the FrequencyEncoder estimator.
    def _fit(self, dataframe: DataFrame):
        input_columns = self.getInputCols()
        output_columns = self.getOutputCols()
        replace_info = []

        # Calculate and collect frequency information for each input column.
        for column, new_column in zip(input_columns, output_columns):
            freq = (dataframe.select(col(column).alias(f'g_{column}'))
                    .groupby(f'g_{column}').count().withColumn(new_column, col('count')))
            freq = freq.drop('count')
            logger.info(f"{column} has [{freq.count()}] unique category")
            replace_info.append(freq.collect())

        # Set the calculated frequency information.
        self.setfrequencyInfo(frequencyInfo=replace_info)

        # Create a FrequencyEncoderModel instance and set its parameters.
        estimator = FrequencyEncoderModel(inputCols=input_columns, outputCols=output_columns)
        estimator.setfrequencyInfo(frequencyInfo=replace_info)
        return estimator


class FrequencyEncoderModel(FrequencyEncoder, Transformer):

    def __init__(self, inputCols: List[str] = None, outputCols: List[str] = None):
        super(FrequencyEncoderModel, self).__init__(inputCols=inputCols, outputCols=outputCols)

    # Transformation logic for the FrequencyEncoderModel.
    def _transform(self, dataframe: DataFrame):
        inputCols = self.getInputCols()
        outputCols = self.getOutputCols()
        freqInfo = self.getfrequencyInfo()

        # Loop through input and output columns along with their respective frequency information.
        for in_col, out_col, freq_info in zip(inputCols, outputCols, freqInfo):
            frequency_dataframe: DataFrame = spark_session.createDataFrame(freq_info)
            columns = frequency_dataframe.columns

            # Join the frequency information with the original dataframe based on input column.
            dataframe = dataframe.join(frequency_dataframe, on=dataframe[in_col] == frequency_dataframe[columns[0]])

            # Drop the extra column used for joining.
            dataframe = dataframe.drop(columns[0])

            # Add the frequency information to the output column.
            if out_col not in dataframe.columns:
                dataframe = dataframe.withColumn(out_col, col(columns[1]))
                dataframe = dataframe.drop(columns[1])

        return dataframe    

class FrequencyImputer(
    Estimator, HasInputCols, HasOutputCols,
    DefaultParamsReadable, DefaultParamsWritable):
    # Define a parameter to store top categories.
    topCategorys = Param(Params._dummy(), "getTopCategorys", "getTopCategorys", typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCols: List[str] = None, outputCols: List[str] = None):
        super(FrequencyImputer, self).__init__()
        self.topCategorys = Param(self, "topCategorys", "")
        self._setDefault(topCategorys="")
        kwargs = self._input_kwargs
        print(kwargs)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols: List[str] = None, outputCols: List[str] = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # Setter method for topCategorys parameter.
    def setTopCategorys(self, value: List[str]):
        return self._set(topCategorys=value)

    # Getter method for topCategorys parameter.
    def getTopCategorys(self):
        return self.getOrDefault(self.topCategorys)

    # Setter method for inputCols parameter.
    def setInputCols(self, value: List[str]):
        return self._set(inputCols=value)

    # Setter method for outputCols parameter.
    def setOutputCols(self, value: List[str]):
        return self._set(outputCols=value)

    # Fit method for the FrequencyImputer estimator.
    def _fit(self, dataset: DataFrame):
        inputCols = self.getInputCols()
        topCategorys = []

        # Identify the top category for each input column.
        for column in inputCols:
            categoryCountByDesc = dataset.groupBy(column).count().filter(f'{column} is not null').sort(desc('count'))
            topCat = categoryCountByDesc.take(1)[0][column]
            topCategorys.append(topCat)

        # Set the calculated top categories.
        self.setTopCategorys(value=topCategorys)

        # Create a FrequencyImputerModel instance and set its parameters.
        estimator = FrequencyImputerModel(inputCols=self.getInputCols(), outputCols=self.getOutputCols())
        estimator.setTopCategorys(value=topCategorys)
        return estimator


class FrequencyImputerModel(FrequencyImputer, Transformer):

    def __init__(self, inputCols: List[str] = None, outputCols: List[str] = None):
        super(FrequencyImputerModel, self).__init__(inputCols=inputCols, outputCols=outputCols)

    # Transformation logic for the FrequencyImputerModel.
    def _transform(self, dataset: DataFrame):
        topCategorys = self.getTopCategorys()
        outputCols = self.getOutputCols()

        # Create a dictionary to update missing values.
        updateMissingValue = dict(zip(outputCols, topCategorys))

        inputCols = self.getInputCols()

        # Fill missing values in output columns with top categories.
        for outputColumn, inputColumn in zip(outputCols, inputCols):
            dataset = dataset.withColumn(outputColumn, col(inputColumn))

        dataset = dataset.na.fill(updateMissingValue)

        return dataset