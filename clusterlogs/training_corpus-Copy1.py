#!/usr/bin/python
import sys
#import getopt
from gensim.models import Word2Vec
from time import time
from pyonmttok import Tokenizer
import json
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType,ArrayType
from pyspark.sql.functions import col,udf,struct,collect_list
from pyspark import SparkContext
from pyspark.sql import SparkSession
import csv
import re
from pyspark.sql.functions import col, lower, count
import numpy as np
from datetime import date, tiemdelta

def spark_context(appname='cms', yarn=None, verbose=False, python_files=[]):
        # define spark context, it's main object which allow
        # to communicate with spark
    if  python_files:
        return SparkContext(appName=appname, pyFiles=python_files)
    else:
        return SparkContext(appName=appname)

def spark_session(appName="log-parser"):
    """
    Function to create new spark session
    """
    sc = SparkContext(appName="log-parser")
    return SparkSession.builder.config(conf=sc._conf).getOrCreate()  

@udf(returnType=StringType()) 
def clean_message(message):
    import re
    message = re.sub(r'\S+\.\S+', ' ', message)  # any URL
    message = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', ' ', message) # remove all substrings with digits
    message = re.sub(r'(\d+)', ' ', message) # remove all other digits
    message = re.sub(r'[^\w\s]', ' ', message) # removes all punctuation
    message = re.sub(r' +', r' ', message)
    message=message.lower()
    return message

def tokenize_message(message, tokenizer_type, spacer_annotate, preserve_placeholders,spacer_new):
    tokenizer = Tokenizer(tokenizer_type, spacer_annotate=spacer_annotate, preserve_placeholders= preserve_placeholders, spacer_new=spacer_new)
    return tokenizer.tokenize(message)[0]


class uniqueMex(object):
    
    
    def __init__(self,spark,days):
        
        self.spark=spark
        self.hdir='hdfs:///project/monitoring/archive/fts/raw/complete'
        self.days=days        
            
    def fts_messages(self,verbose=False):
        """
        Parse fts HDFS records
        """
       #clean_mex_udf=udf(lambda row: clean_message(x) for x in row, StringType()) #user defined function to clean spark dataframe
        clean_mex_udf=udf(lambda x: clean_message(x), StringType())
        self.spark.udf.register('clean_mex_udf',clean_mex_udf)
        hpath = [('%s/%s' % (self.hdir,iDate)) for iDate in self.days]
        # create new spark DataFrame
        schema = StructType([StructField('data', StructType([StructField('t__error_message', StringType(), nullable=True)]))])
        df=self.spark.read.json(hpath, schema)
        df=df.select(col('data.t__error_message').alias('error_message')).where('error_message <> ""')
        df.cache()
        bf_n=df.count()
        print('before cleaning %i messages'% bf_n)
        print('...cleaning messages')
        #df=df.withColumn('error_message', clean_mex_udf(struct(df['error_message']))).dropDuplicates()
        df=df.withColumn('error_message', clean_message(col('error_message'))).dropDuplicates()
        af_n=df.count()
        print('after cleaning %i different messages'% af_n)
        #df.show()
        return df,bf_n,af_n
   

    
class MyCorpus(object):
    
    """An interator that yields sentences (lists of str)."""
    
    def __init__(self,inputDf):
        self.inputDf=inputDf
        self.list_err=self.inputDf.select(collect_list("error_message")).collect()[0][0]      
    
    def __iter__(self):       
                     
        for line in self.list_err:
            tokenized=tokenize_message(line, 'space',False,True,False)
            yield tokenized
        
                     
                    

def main(argv):
    
    spark = spark_session()
        
    start_date=sys.argv[1]
    end_date=sys.argv[2]
    
    days=[]
    dd=pd.date_range(start=start_date,end=end_date)
    for day in dd:
        days.append((day.date().strftime("%y/%m/%d")))        
        fts,bf_n,af_n=uniqueMex(spark,days).fts_messages() #bf_n and af_n number of messages
        start_time = time()
        tokenized = MyCorpus(fts)
        tot_time=time() - start_time
        print("--- time to tokenize corpus: %f seconds ---" % tot_time)
        with open('unique_mex.csv', mode='a',newline='') as tFile:
            file_writer = csv.writer(tFile)
            file_writer.writerow([len(days),bf_n,af_n,tot_time])
#     print('...starting training')
#     try:
#         start_time = time()
#         model = Word2Vec(sentences=tokenized,compute_loss=True,size=300,window=7, min_count=1, workers=4, iter=30)
#         tot_time=time() - start_time
#         print("--- %f seconds ---" % tot_time)
#         loss=model.get_latest_training_loss()
#         print('latest training loss:',loss)
#         with open('training_parameters.csv', mode='a',newline='') as tFile:
#             file_writer = csv.writer(tFile)
#             file_writer.writerow([bf_n,af_n,loss,tot_time])
#         model.save(outputfile)
#         print('Training has finished. Model saved in file. Thanks for coming :)')
#     except Exception as e:
#         print('Training model error:', e)
   
   

if __name__ == "__main__":
    main(sys.argv[1:]) # get everything after the script name
   