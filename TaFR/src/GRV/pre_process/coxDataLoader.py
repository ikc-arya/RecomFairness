# -*- coding: UTF-8 -*-
import logging
import argparse
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt

from pycox.models import CoxTime
import os

class coxDataLoader:
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='TaFR/data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='MIND-small',
                            help='Choose a dataset.')
        parser.add_argument('--prediction_dataset', type=str, default='MIND-small',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--play_rate', type=int, default=1,
                            help='in_features')
        parser.add_argument('--pctr', type=int, default=0,
                            help='in_features')
        parser.add_argument('--start_time', type=int, default=24,
                            help='group hour')
        return parser



    def __init__(self, args):
        self.start_time = args.start_time
        # Load preprocessed data
        dataFolder = os.path.join(args.path, args.dataset)
        if args.dataset.lower() == "mind-small":
            preprocessed_file = "MIND-small_train.csv"
        elif args.dataset.lower() == "mind":
            preprocessed_file = "MIND_train.csv"
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        preprocessed_path = os.path.join(args.path, "MIND", "preprocessed", preprocessed_file)
        if not os.path.exists(preprocessed_path):
            raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_path}")
        
        # Load the CSV file with proper parsing
        self.coxData = pd.read_csv(preprocessed_path, sep=",", quotechar='"', escapechar="\\")
        print(f"Loaded dataset: {args.dataset}, shape: {self.coxData.shape}")

        # Rename columns for consistency
        renameDict = {'news_id': 'photo_id', 'clicked': 'click_rate'}
        for i in range(168):
            renameDict[f'ctr{i}'] = f'click_rate{i}'
        self.coxData.rename(columns=renameDict, inplace=True)
        print(f"Renamed columns: {self.coxData.columns.tolist()}")  # Print column names as a list

    def filtered_data(self):
        caredList=['photo_id']
        for i in range(self.start_time):
            caredList.append('click_rate%d'%(i))
            if self.play_rate:
                caredList.append('play_rate%d'%(i))
            if self.pctr:
                caredList.append('new_pctr%d'%(i))
        self.coxData=self.coxData[caredList]


    def load_data(self,args):
        # self.filtered_data()
        self.diedInfo=pd.read_csv(args.label_path)
        df_train=self.coxData
        df_train.fillna(-1,inplace=True)
        caredList=['died','timelevel','photo_id']
        for i in range(self.start_time):
            caredList.append('click_rate%d'%(i))
            if self.play_rate:
                caredList.append('play_rate%d'%(i))
            if self.pctr:
                caredList.append('new_pctr%d'%(i))
        df_train=df_train[caredList[2:]] 
        df_train=pd.merge(df_train,self.diedInfo[['photo_id','died','timelevel']],on='photo_id')
        logging.info('[coxDataLoader] before died filter %d items'%len(df_train))
        df_train=df_train[df_train['died']==1]
        logging.info(df_train[['died','timelevel']].describe())
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)

        if self.prediction_dataset!='':
            x=args.label_path
            # prediction_label_path=x.replace(args.dataset,args.prediction_dataset)
            # diedInfo=pd.read_csv(prediction_label_path)
            coxData=pd.read_csv(args.path+args.prediction_dataset+'/cox.csv')
            coxData.fillna(-1,inplace=True)
            caredList=['died','timelevel','photo_id']
            for i in range(self.start_time):
                caredList.append('click_rate%d'%(i))
                if self.play_rate:
                    caredList.append('play_rate%d'%(i))
                if self.pctr:
                    caredList.append('new_pctr%d'%(i))
            coxData=coxData[caredList[2:]] 
            df_test=coxData
            print("********",len(df_test))
            df_test['died']=1
            df_test['timelevel']=168
            # df_test=pd.merge(coxData,diedInfo[['photo_id','died','timelevel']],on='photo_id')
            # print("length!!!",len(diedInfo),len(coxData),len(df_test))

        
        return df_train,df_val,df_test,caredList


    def preprocess(self,args):
        df_train,df_val,df_test,cared=self.load_data(args)
        print(len(df_train),len(df_val),len(df_test))
        ignore_length=3
        cols_standardize = cared[ignore_length:]
        # cols_leave = ['photo_id']
        #scaler.mean_,np.sqrt(scaler.var_)
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        # leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize)#+leave

        self.x_train = x_mapper.fit_transform(df_train).astype('float32')
        if len(df_val)>0:
            x_val = x_mapper.transform(df_val).astype('float32')
        self.x_test = x_mapper.transform(df_test).astype('float32')
        self.df_test=df_test

        self.labtrans = CoxTime.label_transform()
        get_target = lambda df: (df['timelevel'].values, df['died'].values)
        self.y_train = self.labtrans.fit_transform(*get_target(df_train))
        y_val = self.labtrans.transform(*get_target(df_val))
        self.val = tt.tuplefy(x_val, y_val)
        self.durations_test, self.events_test = get_target(df_test)
        # self.y_test= self.corpus.labtrans(*get_target(df_test))

    
if __name__ == '__main__':
    print("[dataLoader]")
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = coxDataLoader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    # args.path = '../../data/'
    corpus = coxDataLoader(args)
    # label=Label(corpus)
    # corpus.preprocess()

