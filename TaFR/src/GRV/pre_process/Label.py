# -*- coding: UTF-8 -*-
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pre_process import coxDataLoader
import logging
import argparse
import pandas as pd
from pre_process.coxDataLoader import *
import numpy as np
import os

class Label:
    def parse_data_args(parser):
        parser.add_argument('--label_path', type=str, default='',
                            help='died info csv file')
        parser.add_argument('--agg_hour', type=int, default=1,
                            help='group hour')
        parser.add_argument('--end_time', type=int, default=24*7,
                            help='group hour')
        parser.add_argument('--EXP_hazard', type=float, default=0.5,
                            help='group hour')
        parser.add_argument('--noEXP_hazard', type=float, default=0.5,
                            help='group hour')
        parser.add_argument('--acc_thres', type=float, default=-3,
                            help='group hour')
        parser.add_argument('--prepareLabel', type=int, default=0,
                            help='prepareDied')
        parser.add_argument('--exposed_duration',type=int,default=0)
        return parser

      
    def __init__(self,args,corpus): #dataLoader
        self.agg_hour=args.agg_hour
        self.end_time=args.end_time
        self.EXP_hazard=args.EXP_hazard
        self.noEXP_hazard=args.noEXP_hazard
        self.acc_thres=args.acc_thres
        self.prepareLabel=args.prepareLabel
        self.exposed_duration=args.exposed_duration
        if args.label_path=='':
            log_args = [args.dataset,\
                str(self.agg_hour),str(corpus.start_time),str(self.end_time),\
                str(self.EXP_hazard),str(self.noEXP_hazard),str(self.acc_thres)]
            log_file_name = '__'.join(log_args).replace(' ', '__')
            if args.label_path == '':
                if args.exposed_duration:
                    args.label_path = '../label/{}_v2.csv'.format(log_file_name)
                else:
                    args.label_path = '../label/{}.csv'.format(log_file_name)
        self.label_path=args.label_path
        print("label path",self.label_path)
        if not os.path.exists(self.label_path) or self.prepareLabel:
            self.prepareDied(args,corpus)
        # else:
        #     self.prepareDied(corpus)
        return
    
    def read_all(self, args):
        dataFolder = os.path.join(args.path, args.dataset)
        if args.dataset == "MIND":
            # Load preprocessed MIND data
            df = pd.read_csv(os.path.join(dataFolder, "preprocessed", "MIND_preprocessed.csv"))
            df.rename(columns={"photo_id": "item_id", "click_rate": "is_click"}, inplace=True)
            df["new_pctr"] = df["pctr"]  # Use historical CTR as pCTR
        elif args.dataset == "kwai":
            df = pd.read_csv(f"{dataFolder}/{args.dataset}_10F_167H_hourLog.csv")
            df.rename({"expose_hour": 'timelevel', 'is_click': 'click_rate', 'new_pctr': 'pctr'}, axis=1, inplace=True)
        else:
            raise ValueError("Unsupported dataset")
        
        # Filter data by timelevel
        df = df[df['timelevel'] < self.end_time].copy()
        return df

    def prepareDied(self, args, corpus):
        # Identify new items (20% latest uploads in MIND)
        if args.dataset.lower() == "mind":
            new_item_threshold = corpus.coxData["timestamp"].quantile(0.8)
            new_items = corpus.coxData[corpus.coxData["timestamp"] > new_item_threshold]["photo_id"].unique()
        elif args.dataset.lower() == "mind-small":
            new_items = corpus.coxData.sample(frac=0.2)["photo_id"].unique()
            print(corpus.coxData.columns)
            print(corpus.coxData.sample(frac=0.2).head())
        
        # Track exposed new items
        exposed_new_items = all_predictions[all_predictions["itemID"].isin(new_items)]["itemID"].nunique()

        def getFlag(v,t):
            if t<corpus.start_time:
                return 0
            return v-self.EXP_hazard+self.noEXP_hazard #min(v-self.EXP_hazard,0) #
        
        hourInfo['riskFlag']=hourInfo.apply(lambda v:getFlag(v['riskRank'],v['timelevel']),axis=1)
        logging.info(hourInfo['riskFlag'].describe())
        # hourInfo.sort_values(['photo_id','timelevel'],inplace=True)

        #type=0
        index_list=[]
        ids_list=[]
        time_list=[]
        for id in hourInfo['photo_id'].unique():
            for time in range(corpus.start_time,self.end_time):
                index_list.append("%d-%d"%(id,time))
                ids_list.append(id)
                time_list.append(time)
        new_system=pd.DataFrame({'photo_id':ids_list,'timelevel':time_list,'tag':index_list})
        hourInfo['tag']=hourInfo['photo_id'].astype('str')+'-'+hourInfo['timelevel'].astype('str')
        new_system=pd.merge(new_system,hourInfo[['tag','riskFlag']],on='tag',how='left')
        new_system.fillna(0,inplace=True)
        new_system.sort_values(['photo_id','timelevel'],inplace=True)
        new_system['risk']=new_system.groupby('photo_id')['riskFlag'].cumsum()
        new_system['risk']=new_system['risk']-(new_system['timelevel']-corpus.start_time)*self.noEXP_hazard

        # new_system.to_csv(self.label_path,index=False)
        # print("finish labeling")
        # return

        died=new_system[(new_system['risk']<=self.acc_thres)&(new_system['timelevel']>=corpus.start_time)].copy()
        died['died']=1
        died.sort_values('timelevel',inplace=True)
        died=died.groupby('photo_id').head(1)

        remain=new_system[~new_system['photo_id'].isin(died['photo_id'].tolist())].copy()
        remain['died']=0
        remain.sort_values('timelevel',inplace=True)
        remain=remain.groupby('photo_id').tail(1)

        item=pd.concat([remain,died])
        item.to_csv(self.label_path,index=False)
        logging.info(item.describe())

        return